// Render-test harness: loads the real dashboard HTML, mocks fetch with API-shaped
// data, runs the JS in jsdom, and asserts the UI actually renders (no "loading…",
// 5 system cards present, tabs work). Catches syntax AND runtime errors before deploy.
const fs = require('fs');
// jsdom path: prefer /tmp/jsdom_pkg install, fall back to /tmp/node_modules legacy path
let jsdomMod; try { jsdomMod=require('/tmp/jsdom_pkg/node_modules/jsdom'); } catch(e) { jsdomMod=require('/tmp/node_modules/jsdom'); }
const { JSDOM, VirtualConsole } = jsdomMod;

const HTML = fs.readFileSync(__dirname + '/../reporting/dashboard/templates/paper_dashboard.html', 'utf8');

// Inline mock data — self-contained (no dependency on /tmp/api_*.json being fresh)
const status = {
  as_of: '2026-09-04', risk_on: true, breadth: 0.61, mode: 'PAPER',
  market_status: 'closed', prices_asof: '2026-09-13', updated: '2026-09-13 19:05:00',
  invested_pct: 100, day_pnl_pct: 0.42, overall_pnl_pct: 1.23,
  overall_pnl_rs: 2460, momentum_picks: ['STOCK1','STOCK2','STOCK3'],
  portfolio: [],
  systems: [
    {id:'s1',name:'System 1 · Momentum',desc:'Medium risk · balanced multi-asset',
     nav:203000,overall_pct:1.5,day_pct:0.42,maxdd:-2.1,gross:100,borrow:0,
     positions:[{label:'Momentum stocks',weight:70},{label:'Gold (GOLDBEES)',weight:15},{label:'US Nasdaq (MON100)',weight:15}],
     momentum_picks:['STOCK1','STOCK2']},
    {id:'s2',name:'System 2 · Survivorship',desc:'Low risk · barbell, sleeps well',
     nav:201000,overall_pct:0.5,day_pct:0.15,maxdd:-0.8,gross:100,borrow:30,
     positions:[{label:'Arbitrage fund (safe)',weight:70},{label:'Momentum stocks',weight:30}],
     momentum_picks:['STOCK1','STOCK2']},
    {id:'s3',name:'System 3 · Aggressive',desc:'High risk · 2× leverage',
     nav:206000,overall_pct:3.0,day_pct:0.84,maxdd:-4.2,gross:200,borrow:100,
     positions:[{label:'Momentum stocks',weight:140},{label:'Gold (GOLDBEES)',weight:30},{label:'US Nasdaq (MON100)',weight:30}],
     momentum_picks:['STOCK1','STOCK2']},
    {id:'s4',name:'System 4 · Pure (Regime)',desc:'Regime-gated · pure MID-cap momentum',
     nav:204000,overall_pct:2.0,day_pct:0.50,maxdd:-3.1,gross:100,borrow:0,
     positions:[{label:'Momentum stocks',weight:100}],
     momentum_picks:['A1','A2','A3','A4','A5']},
    {id:'s5',name:'System 5 · Pure (Always)',desc:'Always invested · pure MID-cap momentum',
     nav:205000,overall_pct:2.5,day_pct:0.55,maxdd:-3.8,gross:100,borrow:0,
     positions:[{label:'Momentum stocks',weight:100}],
     momentum_picks:['A1','A2','A3','A4','A5']},
  ]
};
const perf = {
  dates:['2026-09-04','2026-09-07','2026-09-08'],
  inception:'2026-09-04', days:3,
  series:{
    s1_nav:[200000,201000,203000],s2_nav:[200000,200500,201000],s3_nav:[200000,202000,206000],
    s4_nav:[200000,201500,204000],s5_nav:[200000,202000,205000],
    nifty50_nav:[200000,200800,201600],nifty500_nav:[200000,200700,201400]
  },
  stats:{
    s1_nav:{ret:1.5,value:203000,maxdd:-2.1},s2_nav:{ret:0.5,value:201000,maxdd:-0.8},
    s3_nav:{ret:3.0,value:206000,maxdd:-4.2},s4_nav:{ret:2.0,value:204000,maxdd:-3.1},
    s5_nav:{ret:2.5,value:205000,maxdd:-3.8},
    nifty50_nav:{ret:0.8,value:201600,maxdd:-1.0},nifty500_nav:{ret:0.7,value:201400,maxdd:-0.9}
  }
};
const activity = {monitor:['2026-09-13 19:00:00 | EOD tracked','2026-09-13 19:01:00 | done'],signal:[]};

let errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push('jsdomError: ' + (e.detail || e.message)));

const dom = new JSDOM(HTML, {
  runScripts: 'dangerously',
  virtualConsole: vc,
  beforeParse(window) {
    window.Chart = function(){ return { destroy(){} }; };
    window.fetch = (url) => {
      let body = {};
      if (url.includes('/api/status')) body = status;
      else if (url.includes('/api/performance')) body = perf;
      else if (url.includes('/api/activity')) body = activity;
      return Promise.resolve({ json: () => Promise.resolve(body) });
    };
    window.console.error = (...a) => errors.push('console.error: ' + a.map(String).join(' '));
  }
});

// let async fetches + render resolve
setTimeout(() => {
  const doc = dom.window.document;
  const sub = doc.getElementById('sub').textContent;
  const cards = doc.querySelectorAll('#syscards .scard').length;
  const fail = [];

  if (errors.length) fail.push('JS ERRORS:\n   ' + errors.join('\n   '));
  if (sub.includes('loading')) fail.push('subtitle stuck on "loading…" — JS never completed');
  if (sub.includes('error'))   fail.push('subtitle shows error: ' + sub);
  if (cards !== 5)             fail.push(`expected 5 system cards, got ${cards}`);

  // verify all 5 system tabs render holdings
  ['s1','s2','s3','s4','s5'].forEach(id => {
    const html = (doc.getElementById('view-'+id)||{}).innerHTML || '';
    if (!html.includes('Holdings')) fail.push(`System ${id.toUpperCase()} tab did not render holdings`);
  });

  // verify S4/S5 picks appear in the rendered tab
  const s4html = (doc.getElementById('view-s4')||{}).innerHTML || '';
  const s5html = (doc.getElementById('view-s5')||{}).innerHTML || '';
  if (!s4html.includes('A1')) fail.push('S4 momentum picks not displayed');
  if (!s5html.includes('A1')) fail.push('S5 momentum picks not displayed');

  // verify overview chart has 7 datasets' worth of color entries in the cmp section
  const cmpBoxes = doc.querySelectorAll('#cmp-o .b').length;
  if (cmpBoxes !== 7) fail.push(`expected 7 comparison boxes (5 systems + 2 benchmarks), got ${cmpBoxes}`);

  // verify activity log renders something
  const actHTML = doc.getElementById('activity').innerHTML || '';
  if (!actHTML.includes('EOD')) fail.push('activity log did not render any entries');

  if (fail.length) {
    console.log('❌ DASHBOARD RENDER TEST FAILED');
    fail.forEach(f => console.log('  • ' + f));
    process.exit(1);
  } else {
    console.log('✅ DASHBOARD RENDER TEST PASSED');
    console.log(`   subtitle: "${sub}"`);
    console.log(`   system cards: ${cards} | all 5 tabs OK | cmp boxes: ${cmpBoxes}`);
    process.exit(0);
  }
}, 800);
