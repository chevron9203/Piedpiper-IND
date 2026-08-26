"""
Nifty Futures SHORT — Intraday Bear Day Backtest
=================================================
Strategy:
  - Signal: Previous day Nifty close < EMA20 → bear day
  - Entry : SHORT 1 Nifty lot (50 units) at today's OPEN
  - Exit  : at today's CLOSE (intraday, no overnight hold)
  - Signal flip LONG: on bull days SHORT is flat, no trade

Data: Nifty 50 daily from EOD2 DuckDB (real NSE data, 2015-2026)
Costs: ₹40 brokerage round-trip + 0.01% STT (futures sell) + exchange

This is a daily-bar simulation of an intraday futures trade.
Entry at open and exit at close is a valid approximation because
  a Nifty futures lot tracks the index closely intraday.
"""
import sys
sys.path.insert(0, '/Users/abc/Documents/ozi-code/piedpiper')

import pandas as pd
import numpy as np
from storage import store

# ── Constants ─────────────────────────────────────────────────────────────────
CAPITAL         = 500_000     # ₹5L total capital
LOT_SIZE        = 50          # 1 Nifty lot = 50 units
MARGIN_PER_LOT  = 130_000     # ~₹1.3L margin per lot (SPAN + exposure)
BROKERAGE       = 40          # round trip flat brokerage (₹20 each side)
STT_RATE        = 0.0001      # 0.01% on sell side for futures
EXCHANGE_RATE   = 0.0000335   # NSE exchange fee
SEBI_RATE       = 0.000001    # SEBI fee
GST_RATE        = 0.18        # on brokerage + exchange fee
EMA_PERIOD      = 20
MAX_LOTS        = 1           # start with 1 lot, conservative

store.init_schema()

# ── Load Nifty 50 daily data ───────────────────────────────────────────────────
with store.db_conn() as conn:
    rows = conn.execute("""
        SELECT dt, open, high, low, close
        FROM adjusted_ohlcv
        WHERE symbol = 'Nifty 50'
        ORDER BY dt
    """).fetchall()

df = pd.DataFrame(rows, columns=['dt', 'open', 'high', 'low', 'close'])
df['dt'] = pd.to_datetime(df['dt'])
df = df.set_index('dt').sort_index()
df = df[df.index >= '2015-01-01']
print(f"Nifty 50 daily data: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")

# ── EMA and signal ────────────────────────────────────────────────────────────
df['ema20'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
# Signal based on PREVIOUS day's close vs EMA20
df['prev_close'] = df['close'].shift(1)
df['prev_ema20'] = df['ema20'].shift(1)
df['bear_day']   = df['prev_close'] < df['prev_ema20']  # today is bear day
df['bull_day']   = df['prev_close'] >= df['prev_ema20']

# ── Cost calculation ──────────────────────────────────────────────────────────
def calc_cost(entry_price, lots=1):
    notional  = entry_price * LOT_SIZE * lots
    stt       = notional * STT_RATE         # only on sell side
    exch      = notional * EXCHANGE_RATE * 2  # both sides
    sebi      = notional * SEBI_RATE * 2
    brok      = BROKERAGE
    gst       = (brok + exch) * GST_RATE
    return brok + stt + exch + sebi + gst

# ── Backtest loop ─────────────────────────────────────────────────────────────
trades = []
df_valid = df.dropna()

for dt, row in df_valid.iterrows():
    if not row['bear_day']:
        continue
    entry  = row['open']
    exit_p = row['close']
    lots   = MAX_LOTS
    cost   = calc_cost(entry, lots)
    
    # SHORT: profit when price goes DOWN
    gross_pnl = (entry - exit_p) * LOT_SIZE * lots
    net_pnl   = gross_pnl - cost
    
    trades.append({
        'dt'       : dt,
        'entry'    : entry,
        'exit'     : exit_p,
        'direction': 'SHORT',
        'lots'     : lots,
        'gross_pnl': gross_pnl,
        'cost'     : cost,
        'net_pnl'  : net_pnl,
        'day_move_pct': (exit_p - entry) / entry * 100,
    })

# Also run LONG on bull days for comparison
trades_long = []
for dt, row in df_valid.iterrows():
    if not row['bull_day']:
        continue
    entry  = row['open']
    exit_p = row['close']
    lots   = MAX_LOTS
    cost   = calc_cost(entry, lots)
    gross_pnl = (exit_p - entry) * LOT_SIZE * lots
    net_pnl   = gross_pnl - cost
    trades_long.append({
        'dt': dt, 'entry': entry, 'exit': exit_p, 'gross_pnl': gross_pnl,
        'cost': cost, 'net_pnl': net_pnl
    })

def print_stats(trades_list, label, capital, years=11.58):
    if not trades_list:
        print(f"{label}: No trades"); return
    t = pd.DataFrame(trades_list)
    total_net   = t['net_pnl'].sum()
    total_gross = t['gross_pnl'].sum()
    winners     = t[t['net_pnl'] > 0]
    losers      = t[t['net_pnl'] < 0]
    final_eq    = capital + total_net
    cagr        = (final_eq / capital) ** (1/years) - 1
    
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  Trades    : {len(t)}")
    print(f"  Win rate  : {len(winners)/len(t)*100:.1f}%")
    print(f"  Gross P&L : ₹{total_gross:+,.0f}")
    print(f"  Total cost: ₹{t['cost'].sum():,.0f}")
    print(f"  Net P&L   : ₹{total_net:+,.0f}")
    print(f"  CAGR      : {cagr*100:.1f}%  (on ₹{capital:,.0f} capital, {years:.1f}y)")
    print(f"  Avg win   : ₹{winners['net_pnl'].mean():+,.0f}" if len(winners) else "  No winners")
    print(f"  Avg loss  : ₹{losers['net_pnl'].mean():+,.0f}"  if len(losers) else "  No losers")
    
    # Year by year
    t['year'] = pd.to_datetime(t['dt']).dt.year
    yearly = t.groupby('year')['net_pnl'].sum()
    print(f"\n  Year-by-year:")
    for yr, pnl in yearly.items():
        bar  = '█' * min(int(abs(pnl)/5000), 30)
        sign = '+' if pnl >= 0 else ''
        print(f"    {yr}: ₹{pnl:>+10,.0f}  {bar}")
    
    # Max drawdown (cumulative)
    cum = t.sort_values('dt')['net_pnl'].cumsum()
    roll_max = cum.cummax()
    dd = cum - roll_max
    print(f"\n  Max drawdown: ₹{dd.min():,.0f}  ({dd.min()/capital*100:.1f}%)")

print_stats(trades,      "Nifty Futures SHORT  (1 lot, intraday, bear days)", CAPITAL)
print_stats(trades_long, "Nifty Futures LONG   (1 lot, intraday, bull days)", CAPITAL)

# Regime breakdown
total_days = len(df_valid)
bear_days  = df_valid['bear_day'].sum()
bull_days  = df_valid['bull_day'].sum()
print(f"\n  Regime: {bull_days} bull days ({bull_days/total_days*100:.0f}%) | {bear_days} bear days ({bear_days/total_days*100:.0f}%)")
print(f"  Avg Nifty move on bear days: {df_valid[df_valid['bear_day']]['close'].sub(df_valid[df_valid['bear_day']]['open']).div(df_valid[df_valid['bear_day']]['open']).mean()*100:.3f}%")
print(f"  Avg Nifty move on bull days: {df_valid[df_valid['bull_day']]['close'].sub(df_valid[df_valid['bull_day']]['open']).div(df_valid[df_valid['bull_day']]['open']).mean()*100:.3f}%")
