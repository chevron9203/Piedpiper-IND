"""
Do the three "chase less noise" parameter moves COMBINE, or do they overlap?

The single-parameter sweep (research_robustness.py) showed that more positions, a
looser rank buffer, and slower lookbacks each independently lower timing-luck sd and
hold or improve OOS. Those could be three views of one effect, or three additive ones --
single-parameter results cannot tell you which, because interactions are invisible
when you move one knob at a time.

Finalists are run at 21 offsets (not 7) so the error bar is ~2.07/sqrt(21) ~= 0.45pp
and differences of ~1pp become readable instead of noise.

Run:  python scripts/research_combo.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
from scripts.research_tranching import build_panel, stats, seg, STEP
from scripts import research_robustness as RB

BASE = Path(__file__).parent.parent
N_OFF = 21


def eval_full(C, turn_sm, start, label, **kw):
    """Every offset, so the mean is precise enough to compare configs ~1pp apart."""
    p = {**RB.DEFAULTS, **kw}
    feats = RB.make_feats(C, p["looks"], p["ma_long"])
    longest = max(p["looks"])
    cs, ds, os_, shp = [], [], [], []
    for o in range(N_OFF):
        nav = RB.run_one(C, turn_sm, feats, start, o,
                         p["top_n"], p["tier"], p["buffer"], longest)
        if len(nav) < 50: continue
        nav = nav/nav.iloc[0]
        s = stats(nav)
        cs.append(s["cagr"]); ds.append(s["dd"]); shp.append(s["sh"])
        os_.append(seg(nav, 2019, 2026)["cagr"])
    if not cs: return None
    r = dict(label=label, cagr=np.mean(cs), sd=np.std(cs), se=np.std(cs)/np.sqrt(len(cs)),
             dd=np.mean(ds), worst_dd=np.min(ds), oos=np.mean(os_), sharpe=np.mean(shp))
    print(f"{r['label']:38} {r['cagr']:6.2f}% +/-{r['se']:4.2f}  {r['dd']:7.1f}% "
          f"{r['worst_dd']:8.1f}% {r['sharpe']:6.2f} {r['oos']:7.2f}%")
    return r


def main():
    C, T = build_panel()
    turn_sm = T.rolling(21).mean()
    start = int(np.searchsorted(C.index.year.values, RB.FROM_YEAR))
    print(f"panel {C.index[start].date()} -> {C.index[-1].date()} | {C.shape[1]} symbols")
    print(f"each config averaged over ALL {N_OFF} offsets (se ~ sd/sqrt(21))\n")
    print(f"{'config':38} {'CAGR (mean+/-se)':>16}  {'avgDD':>7} {'worstDD':>8} {'Sharpe':>6} {'OOS':>7}")
    print("-"*92)

    SLOW = (126, 252, 378)
    rows = [
        eval_full(C, turn_sm, start, "A deployed (15, buf1.67, 3/6/12)"),
        eval_full(C, turn_sm, start, "B top_n=25", top_n=25),
        eval_full(C, turn_sm, start, "C buffer=2.0", buffer=2.0),
        eval_full(C, turn_sm, start, "D slow looks 6/12/18mo", looks=SLOW),
        eval_full(C, turn_sm, start, "E B+C (25, buf2.0)", top_n=25, buffer=2.0),
        eval_full(C, turn_sm, start, "F B+C+D (25, buf2.0, slow)",
                  top_n=25, buffer=2.0, looks=SLOW),
        eval_full(C, turn_sm, start, "G F + trend OFF",
                  top_n=25, buffer=2.0, looks=SLOW, ma_long=0),
        eval_full(C, turn_sm, start, "H milder (20, buf2.0)", top_n=20, buffer=2.0),
    ]
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    df.to_csv(BASE/"data_store/combo_results.csv", index=False)

    a = df[df.label.str.startswith("A")].iloc[0]
    print("\n" + "="*92); print("READ-OUT vs deployed (A)"); print("="*92)
    for _, r in df.iterrows():
        if r.label.startswith("A"): continue
        dc = r.cagr-a.cagr; dd = r.dd-a.dd; do = r.oos-a.oos
        # two independent means: significant if gap > ~2 combined standard errors
        sig = abs(dc) > 2*np.sqrt(r.se**2 + a.se**2)
        print(f"{r.label:38} CAGR {dc:+5.2f}pp {'(significant)' if sig else '(within noise)':16} "
              f"DD {dd:+5.1f}pp  OOS {do:+5.2f}pp  Sharpe {r.sharpe-a.sharpe:+.2f}")
    print("\nA config is only worth adopting if it improves DD/Sharpe while its CAGR gap")
    print("is 'within noise' -- that is giving up return you cannot prove you had.")
    print("saved -> data_store/combo_results.csv")


if __name__ == "__main__":
    main()
