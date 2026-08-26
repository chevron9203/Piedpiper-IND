"""
Bank Nifty ORB — Daily Bar Backtest (LONG + SHORT)
====================================================
Strategy:
  - Bull day (Bank Nifty prev close > EMA20): LONG at open, exit at close
  - Bear day (Bank Nifty prev close < EMA20): SHORT at open, exit at close
  - Captures intraday directional move in Bank Nifty index

Data: Bank Nifty daily from yfinance (^NSEBANK), 2015-2026

Instrument: Bank Nifty futures
  - Lot size: 15 units (changed from 25 in 2023, but use 15 throughout for simplicity)
  - Margin: ~₹50,000-70,000 per lot

Note: This is a daily-bar approximation. A real ORB would use 15-min data
to identify the 9:15-9:30 opening range. Will upgrade when 15-min data available.
"""
import sys
sys.path.insert(0, '/Users/abc/Documents/ozi-code/piedpiper')
import pandas as pd
import numpy as np
import yfinance as yf

# ── Constants ─────────────────────────────────────────────────────────────────
CAPITAL        = 500_000
BN_LOT_SIZE    = 15        # Bank Nifty lot size (post-2023 change; use 15 throughout)
MARGIN_PER_LOT = 55_000    # ~₹55K per lot (SPAN+exposure approximate)
BROKERAGE      = 40
STT_RATE       = 0.0001
EXCHANGE_RATE  = 0.0000335
GST_RATE       = 0.18
EMA_PERIOD     = 20

# ── Download Bank Nifty daily data ────────────────────────────────────────────
print("Fetching Bank Nifty daily data from yfinance...")
raw = yf.download("^NSEBANK", start="2015-01-01", end="2026-08-26",
                  auto_adjust=True, progress=False)
raw.columns = [c[0].lower() for c in raw.columns]
df = raw[['open','high','low','close']].copy()
df = df.dropna()
print(f"Bank Nifty: {len(df)} days | {df.index[0].date()} → {df.index[-1].date()}")

# ── Signal ────────────────────────────────────────────────────────────────────
df['ema20']      = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
df['prev_close'] = df['close'].shift(1)
df['prev_ema20'] = df['ema20'].shift(1)
df['bull_day']   = df['prev_close'] >= df['prev_ema20']
df['bear_day']   = df['prev_close'] <  df['prev_ema20']
df = df.dropna()

# ── Cost function ─────────────────────────────────────────────────────────────
def calc_cost(price, lots=1):
    notional = price * BN_LOT_SIZE * lots
    stt      = notional * STT_RATE
    exch     = notional * EXCHANGE_RATE * 2
    brok     = BROKERAGE
    gst      = (brok + exch) * GST_RATE
    return brok + stt + exch + gst

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df, direction_col, direction_sign, label):
    trades = []
    for dt, row in df.iterrows():
        if not row[direction_col]:
            continue
        entry  = float(row['open'])
        exit_p = float(row['close'])
        cost   = calc_cost(entry)
        gross  = (exit_p - entry) * direction_sign * BN_LOT_SIZE
        net    = gross - cost
        trades.append({'dt': dt, 'entry': entry, 'exit': exit_p,
                       'gross_pnl': gross, 'cost': cost, 'net_pnl': net})
    return pd.DataFrame(trades)

long_df  = run_backtest(df, 'bull_day',  1, 'LONG')
short_df = run_backtest(df, 'bear_day', -1, 'SHORT')

def print_stats(t, label, capital, years=11.58):
    if t.empty:
        print(f"  {label}: No trades"); return
    total_net   = t['net_pnl'].sum()
    winners     = t[t['net_pnl'] > 0]
    losers      = t[t['net_pnl'] < 0]
    final_eq    = capital + total_net
    cagr        = (final_eq / capital) ** (1/years) - 1
    
    print(f"\n{'═'*60}")
    print(f"  Bank Nifty {label}")
    print(f"{'═'*60}")
    print(f"  Trades    : {len(t)}")
    print(f"  Win rate  : {len(winners)/len(t)*100:.1f}%")
    print(f"  Net P&L   : ₹{total_net:+,.0f}")
    print(f"  CAGR      : {cagr*100:.1f}%  (on ₹{capital:,.0f})")
    print(f"  Avg win   : ₹{winners['net_pnl'].mean():+,.0f}" if len(winners) else "  No winners")
    print(f"  Avg loss  : ₹{losers['net_pnl'].mean():+,.0f}"  if len(losers)  else "  No losers")
    
    t = t.copy()
    t['year'] = pd.to_datetime(t['dt']).dt.year
    yearly = t.groupby('year')['net_pnl'].sum()
    print(f"  Year-by-year:")
    for yr, pnl in yearly.items():
        bar  = '█' * min(int(abs(pnl)/5000), 25)
        sign = '+' if pnl >= 0 else ''
        print(f"    {yr}: ₹{pnl:>+10,.0f}  {bar}")
    
    cum = t.sort_values('dt')['net_pnl'].cumsum()
    roll_max = cum.cummax()
    dd = cum - roll_max
    print(f"  Max DD: ₹{dd.min():,.0f}  ({dd.min()/capital*100:.1f}%)")

print_stats(long_df,  "LONG  (1 lot, bull days, intraday)", CAPITAL)
print_stats(short_df, "SHORT (1 lot, bear days, intraday)", CAPITAL)

# Combined: LONG on bull days + SHORT on bear days
combined_pnl = long_df['net_pnl'].sum() + short_df['net_pnl'].sum()
combined_trades = len(long_df) + len(short_df)
final_eq = CAPITAL + combined_pnl
years = 11.58
cagr = (final_eq / CAPITAL) ** (1/years) - 1
print(f"\n{'═'*60}")
print(f"  COMBINED: LONG bull days + SHORT bear days")
print(f"{'═'*60}")
print(f"  Total trades: {combined_trades} | Net P&L: ₹{combined_pnl:+,.0f}")
print(f"  CAGR: {cagr*100:.1f}%  Final eq: ₹{final_eq:,.0f}")

bull_days = df['bull_day'].sum()
bear_days = df['bear_day'].sum()
avg_bull_move = (df[df['bull_day']]['close'] - df[df['bull_day']]['open']).div(df[df['bull_day']]['open']).mean() * 100
avg_bear_move = (df[df['bear_day']]['close'] - df[df['bear_day']]['open']).div(df[df['bear_day']]['open']).mean() * 100
print(f"\n  Bull days: {bull_days}  |  Bear days: {bear_days}")
print(f"  Avg intraday move on bull days: {avg_bull_move:+.3f}%")
print(f"  Avg intraday move on bear days: {avg_bear_move:+.3f}%")
