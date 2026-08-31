"""
INR Currency Strategies Backtest
==================================
Strategy A: USDINR Trend Following
  - LONG USDINR when close > EMA20 (INR weakening, USD strengthening)
  - SHORT USDINR when close < EMA20 (INR strengthening)
  - 1 lot = 1000 USD (~₹95,720 notional)
  - Margin per lot: ~₹2,500

Strategy B: EURINR/USDINR Pairs Trade (Triangular)
  - Implied EURUSD = EURINR / USDINR
  - If implied EURUSD deviates from actual EURUSD (via EURUSD=X): arbitrage
  - Enter when spread > 1 std, exit at mean
  - Market neutral: long one, short the other

Data: yfinance daily (2015-2026)
"""
import sys
sys.path.insert(0, '/Users/abc/Documents/ozi-code/piedpiper')
import pandas as pd
import numpy as np
import yfinance as yf

CAPITAL       = 500_000
USD_LOT       = 1000       # 1 USDINR lot = 1000 USD
MARGIN_PER_LOT = 3000      # ~₹3K margin (SPAN+exposure for currency futures)
BROKERAGE     = 20         # flat per trade, lower for currency
STT_RATE      = 0.000002   # 0.0002% STT for currency
EXCHANGE_RATE = 0.0000335
GST_RATE      = 0.18
EMA_PERIOD    = 20
MAX_LOTS      = 5          # 5 lots = 5000 USD notional ~₹4.8L

print("Fetching currency data from yfinance...")
usdinr_raw = yf.download("USDINR=X",  start="2015-01-01", end="2026-08-26", auto_adjust=True, progress=False)
eurinr_raw = yf.download("EURINR=X",  start="2015-01-01", end="2026-08-26", auto_adjust=True, progress=False)
eurusd_raw = yf.download("EURUSD=X",  start="2015-01-01", end="2026-08-26", auto_adjust=True, progress=False)

def flatten(raw):
    df = raw.copy()
    df.columns = [c[0].lower() for c in df.columns]
    return df[['open','high','low','close']].dropna()

usdinr = flatten(usdinr_raw)
eurinr = flatten(eurinr_raw)
eurusd = flatten(eurusd_raw)

print(f"USDINR: {len(usdinr)} days | EURINR: {len(eurinr)} days | EURUSD: {len(eurusd)} days")
print(f"Latest USDINR: {usdinr['close'].iloc[-1]:.4f} | EURINR: {eurinr['close'].iloc[-1]:.4f}")

# ── Strategy A: USDINR Trend Following ────────────────────────────────────────
print("\n" + "═"*60)
print("  STRATEGY A: USDINR Trend Following (daily)")
print("═"*60)

usdinr['ema20']      = usdinr['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
usdinr['prev_close'] = usdinr['close'].shift(1)
usdinr['prev_ema20'] = usdinr['ema20'].shift(1)
usdinr['signal']     = np.where(usdinr['prev_close'] > usdinr['prev_ema20'],  1,   # LONG USDINR
                        np.where(usdinr['prev_close'] < usdinr['prev_ema20'], -1, 0))
usdinr = usdinr.dropna()

def calc_cost_fx(price, lots):
    notional = price * USD_LOT * lots
    stt      = notional * STT_RATE
    exch     = notional * EXCHANGE_RATE * 2
    brok     = BROKERAGE
    gst      = (brok + exch) * GST_RATE
    return brok + stt + exch + gst

trades_a = []
position = 0
entry_price = 0

for dt, row in usdinr.iterrows():
    sig = int(row['signal'])
    if sig == 0:
        continue
    price = float(row['open'])   # enter at next day open

    if position != 0 and position != sig:
        # Reverse: exit then enter opposite
        cost = calc_cost_fx(price, MAX_LOTS)
        gross = (price - entry_price) * position * USD_LOT * MAX_LOTS
        net   = gross - cost
        trades_a.append({'dt': dt, 'direction': position, 'entry': entry_price,
                         'exit': price, 'gross_pnl': gross, 'net_pnl': net})
        position    = sig
        entry_price = price
    elif position == 0:
        position    = sig
        entry_price = price

trades_a_df = pd.DataFrame(trades_a) if trades_a else pd.DataFrame()

if not trades_a_df.empty:
    total_net = trades_a_df['net_pnl'].sum()
    winners   = trades_a_df[trades_a_df['net_pnl'] > 0]
    losers    = trades_a_df[trades_a_df['net_pnl'] < 0]
    final_eq  = CAPITAL + total_net
    years     = 11.58
    cagr      = (final_eq / CAPITAL) ** (1/years) - 1
    long_pnl  = trades_a_df[trades_a_df['direction'] == 1]['net_pnl'].sum()
    short_pnl = trades_a_df[trades_a_df['direction'] == -1]['net_pnl'].sum()
    
    print(f"  Trades: {len(trades_a_df)} | Win rate: {len(winners)/len(trades_a_df)*100:.1f}%")
    print(f"  Net P&L : ₹{total_net:+,.0f}  |  CAGR: {cagr*100:.1f}%")
    print(f"  LONG P&L: ₹{long_pnl:+,.0f}  |  SHORT P&L: ₹{short_pnl:+,.0f}")
    
    trades_a_df['year'] = pd.to_datetime(trades_a_df['dt']).dt.year
    yearly = trades_a_df.groupby('year')['net_pnl'].sum()
    print(f"  Year-by-year:")
    for yr, pnl in yearly.items():
        bar = '█' * min(int(abs(pnl)/3000), 25)
        print(f"    {yr}: ₹{pnl:>+10,.0f}  {bar}")

# ── Strategy B: EURINR / USDINR Pairs Trade ───────────────────────────────────
print("\n" + "═"*60)
print("  STRATEGY B: EURINR/USDINR Pairs Trade (Triangular)")
print("═"*60)

# Align dates
common = usdinr.index.intersection(eurinr.index).intersection(eurusd.index)
u = usdinr.loc[common, 'close'].rename('usdinr')
e = eurinr.loc[common, 'close'].rename('eurinr')
eu = eurusd.loc[common, 'close'].rename('eurusd_actual')

pairs = pd.concat([u, e, eu], axis=1).dropna()
pairs['eurusd_implied'] = pairs['eurinr'] / pairs['usdinr']
pairs['spread']         = pairs['eurusd_implied'] - pairs['eurusd_actual']

# Spread stats
spread_mean = pairs['spread'].rolling(60).mean()
spread_std  = pairs['spread'].rolling(60).std()
pairs['zscore'] = (pairs['spread'] - spread_mean) / spread_std

print(f"  Data: {len(pairs)} days  |  Spread range: {pairs['spread'].min():.5f} to {pairs['spread'].max():.5f}")
print(f"  Mean spread: {pairs['spread'].mean():.5f}  |  Std: {pairs['spread'].std():.5f}")

# Pairs trade: enter when z-score > 1.5 (spread too wide) — expect reversion
# z > 1.5: EURINR overvalued vs USDINR → SHORT EURINR, LONG USDINR
# z < -1.5: EURINR undervalued → LONG EURINR, SHORT USDINR
ENTRY_Z = 1.5
EXIT_Z  = 0.5

pairs_trades = []
in_trade = 0  # 1 or -1

for dt, row in pairs.iterrows():
    z = row['zscore']
    if pd.isna(z):
        continue
    
    if in_trade == 0:
        if z > ENTRY_Z:
            in_trade = -1  # spread reverts down → short spread
        elif z < -ENTRY_Z:
            in_trade = 1   # spread reverts up → long spread
    else:
        if abs(z) < EXIT_Z:
            # Exit: approximate P&L = z-movement × std × USD_LOT × lots
            pnl_approx = in_trade * (abs(row['zscore']) - ENTRY_Z) * pairs['spread'].std() * USD_LOT * MAX_LOTS * -1
            cost = BROKERAGE * 4  # 2 pairs × 2 sides
            pairs_trades.append({'dt': dt, 'direction': in_trade, 
                                  'net_pnl': pnl_approx - cost, 'z_exit': z})
            in_trade = 0

pairs_df = pd.DataFrame(pairs_trades) if pairs_trades else pd.DataFrame()
if not pairs_df.empty:
    total_pnl = pairs_df['net_pnl'].sum()
    winners   = pairs_df[pairs_df['net_pnl'] > 0]
    print(f"  Trades: {len(pairs_df)}  |  Win rate: {len(winners)/len(pairs_df)*100:.1f}%")
    print(f"  Net P&L: ₹{total_pnl:+,.0f}")
    print(f"  Note: P&L approximation only — spread mean-reversion model")
    pairs_df['year'] = pd.to_datetime(pairs_df['dt']).dt.year
    for yr, pnl in pairs_df.groupby('year')['net_pnl'].sum().items():
        print(f"    {yr}: ₹{pnl:>+8,.0f}")
else:
    print("  No completed trades")
