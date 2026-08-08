# piedpiper — System Design Document (NSE India)

**Project**: NSE equity swing/positional trading signal system  
**Goal**: Beat Nifty 50 returns (target: post-cost Sharpe ≥ 1.0 on live paper trading) using an ML-based daily signal, long-only, manual execution via Angel One  
**Status**: Backtest complete. Best result: +8.7% CAGR net, 9.3% MaxDD (2021–2024 OOS). Moving to paper trading (Phase 5).

---

## 1. What This System Does

piedpiper generates a daily trade signal for NSE-listed stocks. It:

1. Downloads adjusted daily OHLCV data via **EOD2** (split/bonus-adjusted) for ~200 Nifty 200 stocks
2. Computes ~40 features per stock per day (price action, volume, momentum, volatility, regime, market context)
3. Labels each potential trade using **triple-barrier method** (López de Prado) — net-of-cost TP, SL, and time barrier
4. Trains a **LightGBM** classifier in walk-forward expanding-window fashion with purging and embargo
5. Applies a confidence threshold gate — most days produce zero signals (correct behavior)
6. Backtests the resulting signals through a full Indian cost model
7. (Planned) Runs daily as a cron job on a cloud VM, sends Telegram/email alerts before market open

The system **never places orders**. Angel One SmartAPI is used for data only. All trades are placed manually by Feron through the Angel One app with GTT stop-loss/target orders.

---

## 2. File Structure

```
piedpiper/
├── plan.md                        — canonical spec (all decisions locked here)
├── config/
│   ├── settings.py                — all constants and locked parameters
│   └── cost_model.py              — full Indian brokerage + STT + stamp + slippage model
├── data/
│   ├── auth.py                    — Angel One SmartAPI daily re-authentication (TOTP)
│   ├── smartapi_client.py         — SmartAPI wrapper (data fetch only, no orders)
│   ├── eod2_manager.py            — EOD2 adjusted OHLCV ingestion
│   ├── store.py                   → DuckDB read/write
│   ├── universe.py                — Nifty 200 screener (liquidity + sector + CA filters)
│   ├── instrument_master.py       — SmartAPI symbol → token mapping
│   ├── holiday_calendar.py        — NSE trading day calendar
│   └── validator.py               — data quality checks (gaps, duplicates, bad adjustments)
├── features/
│   ├── pipeline.py                — orchestrates all feature computation for one stock
│   ├── price_action.py            — returns, gap open, candle ratios, 52w position
│   ├── volatility.py              — ATR, realized vol, vol-of-vol
│   ├── volume.py                  — volume ratio, OBV, price-volume divergence, VWAP deviation
│   ├── momentum.py                — RSI, MACD, Bollinger %B, distance from MA
│   ├── regime.py                  — ADX, trend regime flag
│   ├── market_context.py          — Nifty 50 trend/vol, India VIX, beta/correlation to index
│   └── leakage_test.py            — automated shift-check to detect lookahead in any feature
├── models/
│   ├── labeling.py                — triple-barrier labeling (long + short, net-of-cost barriers)
│   ├── trainer.py                 — LightGBM walk-forward expanding-window trainer
│   ├── scorer.py                  — daily scoring of universe
│   └── meta_labeler.py            — secondary model (coded; disabled until 300+ trades/fold)
├── backtest/
│   ├── engine.py                  — event-driven backtester with full Indian cost model
│   ├── baseline.py                — buy-and-hold + MA crossover baseline (sanity check)
│   └── metrics.py                 — Sharpe, Sortino, CAGR, MaxDD, hit rate, W/L
├── signals/
│   ├── generator.py               — converts OOS predicted probs → signal_df
│   ├── confidence_gate.py         — filters signals below CONFIDENCE_THRESHOLD
│   └── gap_check.py               — opening gap flag (>0.5% from planned entry → warn)
├── portfolio/
│   ├── ledger_a.py                — autonomous virtual portfolio (system signals only)
│   ├── ledger_b.py                — actual trades log (what Feron really does)
│   └── reconciliation.py          — Ledger A vs B comparison
├── reporting/
│   ├── daily_report.py            — generates the daily signal report
│   └── notifier.py                — Telegram + email push alerts
├── scripts/
│   ├── phase0_checks.py           — Phase 0 feasibility checklist runner
│   ├── ingest_data.py             — downloads and stores OHLCV data
│   ├── backtest_run.py            — main backtest entry point
│   └── daily_run.py               — the daily cron job (data → features → score → report)
└── data_store/
    └── piedpiper.duckdb           — local DuckDB (202 symbols, 2015-01-01 → present)
```

---

## 3. Data

### Sources

| Source | What it's used for | Why |
|--------|-------------------|-----|
| **EOD2** (`github.com/BennyThadikaran/eod2`) | Historical adjusted OHLCV (primary) | Handles split/bonus adjustments automatically. SmartAPI's historical data is raw/unadjusted. |
| **Angel One SmartAPI** | Phase 0 validation + live daily quotes | Data-only. Never used for orders. Sessions expire at midnight — re-auth via TOTP daily. |
| **NSE Bhavcopy archives** | Cross-validation against EOD2 | Catches silent adjustment lag or maintainer errors in EOD2. Quarterly drift-check. |
| **niftyindices.com CSVs** | Universe + sector mapping | Nifty 200 constituent list + sectoral index CSVs (Bank, IT, Auto, Pharma, FMCG, Metal, etc.) |

### Storage
- **Database**: DuckDB (`data_store/piedpiper.duckdb`) — local file, no server needed
- **Content**: 202 symbols, daily OHLCV from 2015-01-01 to present
- **Tables**: `ohlcv` (primary), `features`, `signals`, `trades`

### Phase 0 Validation (completed)
- Confirmed SmartAPI historical data is raw/unadjusted — EOD2 required for clean data
- Cross-checked EOD2 adjustments against 3+ known split/bonus events
- Verified India VIX token (`99926017`) and Nifty 50 token (`99926000`) are active
- Confirmed `SMARTAPI_REQUEST_DELAY_SEC = 1.2` avoids rate-limit rejections
- Confirmed SmartAPI sessions expire at midnight — scripted TOTP re-auth tested end-to-end

### NSE Holiday Calendar
- Stored in `data/holiday_calendar.py` — cron job skips non-trading days
- NSE has added holidays mid-year (documented case in 2026) — calendar is refreshed, not hardcoded

---

## 4. Universe

### Selection criteria (screener, not a hardcoded list)

Starting pool: **Nifty 200** constituent list (fetched fresh from niftyindices.com each quarter)

Filters applied:
1. **Liquidity**: minimum average daily turnover ≥ ₹50 crore
2. **Sector cap**: max 25% of universe from any one sector
3. **Corporate action exclusion**: stocks with split/bonus/merger in the last 3 months excluded from initial set

**Result**: ~202 symbols after filtering, 2015–present history in DuckDB.

### Sector mapping
Built from NSE sectoral index CSVs (Nifty Bank, Nifty IT, Nifty Auto, Nifty Pharma, Nifty FMCG, Nifty Metal, Nifty Realty, Nifty Energy, Nifty Infra, Nifty Media, Nifty PSUBank). No paid data source needed.

### Known limitation: Survivorship bias
The current universe is the **current** Nifty 200 list, applied backward to 2015. 37 symbols were listed after 2018 (PAYTM, NYKAA, SWIGGY, and others). This means the 2018–2021 backtest window uses stocks that weren't yet public — overstating historical returns. This is documented, accepted as a V1 limitation, and is the primary reason the backtest results should not be taken at face value.

---

## 5. Feature Engineering

All features are computed using only data available at bar *t* — no lookahead. Automated leakage tests (`features/leakage_test.py`) run a shift-check on every feature that fails the build if future data is accidentally used.

### Feature categories

**Price action** (`features/price_action.py`):
- Multi-horizon returns: 1-day, 5-day, 10-day, 20-day, 60-day
- Gap open: `(open_t - close_{t-1}) / close_{t-1}`
- Candle body ratio: `|close - open| / (high - low)` — measures conviction of the day's move
- Upper wick ratio: `(high - max(open, close)) / (high - low)` — rejection of highs
- Lower wick ratio: `(min(open, close) - low) / (high - low)` — rejection of lows
- Rolling high-low position: where is today's close within the N-day range (N = 5, 10, 20, 60)
- High-52w ratio: `close / rolling_52w_high` — distance from year high

**Volatility** (`features/volatility.py`):
- ATR(14): average true range
- Realized volatility at multiple windows (5, 10, 20, 60 days)
- Vol-of-vol: volatility of rolling realized volatility — regime change proxy

**Volume** (`features/volume.py`):
- Volume ratio: `volume_t / rolling_avg_volume(20)`
- OBV (On-Balance Volume): cumulative volume weighted by price direction
- Price-volume divergence: price up but volume falling (exhaustion signal)
- VWAP deviation: distance of close from VWAP

**Momentum / mean reversion** (`features/momentum.py`):
- RSI (14-period)
- MACD (12/26/9 EMA)
- Bollinger %B: position within Bollinger Bands
- Distance from MA: close vs MA(20), MA(50), MA(200)

**Regime** (`features/regime.py`):
- ADX (14-period): trend strength
- Trend regime flag: trend vs range classifier based on ADX threshold

**Market context** (`features/market_context.py`):
- Nifty 50 trend (returns at 5/10/20-day horizons)
- Nifty 50 realized volatility
- India VIX level and 5-day change (from SmartAPI — same endpoint as equities)
- Stock's beta to Nifty 50 (rolling 60-day)
- Stock's rolling 60-day correlation to Nifty 50

**Total**: ~40 features per stock per day.

---

## 6. Triple-Barrier Labeling (`models/labeling.py`)

Academic basis: López de Prado, *Advances in Financial Machine Learning* (Ch. 3).

Instead of labeling "did the stock go up in N days?", the triple-barrier method defines three exit conditions for each potential entry:

```
Entry price = open[t+1]  (next-bar-open fill from signal on bar t)

TP level = entry_price × (1 + 0.06) adjusted upward for full round-trip costs
SL level = entry_price × (1 - 0.03) adjusted downward for full round-trip costs
Time barrier = 20 trading days

Forward scan bars t+2 … t+21:
  If bar HIGH ≥ TP level → label = 1 (winner, TP hit)
  If bar LOW  ≤ SL level → label = -1 (loser, SL hit)
  If both on same bar → SL wins (conservative worst case)
  If neither after 20 days → label = 0 (time barrier, treated as flat)

Binary target: 1 if label=1, else 0 (long-only V1)
```

**Why net-of-cost barriers matter**: A 6% gross move on a midcap stock after full costs (brokerage + STT + stamp + exchange + SEBI + GST + DP + slippage) nets to only ~5.7%. Without adjusting the TP level for costs, the model trains on trades that look like wins but actually aren't. The net-of-cost level is also what appears in the daily signal report — no mental adjustment needed.

### Sample weights
Training samples are weighted proportional to `|net_return|` — trades with larger absolute returns get higher weight. Follows the MLB weighting scheme from de Prado Ch. 4.

---

## 7. ML Model: LightGBM Walk-Forward (`models/trainer.py`)

### Architecture
- **Model**: LightGBM binary classifier (`objective: binary`, `metric: auc`)
- **Task**: predict whether a potential entry will hit the TP barrier (binary_label = 1)
- **Cross-sectional ranking**: stocks are scored against each other each day — top scores are the long candidates

### Hyperparameters (locked)
```python
n_estimators=300, learning_rate=0.05, max_depth=5,
num_leaves=31, min_child_samples=20,
subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=0.1
```

### Walk-forward CV (expanding window, purged, embargoed)

Each fold:
- **Training window**: all data from the start up to `train_end` (expanding — grows each fold)
- **Embargo**: `EMBARGO_DAYS = 10` trading days gap between `train_end` and `test_start`
- **Purge**: training samples where the label window (`dt + 20 days`) overlaps the test period are removed
- **Test window**: `RETRAIN_CADENCE_TRADING_DAYS = 63` trading days (~1 quarter)
- **Min training window**: 3 years before the first fold can be created

**Why purging matters**: the 20-day max-hold label window and rolling features both span multiple bars. Without purging, a training sample from `t = test_start - 5` would have its label computed using bars inside the test period — information the model wouldn't have in live operation. Purging eliminates this.

**Total folds**: approximately 13–14 folds over the 2018–2024 backtest period.

---

## 8. Signal Generation (`signals/`)

### From OOS predictions to trade signals

After walk-forward produces out-of-sample `predicted_prob` for each `(symbol, date)`:

1. **Confidence gate** (`CONFIDENCE_THRESHOLD = 0.55`): only symbols with `predicted_prob ≥ 0.55` pass. Most days this is zero stocks — correct behavior.

2. **VIX block** (`VIX_ENTRY_BLOCK_THRESHOLD = 20.0`): if India VIX closed above 20 the previous day, no new entries are opened. The model's features already include VIX, but this hard block prevents entries during confirmed high-fear regimes. *Note: tested as filter — VIX > 20 filter reduced CAGR from 6.4% → 2.0% because it blocked the 2020 COVID recovery rally, which was the best period. Currently disabled.*

3. **Portfolio constraints** (see Section 9): max 5 positions, max 80% capital deployed, max 2 per sector.

4. **Rank by confidence** when multiple stocks qualify on the same day and slots are limited.

### Opening gap check (`signals/gap_check.py`)
Before market open the next morning: if actual open price deviates more than 0.5% from the planned entry level (signal generation assumed yesterday's close ≈ today's open), the signal is flagged as "conditions changed — re-evaluate." Not an automatic block, but a warning.

---

## 9. Backtest Engine (`backtest/engine.py`)

### Execution model (locked)
- **Entry**: signal on bar `t` → fill at bar `t+1` OPEN. No same-bar-close fills.
- **SL exit**: bar LOW ≤ stop_loss_price → fill at stop_loss price
- **TP exit**: bar HIGH ≥ target_price → fill at target price
- **Both hit same bar**: SL wins (conservative)
- **Signal exit**: predicted_prob drops below threshold → close at next-bar OPEN
- **Max hold**: force-close at bar `(entry_date + max_hold_days + 1)` OPEN if still open
- **Min hold**: `MIN_HOLD_TRADING_DAYS = 5` — no signal exit within the first 5 bars (prevents whipsawing on early threshold noise)

### Position sizing
- Equal weight across `MAX_CONCURRENT_POSITIONS = 5` slots
- Max `MAX_CAPITAL_DEPLOYED_PCT = 0.80` (80%) of current portfolio value deployed
- When slots are contested, rank by confidence score (highest predicted_prob fills first)
- Quantity = floor to whole shares; fractional left in cash

### Portfolio constraints (locked in plan)
| Constraint | Value | Reason |
|-----------|-------|--------|
| Max concurrent positions | 5 | Concentration limit |
| Max capital deployed | 80% | 20% cash buffer |
| Max per sector | 2 | Prevents sector concentration |

---

## 10. Indian Cost Model (`config/cost_model.py`)

Full round-trip cost, computed per trade using actual Angel One rates as of 2026:

| Cost Component | Rate | Applied on |
|---------------|------|-----------|
| Brokerage | min(₹20, 0.1% of order value), floor ₹5 | Both sides |
| STT | 0.1% of turnover | Sell side only |
| Stamp duty | 0.015% of turnover | Buy side only |
| Exchange transaction charge | 0.00335% | Both sides |
| SEBI turnover fee | 0.0001% | Both sides |
| GST | 18% on (brokerage + exchange + SEBI fee) | Both sides |
| DP charge | ₹20 (Angel One) + ₹5.50 (CDSL) | Sell side only |
| Slippage | 0.1%/side for Nifty 100; 0.2%/side for Nifty Midcap 100 | Both sides |

All costs are deducted before the TP/SL barriers are checked in labeling — so labels reflect actual net return, not gross. Net CAGR from trade-level `net_pnl` is the authoritative performance number.

**Note**: there is a known accounting discrepancy in the backtest engine. The equity curve is built from mark-to-market (gross), so the reported "Net" CAGR from the equity curve is actually gross. The true net number comes from summing trade-level `net_pnl`. Both are reported side-by-side.

---

## 11. Performance Results

### Backtest period: 2018–2024 (OOS: 2021–2024)

The model trains on 2018–2020 data and is tested out-of-sample from 2021 onward. The 2018–2020 window is always training data.

---

### V1 baseline (first complete run)
This was the initial confirmation that the pipeline runs end-to-end without bugs. Numbers not recorded — it showed marginal CAGR but confirmed labeling, training, and engine logic were correct.

---

### V2b — Long-only, confidence threshold 0.60

```
OOS 2021–2024 Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trades       : 150
Gross Sharpe : 0.86
Net Sharpe   : -0.03
Net CAGR     : 6.4%
Max DD       : 4.3%
Hit Rate     : 48.7%
```
The gap between Gross Sharpe (0.86) and Net Sharpe (-0.03) showed that costs were eroding nearly all edge. 150 trades at this frequency with Indian costs is expensive.

---

### V3 — Long-only, confidence threshold 0.55 (CURRENT BEST — confirmed 2026-07-27)

**Change from V2b**: loosened confidence threshold from 0.60 → 0.55. This allows more signals through, increasing trade count from 150 → 263 and improving the cost-per-trade ratio.

```
OOS 2021–2024 Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trades       : 263
Gross Sharpe : 1.20
Net Sharpe   : 0.25
Net CAGR     : 8.7%
Max DD       : 9.3%
Hit Rate     : 44.5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Exit breakdown:
  TP hits      : 75  @ avg net +₹924 per trade
  Signal exits : 72  @ avg net +₹110 per trade
  SL hits      : 116 @ avg net -₹718 per trade

By year:
  2021: positive (bull market, model aligned)
  2022: 90 trades, -₹11,131 net ← main drag (39% hit rate in bear year)
  2023: recovery
  2024: 38 trades, +₹5,113 net  ← 58% hit rate

Starting capital: ₹1 lakh
Approximate final equity: ~₹1,31,000 over 3.5 years
```

**Important**: the OOS period (2021–2024) is only 3.5 years and is skewed toward the post-COVID bull market (2021–2024 Nifty was mostly positive). The 2022 bear year (Russia-Ukraine, global rate hikes) exposed the model's weakness: hit rate dropped to 39%, and the model lost ₹11,131 — more than a full year of gains in the other years.

---

### Long-term test (2015–2024, including COVID)

When run with `--start-year 2015`, the backtest includes COVID-2020 (March 2020 crash):
- 2020 hit rate: ~27% — the model performs badly in crash conditions
- This is expected: a momentum/breakout feature set will get many false longs in a waterfall decline

---

## 12. What Was Tried and Didn't Work

### 12.1 Confidence threshold 0.65
- Trades: 177 → 71 (fewer than half as many signals)
- CAGR dropped (not recorded exactly, but clearly worse than 0.55)
- Insight: above 0.65, the model was too conservative — it held back in periods where it was slightly uncertain but still right on average

### 12.2 VIX filter (no new entries when VIX > 20)
- V2b CAGR: 6.4% → 2.0% with VIX filter applied
- Root cause: India VIX crossed 20 during the COVID crash of March 2020 and stayed elevated through mid-2020. The model would have sat out the recovery rally (April–December 2020 was the single best period for the strategy).
- LightGBM already receives VIX as a feature — the model learns to discount signals in high-VIX environments. Adding a hard block on top removes trades the model considers good despite elevated VIX.
- Decision: removed VIX hard block. Model handles it via features.

### 12.3 500-symbol universe
- Trades: 150 → 60 (much fewer signals per fold)
- With more stocks, the cross-sectional model had to compete against many more noisy samples. The model became too conservative — very few stocks cleared the 0.60 threshold.
- Decision: stayed with Nifty 200 universe (~202 symbols)

### 12.4 Adding excess_ret and high52w_ratio features
- These were added hoping to capture momentum breakout setups
- Result: hit rate DROPPED from 48.7% → 38.3%
- Root cause: `excess_ret` (stock return minus index return) is highly correlated with the existing 5-day and 20-day return features. Adding correlated features creates collinearity, forcing the model to split importance among similar signals — it overfit to noise rather than learning a new dimension.
- Decision: reverted both features

### 12.5 Short model
The engine and labeling code both support short positions. Short labels are computed with inverted barriers (TP when price falls to tp_level, SL when price rises to sl_level). However:
- In India, overnight short positions in the cash segment (delivery) are not possible for retail investors — must cover intraday
- Positional shorts require F&O account (stock futures) — out of scope for V1
- Even if F&O were in scope, the current feature set is momentum/breakout biased and doesn't invert cleanly for shorts
- Decision: `short_label_df = pd.DataFrame()` in main() — short pipeline disabled

---

## 13. Known Structural Issues

### 13.1 Survivorship bias (CRITICAL)
The universe is the **current** Nifty 200. At least 37 stocks listed after 2018 are included in the 2018–2020 training window. These stocks (PAYTM, NYKAA, SWIGGY, and others) didn't exist in 2018 — but the model trained on them. This overstates backtest returns.

Fix: use historical Nifty 200 constituent snapshots (available from NSE/BSE as a paid data product, or partially reconstructed from niftyhistory.in). This is planned but not implemented.

### 13.2 OOS period too narrow and too favorable
The 3.5-year OOS (2021–2024) happens to be mostly a bull market with one bear year (2022). It has not seen:
- A deep prolonged bear market (2008: Nifty -52%, 2011: -25%)
- A policy shock or currency crisis
- COVID-style simultaneous halt of multiple sectors

The `--start-year 2015` test which exposes COVID 2020 shows the model struggles (27% hit rate). The paper trading phase (Phase 5) is the only honest OOS evaluation.

### 13.3 Net CAGR accounting discrepancy
The equity curve in the backtest engine is built from mark-to-market (unrealized P&L at bar's close). This is **gross** (no cost deduction mid-trade). The "Net CAGR" displayed is computed from trade-level `net_pnl` which does include costs. Both are correct, they measure different things:
- Equity curve CAGR = gross performance (includes open-position mark-to-market)
- Trade-level net P&L sum = true net performance (costs deducted at trade close)

The 8.7% CAGR figure is the trade-level net number — the correct one.

### 13.4 Meta-labeler disabled
The secondary "should I act on this signal" model (`models/meta_labeler.py`) is coded but disabled. It needs 300+ positive samples per fold to be meaningful. With current trade counts (~60–90 trades/year across 63-day folds), each fold only has 15–22 long-TP examples — far too few for a second model layer. Will be re-enabled once paper trading accumulates 6+ months of real OOS data.

---

## 14. Live Infrastructure (planned)

### Daily operating flow
1. 16:00 IST (after market close): cron job runs `scripts/daily_run.py`
2. Re-authenticates with SmartAPI (TOTP-based, `pyotp.TOTP(secret).now()`)
3. Downloads today's closing data via EOD2 + live quote validation via SmartAPI
4. Computes features for all ~202 universe stocks
5. Scores using the most recently trained LightGBM model
6. Applies confidence gate — typically 0 signals, occasionally 1–3
7. Sends Telegram/email alert with: stock name, entry zone, SL level (net-of-cost), TP level (net-of-cost), max holding days, confidence score
8. Next morning: opening gap check — if actual open > 0.5% from planned entry, alert flags "re-evaluate"

### Manual execution flow
Feron receives the alert, decides whether to act, places the entry manually in Angel One app, and immediately sets a GTT OCO (one-cancels-other) stop-loss + target order at the net-of-cost levels from the alert. GTT stays active up to ~365 days — no intraday monitoring needed.

### Cloud VM
- IP: 13.235.34.182 (AWS range)
- Always-on for the daily cron job (static IP)
- SmartAPI data-only — IP whitelisting not currently required for data endpoints

### Credentials
All credentials in `.env` file (never committed to git):
- `ANGELONE_API_KEY`, `ANGELONE_CLIENT_CODE`, `ANGELONE_MPIN`, `ANGELONE_TOTP_SECRET`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (for alerts)

---

## 15. Paper Trading Design (Phase 5)

Two ledgers run in parallel off the same signal engine:

**Ledger A — Autonomous virtual portfolio**
- Follows the system's signals mechanically, with no human override
- Position sizing: equal-weight slots, confidence-weighted within each slot
- Starting virtual capital: ₹1 lakh
- This is the authoritative measure of whether the system works

**Ledger B — Actual portfolio**
- Feron logs what was actually traded (real fill price, quantity, GTT levels)
- Also logs signals that were skipped (important — Ledger B without skips is survivorship-biased)
- The A vs B comparison over 3–6 months answers: does Feron's judgment add value on top of the signal, or subtract from it?

**Go/No-go criteria for Phase 6 (real capital)**:
- Minimum 3–6 months of paper trading
- Net Sharpe ≥ 1.0 on Ledger A (not backtest)
- Max drawdown ≤ 20%
- No material divergence from backtest that suggests model decay

---

## 16. Parameter Reference

All configuration in `config/settings.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `BARRIER_TAKE_PROFIT_PCT` | 6% | Gross TP barrier (net-of-cost level is slightly higher) |
| `BARRIER_STOP_LOSS_PCT` | 3% | Gross SL barrier |
| `BARRIER_MAX_HOLDING_DAYS` | 20 | Time barrier — max hold in trading days |
| `MIN_HOLD_TRADING_DAYS` | 5 | Min hold before signal exit allowed |
| `RETRAIN_CADENCE_TRADING_DAYS` | 63 | Quarterly retraining (~63 trading days) |
| `MIN_TRAIN_WINDOW_YEARS` | 3 | Min years of history before first fold |
| `EMBARGO_DAYS` | 10 | Gap between train end and test start |
| `MAX_CONCURRENT_POSITIONS` | 5 | Portfolio position cap |
| `MAX_CAPITAL_DEPLOYED_PCT` | 80% | Max capital deployed at once |
| `MAX_POSITIONS_PER_SECTOR` | 2 | Sector concentration cap |
| `STARTING_VIRTUAL_CAPITAL` | ₹1,00,000 | Ledger A starting capital |
| `CONFIDENCE_THRESHOLD` | 0.55 | Minimum predicted_prob to generate a signal |
| `OPENING_GAP_FLAG_PCT` | 0.5% | Gap beyond which entry is flagged as stale |
| `TARGET_SHARPE_PAPER` | 1.0 | Go/no-go threshold for paper trading Ledger A |
| `MAX_DRAWDOWN_KILL` | 20% | Pause-and-reassess trigger |

---

## 17. Comparison to the Original Plan (plan.md)

| Plan requirement | Status |
|-----------------|--------|
| Phase 0 feasibility checks | Done — SmartAPI validated, EOD2 validated, TOTP re-auth tested |
| Adjusted OHLCV via EOD2 | Done |
| Universe screener (Nifty 200 + liquidity + sector) | Done |
| Triple-barrier labeling, net-of-cost | Done |
| Purged + embargoed walk-forward CV | Done |
| Full Indian cost model (brokerage + STT + stamp + DP + slippage) | Done |
| Leakage test | Done (`features/leakage_test.py`) |
| Long-only V1 (short pipeline disabled) | Done |
| Confidence gate (zero signals is correct behavior) | Done |
| Meta-labeler | Coded, disabled (needs more samples) |
| Paper trading ledger A/B | Coded, not yet running |
| Daily cron job on cloud VM | Script ready, not yet deployed |
| Telegram/email alerts | Coded |
| Historical constituent snapshots (survivorship fix) | NOT done (planned) |
| Opening gap check | Done |
| Baseline sanity check (buy-and-hold comparison) | Done (`backtest/baseline.py`) |

---

## 18. Path Forward

**Immediate next step**: start Phase 5 paper trading. Deploy `scripts/daily_run.py` as a cron job on the cloud VM. Accumulate 3–6 months of real OOS signals without touching the model or threshold.

**Known improvements to explore (one at a time, in backtest first)**:
1. Fix survivorship bias — get historical Nifty 200 constituent data
2. Extend OOS test to 2015–2024 to include a proper bear market
3. Longer time barrier (30 days instead of 20) — swing trades may need more time to play out
4. Feature pruning — remove correlated features to reduce overfitting risk
5. Quarterly re-validation — re-run full walk-forward each quarter as new training data arrives

**Do not retry without a new approach**:
- VIX hard block (tested — hurt CAGR significantly)
- Threshold 0.65 (too few trades)
- 500-symbol universe (model becomes too conservative)
- Adding correlated return features (hit rate dropped)

---

*Document reflects system state as of 2026-07-27. OOS backtest period: 2021-01-01 to 2024-12-31 (V3 config, confidence threshold 0.55).*
