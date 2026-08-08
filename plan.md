# Candle-Based Trading Signal System — Build Plan

**Owner:** Feron
**Purpose of this document:** Hand this to Claude Code as the spec for building a research/paper-trading system that generates trade signals from historical + live candlestick (OHLCV) data. This is NOT a plan for a fully automated order-placing bot — see Section 9 for why that's a deliberate choice.

---

## 1. Objective — stated honestly

Build a system that:
1. Ingests historical and live OHLCV data for a defined stock universe.
2. Generates a scored trade signal (long / flat, with confidence) per stock per day — long-only in V1, for reasons covered in Section 2.
3. Backtests that signal rigorously, with real Indian trading costs included, to get an honest read on whether it has any edge.
4. Runs the signal forward in a virtual/paper portfolio against live prices, so performance can be judged out-of-sample before any real capital is involved.
5. Produces a daily report Feron can read and manually act on — the system never places real orders.

**What "success" actually means here:** not "the system makes money in backtest" (that's trivial to produce and easy to overfit into existing). Success means the system survives walk-forward validation, remains profitable after realistic costs, and holds up over several months of live paper trading before a single real trade is placed. Anything short of that is a research result, not a trading strategy.

---

## 2. Scope

**In scope (V1):**
- Data pipeline for a defined universe of liquid NSE stocks
- Feature engineering from OHLCV only (price action + volume derived)
- A baseline ML model producing daily long/flat scores (long-only — see below)
- A proper backtesting engine with realistic Indian cost modeling
- A virtual/paper portfolio tracker that runs against live prices
- A daily signal report (for manual reading and manual execution only)

**Explicitly out of scope (V1):**
- Automated order placement via Angel One SmartAPI
- Options/derivatives strategies
- Intraday/HFT-speed strategies (this is a swing/positional system — daily or multi-day holding periods)
- Any claim of guaranteed or backtested-only performance being predictive of the future

**Long-only, and why (this isn't a modeling choice, it's a market-structure constraint):** Indian retail investors can only short-sell in the cash segment intraday — the position must be covered the same day. Holding a short position overnight requires either the SLB (Securities Lending & Borrowing) mechanism, which is impractical for retail at this scale, or F&O, which is explicitly out of scope for V1. Since this system is designed around multi-day swing holding periods, it must be **long-only** — the model should score and rank stocks for long entries, not generate short signals that can't actually be acted on the way the rest of this plan assumes (manual entry + GTT exit). If short exposure is ever wanted later, that's a V2+ decision that would need F&O brought into scope deliberately, with its own cost model and risk profile — not a passive side effect of the ranking model producing a low score.

**Possible V2 (only if V1 shows a real, cost-adjusted, paper-traded edge over several months):**
- Explore fine-tuning a pretrained financial foundation model (e.g., Kronos-style architecture) on NSE data, compared against the V1 baseline
- Broaden universe / add cross-sectional ranking across a larger stock set

---

## 3. High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Data Layer     │ --> │ Feature Engine    │ --> │  Model / Scorer    │
│ (SmartAPI + NSE) │     │ (price/volume     │     │ (baseline ML,      │
│                  │     │  derived only)    │     │  daily signal)     │
└─────────────────┘     └──────────────────┘     └─────────┬──────────┘
                                                              │
                     ┌────────────────────────────────────────┘
                     v
         ┌───────────────────────┐        ┌─────────────────────────┐
         │  Backtesting Engine    │        │  Virtual Portfolio /     │
         │  (walk-forward, cost-  │        │  Paper Trading Tracker   │
         │  aware, no lookahead)  │        │  (live prices, no real   │
         │                        │        │  orders placed)          │
         └───────────────────────┘        └───────────┬─────────────┘
                                                          │
                                                          v
                                              ┌───────────────────────┐
                                              │  Daily Signal Report   │
                                              │  (you read, you decide,│
                                              │   you manually trade)  │
                                              └───────────────────────┘
```

---

## 4. Data Layer

**Sources:**
- **Angel One SmartAPI** — historical OHLCV (free), via the historical candle endpoint. Known per-request limits by interval (verify current values against live docs, as these have changed before): ONE_DAY candles allow long lookback windows (historically ~2000 days/request), intraday intervals have shorter windows and need pagination. **Important: SmartAPI's candle data is raw/unadjusted — it does not backward-adjust for splits or bonuses.** Treat this as confirmed, not assumed, and validate it explicitly in Phase 0 (Section 13) by checking a known historical split/bonus date against the raw series.
- **Corporate action data (for adjustment)** — use the **EOD2 project** (github.com/BennyThadikaran/eod2) as the primary source for adjusted OHLCV and NSE holiday calendar data. EOD2 solves both gaps in one dependency and avoids building adjustment logic from scratch. **Treat it as a single source that must be validated, not trusted blindly:** in Phase 0, cross-check its adjustments against at least 3 known historical split/bonus events, and run periodic drift-checks (quarterly or after any known corporate action) comparing EOD2-adjusted prices against SmartAPI raw + bhavcopy to catch any silent staleness or maintainer-lag. If EOD2's adjustments fail validation for a stock, fall back to manual adjustment from NSE/BSE corporate actions pages for that stock rather than silently using bad data.
- **Instrument master** — SmartAPI's instruments list to map trading symbols to `symboltoken` values (required for every historical/live data call).
- **Cross-check source** — pull the same date range from a second source (e.g., NSE bhavcopy archives) for a sample of stocks to catch data quality issues (missing days, bad corporate-action adjustments) before trusting the primary feed.
- **NSE trading holiday calendar** — the daily cron job needs this to skip non-trading days gracefully instead of treating "no data today" as a data-quality failure. Pull from NSE's official holiday calendar page; note that NSE has added holidays mid-year before (a 2026 addition after the annual calendar was published is a documented example), so refresh this periodically rather than hardcoding it once.

**Requirements:**
- Store raw OHLCV + adjusted OHLCV (for splits/bonuses) separately — never silently overwrite raw data.
- Corporate action adjustment logic must be explicit and auditable (log every adjustment applied).
- Minimum history: at least 5–7 years of daily data per stock for meaningful walk-forward testing; more for any intraday work.
- Data validation checks: gaps, duplicate timestamps, zero-volume days, price jumps inconsistent with known corporate actions.

**Universe selection — criteria, not picks:**
Claude Code should build a **screener**, not a hardcoded list:
- Filter to F&O-eligible or Nifty 50/100/200 constituents (liquidity, tight spreads, clean long history)
- Minimum average daily traded value threshold (configurable — start high, e.g., top-200 by turnover)
- Exclude stocks with a corporate action (split/bonus/merger) in the last N months from the initial test set, to avoid early data-quality noise
- Sector cap (e.g., no more than 25% of universe from one sector) to avoid the model just learning a sector regime

**Concrete starting point:** Use the **Nifty 200 constituent list** as the raw pool, then apply the liquidity/corporate-action/sector filters above to shrink it to a working universe (likely 80–150 names). This isn't a stock pick — it's a well-defined, liquid, long-history index membership list that's easy to pull programmatically and re-derive each quarter as constituents change. Concrete source: NSE Indices publishes this directly as a CSV at `niftyindices.com/IndexConstituent/ind_nifty200list.csv` — fetch it fresh each time rather than hardcoding today's list into the codebase, since URLs and formats on this domain have shifted before and should be re-verified when Claude Code implements this.

**Sector classification source:** the sector cap needs stock-to-sector mapping, which isn't in the SmartAPI instrument master. The same niftyindices.com constituent-CSV pattern used above also exists for NSE's sectoral indices (Nifty Bank, Nifty IT, Nifty Auto, Nifty Pharma, Nifty FMCG, Nifty Metal, Nifty Realty, Nifty Energy, etc.) — cross-referencing which sectoral index each stock belongs to is a practical, verifiable way to build the sector map without a separate paid classification data source.

**Survivorship bias — a decision to make, not just a risk to note:** getting full price history for stocks that were delisted or dropped out of the index entirely is genuinely hard via SmartAPI (their instrument master only covers currently-listed tokens). Two options: (a) use a historical (survivorship-bias-free) index constituent snapshot — sources like niftyhistory.in publish Nifty 200 membership as of each historical rebalancing date, which at least ensures the *universe* reflects what was actually investable at each point in time, even without full delisted-stock price series, or (b) accept and explicitly document the remaining bias as a known V1 limitation rather than solving it fully. Given the effort-to-value ratio for a solo project, (a) combined with documenting the residual gap is the pragmatic choice — full delisted-stock history from a paid third-party source isn't worth it for V1.

---

## 5. Feature Engineering (OHLCV-derived only, per the original brief)

Build these as a versioned feature library, not ad hoc:

- **Price action:** multi-horizon returns (1, 5, 10, 20, 60 day), gap-open size, candle body-to-range ratio, upper/lower wick ratio, rolling high/low position (where is price relative to N-day range)
- **Volatility:** ATR, realized volatility at multiple windows, volatility-of-volatility (regime shift proxy)
- **Volume:** volume relative to rolling average, OBV, price-volume divergence, VWAP deviation
- **Momentum/mean-reversion (as features, not standalone signals):** RSI, MACD, Bollinger %B, distance from moving averages
- **Regime flags:** simple trend-vs-range classifier (e.g., ADX-based), so the model can condition behavior on regime rather than assuming one regime always
- **Market-wide context (important — don't skip this):** a single stock's candles behave very differently depending on the broader market regime. Add index-level features — Nifty/Sensex trend and volatility, India VIX level and change, and the stock's own beta/correlation to the index over a rolling window. Without this, the model has no way to distinguish "this stock looks bullish in isolation" from "this stock looks bullish only because the whole market is up" — a meaningfully different signal. **India VIX is available on SmartAPI** as an NSE index instrument (trading symbol "India VIX", symboltoken 99926017 as of recent instrument dumps — confirm against the current instrument master before relying on it, as tokens can occasionally change) via the same historical-candle and quote endpoints used for equities, so no separate data source is needed for this feature.

Every feature must be computed using only data available at time *t* — no leakage from future bars. Claude Code should write an automated leakage test (shift-check) that fails the build if any feature accidentally uses future data.

---

## 6. Modeling Layer (phased)

**V1 — baseline, ship this first:**
- Gradient-boosted trees (LightGBM or XGBoost) or a simple logistic/linear ranking model predicting forward returns per stock in the universe
- Cross-sectional ranking approach (rank stocks against each other each day) tends to be more robust than absolute return prediction — recommend this framing
- **Long-only output** (per Section 2): the model ranks the universe and surfaces long candidates from the top of the ranking that clear the confidence bar. Low-ranked/bearish-scored stocks are simply not surfaced as trades — they aren't converted into short signals, since those aren't practically actionable in V1's manual, cash-delivery execution model.
- Fully interpretable, fast to iterate, cheap to run on CPU — appropriate for a solo project with a day job

**Labeling — use the triple-barrier method, not a naive fixed-horizon return.** This is standard practice in quantitative finance ML (from López de Prado's widely-used framework) and fixes a real flaw in the simpler "predict the N-day forward return" approach: instead of always measuring the return exactly N days later regardless of what happened in between, define three barriers for every entry point — an upper barrier (take-profit), a lower barrier (stop-loss), and a time barrier (max holding period) — and label the outcome by whichever is hit first. This produces labels that reflect how a position would actually be exited, not an arbitrary fixed-day snapshot, and avoids labels that ignore intra-period drawdowns a real stop-loss would have triggered.
- **Critical: compute barriers and labels on net-of-cost returns, not gross price moves.** Subtract round-trip costs (brokerage + STT + stamp duty + exchange charges + GST + assumed slippage on both entry and exit) before checking which barrier was hit. A price move that looks like a win before costs can be a wash or a real loss after them — this matters more on lower-priced stocks where flat/minimum fee components are a bigger relative bite. This means the model gets trained to find trades that clear a genuine profit bar, not just a directionally correct price move. The stop-loss and target levels shown in the daily report (Section 9) should be the same cost-aware levels — not raw price targets that still need a mental cost adjustment from you.
- **Meta-labeling (worth adding once V1's basic direction model works):** train a primary model to predict direction (long/short/flat), then a secondary model to predict whether to *act* on that primary signal and at what size — this separates "which way" from "how much conviction," and tends to improve precision without needing a more complex primary model.

**Confidence gating is a hard requirement, not an optional refinement.** The system must be able to output *zero* qualifying signals on a given day, and this should be the common case, not the exception. Set an explicit score/probability threshold during backtesting — tuned against historical precision, not against "how many signals does this produce" — and do not adjust it after deployment based on how few or many signals show up in a given week. A system that always forces out a fixed number of daily picks "to have something to show" is one of the most common ways these projects quietly turn into noise generators. No qualifying trade on a given day is a correct output, not a failure of the system.

**V2 — only pursue if V1 clears the bar in Section 12:**
- Fine-tune a sequence/foundation model on NSE candle data (this is the Kronos-style approach) and compare head-to-head against the V1 baseline on the *same* walk-forward splits and *same* cost model
- Requires GPU access (rent, don't buy, for this stage) and materially more engineering time — not worth it until V1 proves the basic approach has legs

---

## 7. Backtesting Engine

**Non-negotiable requirements:**
- **Walk-forward / expanding-window validation** — never a single train/test split. Retrain periodically (e.g., quarterly) using only data available up to that point. Whatever retraining cadence is validated in backtest must be carried over unchanged into live/paper operation (Section 9) — if the backtest assumed quarterly retraining, the live system retrains quarterly too, not on an ad hoc schedule decided later.
- **Purging and embargoing between train/test splits** — because features and triple-barrier labels both use rolling/forward windows, adjacent train and test samples can overlap in time even in a "correct" walk-forward split, which leaks information. Purge any training samples whose label window overlaps the test period, and add a small embargo buffer (e.g., a few weeks) immediately after each test window before training resumes. This is the single most common correctness gap in DIY quant backtests — worth getting right before trusting any result.
- **Deflated Sharpe ratio / awareness of multiple-testing:** if you (or Claude Code) end up trying many feature combinations, model variants, or parameter settings before landing on "the" model, the reported backtest Sharpe of the winner is optimistically biased purely from the number of variants tried — this is the same statistical trap as p-hacking. Track how many variants were tested and discount the final Sharpe accordingly, or use a formal deflated Sharpe ratio calculation. This matters more than most people building solo projects assume.
- **No lookahead bias** — signal generated using bar *t* data can only be acted on at bar *t+1* open (or later), never at bar *t* close.
- **Realistic fills** — assume execution at **next-bar open**. Locked for V1 — no VWAP, no intraday data dependency.
- **Full Indian cost model built in from day one**, not bolted on later:
  - Brokerage (use your actual Angel One rate/plan)
  - STT (different rates for delivery vs intraday vs F&O — model correctly per trade type)
  - Stamp duty
  - Exchange transaction charges
  - SEBI turnover fees
  - GST on brokerage + transaction charges
  - Slippage assumption: **0.1% each way (entry + exit) for Nifty 100 names; 0.2% each way for Nifty Midcap 100 names.** Locked as V1 defaults — to be tightened (not loosened) as real paper-trading fill data comes in. Do not widen these after deployment based on bad fills; that's a sign the position size or universe filter needs revisiting, not that the slippage model was wrong.
- **Output metrics:** CAGR, Sharpe, Sortino, max drawdown, hit rate, average win/loss, turnover, and — critically — all of the above **before and after costs**, side by side, so the cost drag is never hidden.

**Suggested tools:** `vectorbt` or `backtrader` for the engine; `pandas`/`numpy` for the feature pipeline. Qlib (used in Kronos's own reference pipeline) is an option if Claude Code is comfortable with it, but it has a steeper setup cost — don't let tooling choice block getting a working V1 backtest running quickly.

---

## 8. Virtual Portfolio / Paper Trading Layer — dual-ledger design

Run **two separate, explicitly tracked ledgers** off the same signal engine:

**Ledger A — Autonomous Virtual Portfolio:**
- Runs entirely on the system's own signals and the system's own suggested position size — completely independent of what you actually do. Long-only, consistent with Section 2 — every position is a long entry with a GTT-style stop-loss/target, never a short.
- **Position sizing must be confidence-weighted, not flat.** Tie suggested size to the model/meta-model's confidence score (e.g., tiered sizing bands, or a fractional-Kelly-style scaling capped at a conservative max per trade) — a high-confidence signal and a barely-qualifying one shouldn't get the same allocation.
- Assumes a defined **starting virtual capital base**, ideally matching whatever real capital you'd eventually consider deploying, so the comparison to Ledger B later is apples-to-apples.
- Enforces **portfolio-level risk constraints**, not just per-trade stop-loss/target. Locked defaults:
  - **Max 5 concurrent open positions.**
  - **Max 80% of virtual capital deployed at any time** — the 20% cash buffer is a structural liquidity reserve, not idle money to be optimized away.
  - **Max 2 of the 5 positions from the same sector** — the universe-level sector cap (Section 4) ensures no sector dominates the pool, but doesn't prevent 5 signals coincidentally landing in one sector on the same day. This per-portfolio sector constraint closes that gap directly.
  - These defaults are starting points for backtesting. They may be tightened during Phase 3 if the backtest reveals concentration risk, but should not be loosened without a specific validated reason.

**Ledger B — Actual Portfolio:**
- You log what you actually bought: ticker, real fill price, quantity, date, and your own GTT levels — regardless of whether it matches Ledger A's signal, sizing, or timing.
- Also log signals you *declined* to act on, not just the ones you took — an accurate comparison needs the full picture of what you skipped, not just a survivorship-biased list of trades taken.

**Why both matter:** comparing Ledger A (disciplined, system-only) against Ledger B (your real behavior) over the evaluation window tells you something genuinely useful about yourself — whether your manual judgment on top of the signals adds value or quietly subtracts from it (hesitation, worse entries, second-guessing good signals). That comparison is itself part of the "is this system worth using" answer, not just the raw backtest numbers.

- Should run continuously for a minimum evaluation window before any capital decision is made (see Section 12).

---

## 9. Manual Execution & Compliance Boundary

This is intentional, not a limitation to work around later:

- SmartAPI is used **only for data** (historical + live quotes) in this system — never for order placement.
- The daily signal report is something you read and act on by manually placing the trade yourself through the Angel One app.
- This keeps the system on the manual-trading side of SEBI's April 2026 retail algo framework, which specifically targets orders placed/modified/cancelled automatically without manual confirmation — pure manual execution based on your own read of a report is unaffected by that framework.
- Caveat: this is based on current public guidance, not a legal opinion. If you ever consider adding automated order placement later, confirm the compliance boundary directly with Angel One first, since brokers now carry responsibility for algo classification.

**Daily operating flow:**
1. After market close, the system refreshes data, recomputes features, and scores the universe.
2. The confidence gate is applied (Section 6) — most days this should yield zero or a small handful of qualifying signals.
3. A report is generated before the next market open: for each qualifying stock — direction, entry zone, stop-loss level, target level (both already net-of-cost, per Section 6 — not raw price targets you need to mentally adjust), expected max holding period, and confidence score.
4. You review it and decide. If you act: place the entry manually through the Angel One app, and immediately set a **GTT order with an OCO (one-cancels-other) stop-loss + target** using the report's levels. Angel One's GTT stays active until triggered (up to ~365 days) and handles the exit automatically at the broker level — no intraday monitoring or system-side order placement required once it's set.
5. Log what you actually did (ticker, fill price, quantity, GTT levels) somewhere the system can read, so the virtual/paper tracker can reconcile its assumed fills against your real execution and quantify the gap (slippage, timing drag).

**Why after-close timing is correct, not a compromise:** the model's features (candle body ratio, close-relative-to-range, etc.) are only well-defined once a day's candle is fully complete. Scoring mid-day on a partial candle would feed the model a materially different kind of input than it was trained and validated on — a classic train/serve mismatch that degrades accuracy rather than improving it. This isn't a speed limitation to work around; it's the technically correct cadence for a daily-bar model. A genuinely real-time/intraday system would require a different model entirely (tick-level features, different cost sensitivity, and it starts approaching SEBI's order-frequency thresholds from Section 9) — that's a separate future project, not a variant of this one.

**Opening-gap sanity check (addresses the real risk without redesigning the system):** before acting on a signal, compare the actual market-open price to what the signal assumed at generation time. If the stock has already gapped beyond the planned entry zone, the report should flag it as "conditions changed since signal generation — re-evaluate" rather than presenting it as still-valid. This catches the genuine "did something change overnight" risk without needing real-time scoring. **Default tolerance: flag if the actual open is more than 0.5% beyond the planned entry zone** — configurable, but 0.5% is a reasonable starting point that won't flag routine noise while still catching genuine overnight gaps.

---

## 10. Tech Stack

- **Language:** Python (matches SmartAPI's official SDK support and the broader quant tooling ecosystem)
- **Data storage:** Parquet files or a local DuckDB/SQLite database — no need for a heavy DB at this scale
- **Feature/backtest:** pandas, numpy, LightGBM/XGBoost, vectorbt or backtrader
- **Scheduling:** a simple cron job or scheduled script for daily data pull + signal generation (no need for anything fancier at V1 scale)
- **Hosting:** the daily job needs to run on something that's actually on at the scheduled time — a personal laptop that might be asleep or shut isn't reliable. Use a small always-on host (a low-cost cloud VM — AWS/GCP/Oracle free-tier tiers are all sufficient for this workload — or a home Raspberry Pi/mini-PC left running). **Use a static (not dynamic/elastic-on-restart) IP for the VM** even though it isn't currently mandated for data-only APIs — it avoids a class of silent auth failures if the VM ever reboots and picks up a new address, and keeps things simple if requirements ever tighten later.
- **Daily re-authentication:** Angel One SmartAPI sessions expire daily at midnight regardless of activity — the scheduled job needs a scripted re-login (client code + MPIN + TOTP) at the start of each run, not a token assumed to persist. Concretely, this means generating the TOTP code programmatically each run (e.g., Python's `pyotp.TOTP(secret).now()`, using the secret from Angel One's "Enable TOTP" setup) rather than any static or manually-entered code. Since this system uses SmartAPI purely for data (never Orders/GTT endpoints), the static-IP whitelisting SEBI/Angel One require for order-placing APIs does **not** apply here — one less piece of infrastructure to manage.
- **Cost model reference point:** as of Angel One's own published rate card, equity delivery is currently charged at ₹20 or 0.1% of order value (whichever is lower, ₹5 minimum) per executed order, plus DP charges (~₹20 + ~₹5.50 CDSL fee per scrip on the sell side) — this replaced their older zero-brokerage-delivery model. Rates do change, so confirm your actual current plan on Angel One's charges page (Account → Trades & Charges) before finalizing the cost model rather than trusting any single external source, including this one.
- **Reporting — two-tier design:**
  - **Push alert (daily, mobile-first):** email or Telegram/Discord message sent automatically after signal generation — the "check this now before market open" notification. Contains only the qualifying signals (if any), levels, and confidence scores. No port exposure, no SSH required.
  - **Local dashboard (deliberate, deeper):** the full Ledger A/B views, backtest results, universe snapshot, and settings (Section 17). Not something you check daily — something you open when you want to dig in. Served from the VM but accessed deliberately, not pushed.

---

## 11. Development Phases

1. **Phase 0 — Feasibility checks** (see Section 13 checklist) — 1–2 weeks
2. **Phase 1 — Data pipeline + universe screener** — get clean, validated OHLCV for the chosen universe
3. **Phase 2 — Feature library + leakage tests**
4. **Phase 3 — V1 baseline model + walk-forward backtest with full cost model**
5. **Phase 4 — Review backtest results honestly against Section 12 criteria before proceeding**
6. **Phase 5 — Virtual/paper trading harness, run live for the full evaluation window**
7. **Phase 6 — Only after Phase 5 clears the bar: decide whether to commit real (small) capital, manually executed**

Don't let Claude Code skip from Phase 3 straight to "looks great, let's go live" — Phase 4 is a deliberate gate.

---

## 12. Success Metrics & Kill Criteria

Set these **before** looking at any results, so you're not tempted to move the goalposts. **Evaluate these against Ledger A (the autonomous virtual portfolio, Section 8), not Ledger B.** Ledger A isolates whether the system itself works; Ledger B reflects your behavior on top of it, which is a separate, useful question but shouldn't be conflated with the system's own validation.

- Minimum backtest period: at least 5 years, spanning at least one significant market drawdown (so the model is tested outside a pure bull regime)
- Minimum paper-trading period before considering real capital: **3–6 months**, not weeks
- **Retraining cadence: locked in at quarterly** (roughly every 63 trading days), consistent across backtest and live operation (Section 7) — not to be adjusted after seeing results.
- **Starting virtual capital for Ledger A: ₹1 lakh.**
- **Post-cost Sharpe ratio threshold: 1.0, applied strictly to the live paper-trading result (Ledger A), not the backtest number.** Reference point: the Nifty 50 itself has historically run a Sharpe around 0.4–0.6 over long periods, so 1.0 represents a real, meaningful improvement over just holding the index — not an arbitrary round number. The backtest Sharpe is a separate, less trustworthy number, since it's exactly what gets inflated by however many feature/parameter variants get tried (Section 7's deflated-Sharpe point) — treat anything under ~1.3 in backtest with real suspicion before even entering the paper-trading phase. The paper-trading result can't be retroactively cherry-picked, which is why the strict 1.0 bar belongs there.
- **Max acceptable drawdown in the virtual book: 20%.** This system has per-trade stop-losses and isn't always fully invested, so it should structurally draw down less than buy-and-hold — for reference, the Nifty 50 has seen drawdowns of roughly 23% (2020) and 38% (2008). 20% is set as the trigger point to pause and reassess, not to treat as ordinary volatility to ride out.
- **Kill criteria, decided in advance:** if paper-traded performance materially diverges from backtest (worse Sharpe, higher drawdown, signal decay), that's information — the honest move is to stop or revisit, not to keep tweaking parameters until the paper period looks good too (that's just overfitting on a longer window).

---

## 13. Feasibility Checklist (run these early, in Phase 0)

- [ ] Confirm current Angel One SmartAPI historical data limits per interval (rate limits and max-days-per-request change over time — check live docs, not this document)
- [ ] Explicitly verify SmartAPI's historical data is unadjusted by checking a known historical split/bonus date against the raw series — confirm EOD2's adjustments are correct for the same events (cross-check at least 3 known split/bonus dates) before trusting any backtest
- [ ] Set up EOD2 drift-check: quarterly comparison of EOD2-adjusted prices vs SmartAPI raw + bhavcopy for a sample of stocks — make this a scheduled check, not a one-off
- [ ] Build and test the scripted daily re-authentication flow (client code + MPIN + TOTP) end-to-end before relying on the cron job — sessions expire daily at midnight, so this has to work unattended every single run
- [ ] Pull a test sample of 5 years of daily data for 10 stocks and validate against a second source
- [ ] Confirm instrument token mapping works reliably for your target universe, including the India VIX token (Section 5)
- [ ] Confirm the NSE holiday calendar source is wired in so the cron job skips non-trading days without flagging false data gaps
- [ ] Build and run the leakage-detection test on the feature library before trusting any backtest result
- [ ] Run a "dumb baseline" backtest (e.g., buy-and-hold, or a simple moving-average crossover) through the same engine first, to confirm the backtest engine itself produces sane, trustworthy numbers before testing the real model
- [ ] Confirm the cost model matches your actual, current Angel One brokerage plan and current STT/stamp duty rates (Section 10 has a starting reference point, but confirm directly)

---

## 14. Known Risks & Limitations (be honest with yourself about these)

- **Overfitting risk is the single biggest threat** — with enough features and enough historical data, it's trivial to find a "strategy" that looks great in backtest and means nothing.
- **Regime shift** — a pattern that worked 2015–2023 may simply stop working; markets adapt as more participants exploit the same signals.
- **Candle-shape pattern recognition alone has a genuinely mixed evidence base, and it splits along an interesting line.** Studies on mature, highly efficient markets have generally found weak or no edge — <cite index="48-1">Marshall, Young and Rose (2006) concluded that 28 common candlestick patterns offered no real edge when tested on Dow 30 stocks over a decade</cite>, and similar null results have been found on European instruments. But studies on less mature, more retail-dominated markets have found more consistent (though still not universal) profitability after transaction costs — research on <cite index="50-1">the 30 component stocks of the DJIA combined with corrections for data-snooping bias found some candlestick strategies profitable after transaction costs</cite>, and separately, <cite index="49-1">a study of the Malaysian stock market from 2000–2014 found that bullish reversal patterns remained profitable for investors even after accounting for transaction costs and out-of-sample testing</cite>. That market-maturity split is actually a reasonable basis for testing this on Indian data specifically rather than assuming it's a settled "doesn't work" question — India's retail-dominated market structure has more in common with the markets where residual edges were found than with the Dow. That said, "mixed evidence with some positive findings in similar markets" is a reason to test rigorously, not a reason to expect it'll definitely work — treat it as a hypothesis worth a properly validated test, not a known result.
- **Survivorship bias** — addressed via the historical-constituent-snapshot approach in Section 4, with the residual gap (full delisted-stock price history) explicitly accepted and documented rather than silently ignored.
- **Liquidity constraints** — a strategy that backtests well on wider small/mid-cap universes may not be executable at real size without moving the price; this is part of why the universe screener (Section 4) matters.
- **Tax treatment isn't a code problem, but it affects the real economics — check it before deploying real capital.** Frequent trading (even swing-frequency, not intraday) can get classified differently for Indian tax purposes than simple buy-and-hold investing — potentially as business income rather than capital gains, which changes the applicable tax treatment and may bring in other compliance considerations depending on volume. This is genuinely worth a conversation with a CA before Phase 6, not something to guess at from a backtest.

---

## 15. Research Basis for This Plan

This plan isn't improvised — the methodology choices above are drawn from how this is actually done in practice, not just first-principles engineering:

- **Purged/embargoed cross-validation, triple-barrier labeling, meta-labeling, and the deflated Sharpe ratio** (Sections 6–7, 12) come from Marcos López de Prado's *Advances in Financial Machine Learning*, which is the standard reference practitioners and quant funds use specifically to avoid the overfitting traps that sink most DIY quant projects. <cite index="40-1">Purged cross-validation was developed specifically to prevent look-ahead bias in financial time series, as an alternative to conventional cross-validation and walk-forward backtesting, which often yield overly optimistic performance estimates due to information leakage and overfitting.</cite>
- **The Kronos paper you originally linked** confirms candlestick/OHLCV sequence modeling is an active, credible research area — foundation-model approaches on this exact data type are what motivated the V2 stretch goal in Section 6.
- **The candlestick-pattern academic literature** (Section 14) is what informs the honest framing that this is a testable hypothesis with some supporting evidence in comparable markets, not a proven strategy.

If Claude Code (or you) want to go deeper on the validation methodology specifically, "triple-barrier method," "purged cross-validation," and "meta-labeling" are the exact search terms that lead back to this literature.

---

## 16. Note for Claude Code

When implementing this, prioritize correctness and leakage-safety over sophistication. A simple, correctly-validated V1 that shows no edge is a more useful and honest outcome than a complex V1 that "backtests amazingly" due to a subtle bug. Build the leakage tests and the dumb-baseline sanity check (Section 13) before trusting any result from the real model.

---

## 17. UI / Reporting Layer

Keep this simple — it's a single-user personal tool, not a product. A local dashboard (or even a set of generated static HTML/markdown pages refreshed daily) with these tabs is enough:

1. **Today's Signals** — the daily report: qualifying stocks (if any), direction, cost-aware entry/stop-loss/target levels, confidence score, suggested position size, and the opening-gap check status once the market opens.
2. **Ledger A — Virtual Portfolio** — current open virtual positions, closed trade history, equity curve, running P&L shown both gross and net of costs, and headline metrics (Sharpe, max drawdown, hit rate).
3. **Ledger B — Actual Portfolio** — your logged real trades and real P&L, plus a simple input form to log a new trade or mark a signal as declined.
4. **Comparison / Reconciliation** — Ledger A vs Ledger B side by side: where they diverge and why (timing, sizing, declined signals that would have won or lost).
5. **Backtest & Research** — walk-forward backtest results, before/after cost comparison, and a place to review results each time the model is retrained (this is where Phase 4's gate actually gets exercised).
6. **Universe & Settings** — current filtered universe list, each stock's latest feature snapshot, and the configuration that matters to keep visible: confidence threshold, cost model assumptions (your actual brokerage plan, slippage assumption), position sizing parameters. Visible for transparency — but per Section 6, the threshold shouldn't be casually retuned mid-evaluation just because it's editable.

That's enough surface area for a functioning personal tool. Nothing here needs to be more polished than a clean local dashboard.