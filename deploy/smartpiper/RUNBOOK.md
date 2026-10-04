# smartpiper — real-money runbook (manual execution)

> **CURRENT STATUS: PAPER TRADING ONLY (decided 2026-10-04).** Nothing below needs to be done while on paper — the three paper
> books fill themselves each evening. This runbook is kept for the day real money is considered (suggested only after 8+ weeks
> of paper results that track the backtest and a healthy "Edge check").

The primary system is the **smart 50/50 book** (50% ML rank + 50% momentum rank, 20 stocks, equal weight, weekly).
The system generates signals and an order ticket. **It never places orders** — you do, by hand, in your broker app.

## Weekly rhythm
| When | What |
|---|---|
| **Last trading day of the week, ~20:30 IST** | The daily job runs and writes the new signal + **order ticket**. Open the dashboard (`ssh -L 5010:localhost:5010 …` then http://localhost:5010) and read the ticket card. |
| **Next trading day, 15:00–15:30 IST** | Place the ticket's orders (delivery/CNC, limit within +0.5% of the reference price). The backtest assumes the close price; NSE's close is the VWAP of the last 30 minutes. |
| **Right after each fill** | `python -m smart.run record BUY|SELL SYMBOL QTY PRICE` (on the box, in `~/smartpiper`). The next ticket is built from what you actually hold. |
| Daily | Nothing. Typically 2–5 trades a week; many weeks fewer. |

## Rules (decided in advance — do not improvise)
1. **Follow the ticket.** Overriding picks by hand destroys the edge the backtest measured.
2. **Delivery only, no leverage, no margin, no intraday.**
3. A stock **locked at the upper circuit** (cannot buy): skip it, buy the next-ranked name. **Locked at the lower circuit** (cannot sell): sell it the next day.
4. An order above ~1% of the stock's normal daily traded value: split it over two days (the ticket warns).
5. **Size:** start with a fraction of the money you eventually intend to use (suggestion: a quarter) and scale up only if the live results track the paper results for 8+ weeks. Edge fades above ~₹5 crore.
6. Small accounts: with ₹2 lakh each slot is ₹10,000, so a few high-priced stocks cannot be bought — the ticket substitutes the next-ranked affordable name. ₹5 lakh+ avoids this.

## Pause rules (check the "Guardrails" line on the ticket card)
* **PAUSE new buys and review** if any: smart paper book drawdown from its peak ≥ **35%** (the backtest's worst); data stale (no new daily NAV for > 5 days — never trade on an old signal); the system health check reports FAIL.
* **REVIEW** if: drawdown ≥ 25%; or after ≥ 12 weekly decisions the **Edge check** shows the blend's top-20 21-day excess return below zero (the model's edge may have faded).
* A pause means: keep existing holdings, stop adding, investigate. It is not an instruction to sell everything.

## Honest expectations
* Backtest 2013–Oct 2026: +40% a year, worst drop −36%. **Plan for 20–25% a year, 30–35% drops along the way, and roughly two losing years in ten** (2018: −7%, 2025: −8% in the backtest).
* Short-term capital gains tax (currently 20%) is not included. Keep `real_trades.jsonl` for your records.
* 2026 so far: momentum has been the strongest half; ML-only the weakest. Two comparison books (momentum-only, ML-only) run beside it on paper.

## Commands (run in `~/smartpiper` with `.venv/bin/python`)
```
python -m smart.run status                       # all three paper books
python -m smart.run ticket [--capital 500000]    # rebuild the order ticket
python -m smart.run capital 500000               # set the capital the ticket sizes for
python -m smart.run record BUY BLSE 31 322.05    # record a fill you made
python -m smart.run scorecard                    # live edge check
```
