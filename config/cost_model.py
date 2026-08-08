"""
Full Indian cost model for NSE equity delivery trades.

Rates as of 2026 — verify against Angel One's charges page before Phase 3.
All rates are per-side unless noted as round-trip.

Sources:
  Angel One brokerage: ₹20 or 0.1% (whichever lower, ₹5 min) per order
  STT delivery: 0.1% of turnover on SELL side only
  Stamp duty: 0.015% of buy-side turnover (Maharashtra; varies by state but 0.015% is standard)
  Exchange txn charge: 0.00335% each side (NSE equity)
  SEBI turnover fee: 0.0001% each side
  GST: 18% on (brokerage + exchange txn charge + SEBI fee)
  DP charge: ~₹20 + ₹5.50 CDSL per scrip on sell side
  Slippage: configurable per universe tier (locked in plan)
"""
from dataclasses import dataclass
from config import settings


@dataclass(frozen=True)
class TradeResult:
    gross_pnl: float
    total_cost: float
    net_pnl: float
    cost_breakdown: dict

    @property
    def net_return(self) -> float:
        return self.net_pnl / self.entry_value if hasattr(self, "entry_value") else 0.0


def brokerage(order_value: float) -> float:
    """Angel One delivery brokerage: min(₹20, 0.1% of order_value), floor ₹5."""
    return max(5.0, min(20.0, order_value * 0.001))


def stt(turnover: float, side: str) -> float:
    """STT: 0.1% of turnover on SELL side only for delivery equity."""
    if side.upper() == "SELL":
        return turnover * 0.001
    return 0.0


def stamp_duty(turnover: float, side: str) -> float:
    """Stamp duty: 0.015% of buy-side turnover only."""
    if side.upper() == "BUY":
        return turnover * 0.00015
    return 0.0


def exchange_txn_charge(turnover: float) -> float:
    """NSE equity segment exchange transaction charge: 0.00335% per side."""
    return turnover * 0.0000335


def sebi_fee(turnover: float) -> float:
    """SEBI turnover fee: 0.0001% per side."""
    return turnover * 0.000001


def gst_on_charges(brok: float, exc: float, sebi: float) -> float:
    """18% GST on brokerage + exchange txn charge + SEBI fee."""
    return (brok + exc + sebi) * 0.18


def dp_charge(side: str) -> float:
    """DP charge on sell side: ₹20 (Angel One) + ₹5.50 (CDSL) per scrip."""
    return 25.50 if side.upper() == "SELL" else 0.0


def slippage_cost(quantity: int, price: float, symbol_tier: str) -> float:
    """
    Slippage per side (locked plan defaults):
      Nifty 100 names: 0.1% of trade value each way
      Nifty Midcap 100 names: 0.2% of trade value each way
    """
    rates = {"nifty100": 0.001, "midcap100": 0.002}
    rate = rates.get(symbol_tier.lower(), 0.002)
    return quantity * price * rate


def per_side_cost(quantity: int, price: float, side: str, symbol_tier: str = "midcap100") -> float:
    """Total cost for one side of a trade (buy or sell)."""
    turnover = quantity * price
    brok = brokerage(turnover)
    exc = exchange_txn_charge(turnover)
    sebi = sebi_fee(turnover)
    return (
        brok
        + stt(turnover, side)
        + stamp_duty(turnover, side)
        + exc
        + sebi
        + gst_on_charges(brok, exc, sebi)
        + dp_charge(side)
        + slippage_cost(quantity, price, symbol_tier)
    )


def round_trip_cost(quantity: int, entry_price: float, exit_price: float,
                    symbol_tier: str = "midcap100") -> dict:
    """
    Full round-trip cost breakdown for a completed trade.
    Returns dict with per-component and total costs.
    """
    entry_turnover = quantity * entry_price
    exit_turnover = quantity * exit_price

    entry_brok = brokerage(entry_turnover)
    exit_brok = brokerage(exit_turnover)

    entry_exc = exchange_txn_charge(entry_turnover)
    exit_exc = exchange_txn_charge(exit_turnover)

    entry_sebi = sebi_fee(entry_turnover)
    exit_sebi = sebi_fee(exit_turnover)

    breakdown = {
        "brokerage":      entry_brok + exit_brok,
        "stt":            stt(exit_turnover, "SELL"),
        "stamp_duty":     stamp_duty(entry_turnover, "BUY"),
        "exchange_txn":   entry_exc + exit_exc,
        "sebi_fee":       entry_sebi + exit_sebi,
        "gst":            (gst_on_charges(entry_brok, entry_exc, entry_sebi)
                           + gst_on_charges(exit_brok, exit_exc, exit_sebi)),
        "dp_charge":      dp_charge("SELL"),
        "slippage":       (slippage_cost(quantity, entry_price, symbol_tier)
                           + slippage_cost(quantity, exit_price, symbol_tier)),
    }
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def net_return(quantity: int, entry_price: float, exit_price: float,
               symbol_tier: str = "midcap100") -> float:
    """Net return after all costs, as a fraction of entry value."""
    gross = quantity * (exit_price - entry_price)
    costs = round_trip_cost(quantity, entry_price, exit_price, symbol_tier)["total"]
    entry_value = quantity * entry_price
    return (gross - costs) / entry_value


def cost_adjusted_barrier(entry_price: float, gross_barrier_pct: float,
                           direction: str, quantity: int = 100,
                           symbol_tier: str = "midcap100") -> float:
    """
    Convert a gross price barrier (e.g. +6% target) into a net-of-cost level.
    Used for triple-barrier label computation and for the daily signal report.
    """
    costs = round_trip_cost(quantity, entry_price,
                            entry_price * (1 + gross_barrier_pct),
                            symbol_tier)["total"]
    cost_per_share = costs / quantity
    if direction.upper() == "UP":
        return entry_price * (1 + gross_barrier_pct) - cost_per_share
    else:
        return entry_price * (1 - gross_barrier_pct) + cost_per_share


def estimate_symbol_tier(symbol: str, nifty100_symbols: set) -> str:
    """Classify a symbol into cost tier based on index membership."""
    return "nifty100" if symbol in nifty100_symbols else "midcap100"
