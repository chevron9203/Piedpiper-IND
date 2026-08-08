"""
Angel One SmartAPI order execution wrapper.

Supports both LIVE and PAPER modes:
  - PAPER mode (default): logs orders locally, no real API calls.
  - LIVE mode: places real orders via SmartConnect.

All intraday orders use product_type=MIS (Margin Intraday Square-off),
which Angel One auto-squares off by 3:15 PM if not already closed.

Usage:
    executor = AngelOrderExecutor(paper=True)   # paper mode
    executor = AngelOrderExecutor(paper=False)  # live mode

    order_id = executor.place_market_order("RELIANCE-EQ", "99926000", qty=10, side="BUY")
    executor.cancel_order(order_id)
    positions = executor.get_positions()
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger

IST = timezone(timedelta(hours=5, minutes=30))

# Angel One order constants
EXCHANGE     = "NSE"
EXCHANGE_NFO = "NFO"
PRODUCT      = "MIS"        # Intraday (auto square-off by 3:15 PM)
VARIETY      = "NORMAL"
VARIETY_SL   = "STOPLOSS"   # Required for SL-M orders — "NORMAL" causes rejection
DURATION     = "DAY"


class AngelOrderExecutor:
    """Wrap Angel One placeOrder / cancelOrder / positions for intraday use."""

    def __init__(self, paper: bool = True):
        self.paper = paper
        self._paper_orders: dict[str, dict] = {}   # order_id → order details (paper mode)
        if paper:
            logger.info("AngelOrderExecutor: PAPER mode — no real orders will be placed")
        else:
            logger.info("AngelOrderExecutor: LIVE mode — orders will be placed on Angel One")
            from data.auth import get_client  # noqa: PLC0415
            self._get_client = get_client

    # ── Order placement ────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        token: str,
        qty: int,
        side: str,              # "BUY" or "SELL"
    ) -> str:
        """Place a MARKET intraday order. Returns order_id."""
        order_id = str(uuid.uuid4())[:12]
        now = datetime.now(IST)

        if self.paper:
            self._paper_orders[order_id] = {
                "order_id":   order_id,
                "symbol":     symbol,
                "token":      token,
                "qty":        qty,
                "side":       side,
                "order_type": "MARKET",
                "product":    PRODUCT,
                "status":     "FILLED",
                "placed_at":  now,
                "fill_price": None,   # filled by caller after fetching LTP
            }
            logger.info("PAPER ORDER: {} {} {} qty={}", side, symbol, "MARKET", qty)
            return order_id

        # Live path
        client = self._get_client()
        params = {
            "variety":         VARIETY,
            "tradingsymbol":   symbol,
            "symboltoken":     token,
            "transactiontype": side,
            "exchange":        EXCHANGE,
            "ordertype":       "MARKET",
            "producttype":     PRODUCT,
            "duration":        DURATION,
            "quantity":        str(qty),
        }
        try:
            resp = client.placeOrder(params)
            if resp and resp.get("status"):
                real_id = resp["data"]["orderid"]
                logger.info("LIVE ORDER placed: {} {} {} qty={} → id={}", side, symbol, "MARKET", qty, real_id)
                return real_id
            else:
                raise RuntimeError(f"placeOrder failed: {resp}")
        except Exception as exc:
            logger.error("placeOrder error for {}: {}", symbol, exc)
            raise

    def place_sl_market_order(
        self,
        symbol: str,
        token: str,
        qty: int,
        side: str,
        trigger_price: float,
        exchange: str = EXCHANGE,   # override with EXCHANGE_NFO for futures SL orders
    ) -> str:
        """Place a Stop-Loss Market order (SL-M). Triggers at trigger_price."""
        order_id = str(uuid.uuid4())[:12]
        now = datetime.now(IST)

        if self.paper:
            self._paper_orders[order_id] = {
                "order_id":      order_id,
                "symbol":        symbol,
                "token":         token,
                "qty":           qty,
                "side":          side,
                "order_type":    "STOPLOSS_MARKET",
                "trigger_price": trigger_price,
                "product":       PRODUCT,
                "exchange":      exchange,
                "status":        "TRIGGER_PENDING",
                "placed_at":     now,
            }
            logger.info("PAPER SL-M ORDER: {} {} trigger={:.2f} qty={}", side, symbol, trigger_price, qty)
            return order_id

        client = self._get_client()
        params = {
            "variety":         VARIETY_SL,   # SL orders require "STOPLOSS" variety
            "tradingsymbol":   symbol,
            "symboltoken":     token,
            "transactiontype": side,
            "exchange":        exchange,
            "ordertype":       "STOPLOSS_MARKET",
            "producttype":     PRODUCT,
            "duration":        DURATION,
            "quantity":        str(qty),
            "triggerprice":    str(round(trigger_price, 2)),
            "price":           "0",
        }
        try:
            resp = client.placeOrder(params)
            if resp and resp.get("status"):
                real_id = resp["data"]["orderid"]
                logger.info("LIVE SL-M placed: {} {} trigger={} qty={} → id={}", side, symbol, trigger_price, qty, real_id)
                return real_id
            raise RuntimeError(f"placeOrder (SL-M) failed: {resp}")
        except Exception as exc:
            logger.error("SL-M order error for {}: {}", symbol, exc)
            raise

    def place_fno_market_order(
        self,
        trading_symbol: str,   # e.g. RELIANCE25AUG26FUT
        token:          str,
        qty:            int,   # in lots × lot_size (total shares)
        side:           str,   # "BUY" (close short) or "SELL" (open short)
    ) -> str:
        """Place a MARKET MIS order on NFO exchange (stock/index futures)."""
        order_id = str(uuid.uuid4())[:12]
        now = datetime.now(IST)

        if self.paper:
            self._paper_orders[order_id] = {
                "order_id":   order_id,
                "symbol":     trading_symbol,
                "token":      token,
                "qty":        qty,
                "side":       side,
                "order_type": "MARKET",
                "product":    PRODUCT,
                "exchange":   EXCHANGE_NFO,
                "status":     "FILLED",
                "placed_at":  now,
            }
            logger.info("PAPER NFO ORDER: {} {} {} qty={}", side, trading_symbol, "MARKET", qty)
            return order_id

        client = self._get_client()
        params = {
            "variety":         VARIETY,
            "tradingsymbol":   trading_symbol,
            "symboltoken":     token,
            "transactiontype": side,
            "exchange":        EXCHANGE_NFO,
            "ordertype":       "MARKET",
            "producttype":     PRODUCT,
            "duration":        DURATION,
            "quantity":        str(qty),
        }
        try:
            resp = client.placeOrder(params)
            if resp and resp.get("status"):
                real_id = resp["data"]["orderid"]
                logger.info("LIVE NFO ORDER: {} {} qty={} → id={}", side, trading_symbol, qty, real_id)
                return real_id
            raise RuntimeError(f"NFO placeOrder failed: {resp}")
        except Exception as exc:
            logger.error("NFO placeOrder error for {}: {}", trading_symbol, exc)
            raise

    def cancel_order(self, order_id: str, variety: str = VARIETY) -> bool:
        """Cancel an open order by order_id. Use variety=VARIETY_SL for SL-M orders."""
        if self.paper:
            if order_id in self._paper_orders:
                self._paper_orders[order_id]["status"] = "CANCELLED"
                logger.info("PAPER ORDER cancelled: {}", order_id)
                return True
            return False

        client = self._get_client()
        try:
            resp = client.cancelOrder(order_id, variety)
            ok = bool(resp and resp.get("status"))
            logger.info("cancel_order {}: {}", order_id, "OK" if ok else "FAILED")
            return ok
        except Exception as exc:
            logger.error("cancel_order error {}: {}", order_id, exc)
            return False

    # ── Positions ──────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Return list of current intraday positions (MIS product only)."""
        if self.paper:
            return []

        client = self._get_client()
        try:
            resp = client.position()
            if not resp or not resp.get("status"):
                return []
            positions = resp.get("data", []) or []
            return [p for p in positions if p.get("producttype") == "MIS"]
        except Exception as exc:
            logger.error("get_positions error: {}", exc)
            return []

    def get_ltp(self, exchange: str, symbol: str, token: str) -> float | None:
        """Fetch last traded price for a symbol."""
        if self.paper:
            return None   # caller uses close price from candle data instead

        client = self._get_client()
        try:
            resp = client.ltpData(exchange, symbol, token)
            if resp and resp.get("status"):
                return float(resp["data"]["ltp"])
            return None
        except Exception as exc:
            logger.warning("get_ltp error for {}: {}", symbol, exc)
            return None

    def square_off_all(self) -> int:
        """Market-sell all open MIS positions. Returns number closed."""
        if self.paper:
            logger.info("PAPER square_off_all — no real positions to close")
            return 0

        positions = self.get_positions()
        closed = 0
        for pos in positions:
            net_qty = int(pos.get("netqty", 0))
            if net_qty == 0:
                continue
            sym   = pos.get("tradingsymbol", "")
            tok   = pos.get("symboltoken", "")
            side  = "SELL" if net_qty > 0 else "BUY"
            qty   = abs(net_qty)
            try:
                self.place_market_order(sym, tok, qty, side)
                closed += 1
            except Exception as exc:
                logger.error("square_off failed for {}: {}", sym, exc)
        logger.info("square_off_all: {} positions closed", closed)
        return closed
