"""
Angel One SmartFeed WebSocket wrapper.

Subscribes to NSE stocks in QUOTE mode (mode 2) and fires on_tick callbacks
with parsed price/volume data. Handles reconnection automatically.

Usage:
    feed = LiveFeed(on_tick=my_callback, on_connect=my_connect_cb)
    feed.subscribe(tokens=["2885", "1594"])   # symboltoken strings
    feed.start()   # blocks; run in a thread
    feed.stop()
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Callable
from loguru import logger

IST = timezone(timedelta(hours=5, minutes=30))

# Tick dict keys (prices already divided by 100 — in ₹)
# {token, symbol, ltp, open, high, low, volume, timestamp}


class LiveFeed:
    """Thread-safe wrapper around SmartWebSocketV2 for live NSE equity ticks."""

    NSE_CM       = 1      # exchange_type for NSE cash market
    QUOTE_MODE   = 2      # gives LTP + OHLCV + volume

    def __init__(
        self,
        on_tick:    Callable[[dict], None],
        on_connect: Callable[[], None] | None = None,
        on_error:   Callable[[str], None]    | None = None,
    ):
        self._on_tick    = on_tick
        self._on_connect = on_connect
        self._on_error   = on_error

        self._token_to_symbol: dict[str, str] = {}
        self._subscribed_tokens: list[str]    = []
        self._ws = None
        self._stop_event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_symbol_map(self, token_to_symbol: dict[str, str]) -> None:
        """Map from symboltoken string → trading symbol (e.g. "2885" → "RELIANCE")."""
        self._token_to_symbol = token_to_symbol

    def subscribe(self, tokens: list[str]) -> None:
        """Set tokens to subscribe. Call before start(), or after connect."""
        self._subscribed_tokens = [str(t) for t in tokens]
        if self._ws is not None:
            self._send_subscribe(self._ws)

    def start(self) -> None:
        """Connect and block until stop() is called."""
        self._stop_event.clear()
        self._connect()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        from data.auth import _session as auth_session

        from config import settings as cfg
        auth_session.get_client()   # ensure logged in / refresh if expired
        auth_token  = auth_session.auth_token
        feed_token  = auth_session.feed_token
        api_key     = cfg.ANGELONE_API_KEY
        client_code = cfg.ANGELONE_CLIENT_CODE

        ws = SmartWebSocketV2(
            auth_token=auth_token,
            api_key=api_key,
            client_code=client_code,
            feed_token=feed_token,
            max_retry_attempt=5,
            retry_strategy=1,
            retry_delay=5,
            retry_multiplier=2,
            retry_duration=120,
        )

        ws.on_open    = self._make_on_open(ws)
        ws.on_data    = self._on_data
        ws.on_error   = self._handle_error
        ws.on_close   = self._handle_close

        self._ws = ws
        logger.info("LiveFeed: connecting to Angel One SmartFeed ...")
        ws.connect()

    def _make_on_open(self, ws) -> Callable:
        def on_open(wsapp):
            logger.info("LiveFeed: WebSocket connected")
            self._send_subscribe(ws)
            if self._on_connect:
                self._on_connect()
        return on_open

    def _send_subscribe(self, ws) -> None:
        if not self._subscribed_tokens:
            return
        token_list = [{"exchangeType": self.NSE_CM, "tokens": self._subscribed_tokens}]
        ws.subscribe("live_orb_feed", self.QUOTE_MODE, token_list)
        logger.info("LiveFeed: subscribed {} tokens in QUOTE mode", len(self._subscribed_tokens))

    def _on_data(self, wsapp, message) -> None:
        try:
            token = str(message.get("token", ""))
            ltp   = message.get("last_traded_price", 0) / 100.0
            vol   = message.get("volume_trade_for_the_day", 0)
            open_ = message.get("open_price_of_the_day", 0) / 100.0
            high  = message.get("high_price_of_the_day", 0) / 100.0
            low   = message.get("low_price_of_the_day", 0) / 100.0
            ts    = message.get("exchange_timestamp", 0)

            tick = {
                "token":     token,
                "symbol":    self._token_to_symbol.get(token, token),
                "ltp":       ltp,
                "open":      open_,
                "high":      high,
                "low":       low,
                "volume":    vol,
                "timestamp": ts,
            }
            if ltp > 0:
                self._on_tick(tick)
        except Exception as exc:
            logger.warning("LiveFeed: tick parse error: {}", exc)

    def _handle_error(self, wsapp, error) -> None:
        msg = str(error)
        logger.error("LiveFeed: WebSocket error: {}", msg)
        if self._on_error:
            self._on_error(msg)

    def _handle_close(self, wsapp) -> None:
        logger.warning("LiveFeed: WebSocket closed")
        if not self._stop_event.is_set():
            logger.info("LiveFeed: reconnecting in 5s ...")
            time.sleep(5)
            if not self._stop_event.is_set():
                self._connect()
