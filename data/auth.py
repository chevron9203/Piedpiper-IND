"""
Angel One SmartAPI authentication.

Sessions expire daily at midnight IST. This module re-authenticates
on every call if the token is expired, using pyotp for TOTP generation.
Designed for unattended cron use — no interactive input.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from loguru import logger
import pyotp
from SmartApi import SmartConnect
from config import settings

IST = timezone(timedelta(hours=5, minutes=30))


class SmartAPISession:
    def __init__(self):
        self._smart = None
        self._auth_token: str | None = None
        self._feed_token: str | None = None
        self._expires_at: datetime | None = None

    def _is_expired(self) -> bool:
        if self._auth_token is None or self._expires_at is None:
            return True
        return datetime.now(IST) >= self._expires_at

    def _midnight_ist(self) -> datetime:
        """Next midnight IST — when the current session expires."""
        now = datetime.now(IST)
        return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    def authenticate(self) -> "SmartAPISession":
        """
        Authenticate with Angel One SmartAPI using MPIN + TOTP.
        Generates a fresh TOTP each call — never stores static codes.
        """
        totp_code = pyotp.TOTP(settings.ANGELONE_TOTP_SECRET).now()
        logger.info("Authenticating with SmartAPI (client: {})", settings.ANGELONE_CLIENT_CODE)

        smart = SmartConnect(api_key=settings.ANGELONE_API_KEY)
        try:
            data = smart.generateSession(
                clientCode=settings.ANGELONE_CLIENT_CODE,
                password=settings.ANGELONE_MPIN,
                totp=totp_code,
            )
        except Exception as exc:
            logger.error("SmartAPI auth failed: {}", exc)
            raise

        if not data or data.get("status") is False:
            msg = data.get("message", "unknown error") if data else "no response"
            raise RuntimeError(f"SmartAPI auth rejected: {msg}")

        self._smart = smart
        self._auth_token = data["data"]["jwtToken"]
        self._feed_token = smart.getfeedToken()
        self._expires_at = self._midnight_ist()

        logger.info("Authenticated. Session valid until {}", self._expires_at.strftime("%Y-%m-%d %H:%M IST"))
        return self

    def get_client(self) -> SmartConnect:
        """Return a live SmartConnect client, re-authenticating if expired."""
        if self._is_expired():
            logger.info("Session expired or missing — re-authenticating")
            self.authenticate()
        return self._smart

    @property
    def feed_token(self) -> str:
        if self._is_expired():
            self.authenticate()
        return self._feed_token

    @property
    def auth_token(self) -> str:
        if self._is_expired():
            self.authenticate()
        return self._auth_token


# Module-level singleton — share across the process, never across processes
_session = SmartAPISession()


def get_session() -> SmartAPISession:
    return _session


def get_client() -> SmartConnect:
    return _session.get_client()


if __name__ == "__main__":
    import json
    logger.info("Testing SmartAPI authentication...")
    session = SmartAPISession().authenticate()
    client = session.get_client()
    profile = client.getProfile(session.feed_token)
    logger.info("Auth OK. Profile: {}", json.dumps(profile, indent=2))
