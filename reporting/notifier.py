"""
Push alert sender — Telegram and SMTP email.

Design principles
-----------------
- Never raises: a delivery failure logs a warning and returns False.
  The daily run must not crash because Telegram is slow.
- No credentials → silently skips that channel (logs once at INFO level).
- Both channels are attempted independently; failure on one does not
  suppress the other.
- Alert format is terse enough to read on a phone notification.
"""
from __future__ import annotations

import smtplib
import textwrap
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import requests
from loguru import logger

from config.settings import (
    ALERT_EMAIL_TO,
    CONFIDENCE_THRESHOLD,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_TELEGRAM_MSG = 4096  # Telegram hard limit
_REQUEST_TIMEOUT = 10     # seconds


class Notifier:
    """Push alert sender — Telegram and/or email."""

    # ------------------------------------------------------------------
    # Channel: Telegram
    # ------------------------------------------------------------------

    def send_telegram(self, message: str) -> bool:
        """
        Send a message via the Telegram Bot API.

        Uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from settings.
        Returns False (does not raise) if credentials are not configured
        or if the send fails.
        """
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.info(
                "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing) — skipping"
            )
            return False

        # Telegram caps individual messages at 4096 chars
        if len(message) > _MAX_TELEGRAM_MSG:
            message = message[: _MAX_TELEGRAM_MSG - 3] + "..."

        url = _TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN)
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                logger.warning("Telegram API returned not-ok: {}", data)
                return False
            logger.info("Telegram alert sent (chat_id={})", TELEGRAM_CHAT_ID)
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Channel: Email (SMTP)
    # ------------------------------------------------------------------

    def send_email(self, subject: str, body: str) -> bool:
        """
        Send via SMTP.

        Uses SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, and
        ALERT_EMAIL_TO from settings (sourced from .env).
        Returns False if credentials are not configured or send fails.
        """
        if not all([SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO]):
            logger.info(
                "Email not configured (SMTP_USER / SMTP_PASSWORD / ALERT_EMAIL_TO missing) — skipping"
            )
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL_TO

        # Plain-text part
        msg.attach(MIMEText(body, "plain", "utf-8"))

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=_REQUEST_TIMEOUT) as server:
                server.ehlo()
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, ALERT_EMAIL_TO, msg.as_string())
            logger.info("Email alert sent to {}", ALERT_EMAIL_TO)
            return True
        except Exception as exc:
            logger.warning("Email send failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Composed alert
    # ------------------------------------------------------------------

    def send_daily_alert(
        self,
        signals: pd.DataFrame,
        date: date,
        universe_size: int = 0,
    ) -> None:
        """
        Format and dispatch the daily "check this now" push alert.

        Content varies by signal count:
        - Zero signals: brief no-action notification.
        - One or more signals: each stock listed with key metrics.

        Sends to all configured channels (Telegram + email).
        Logs a warning if neither channel is configured.
        """
        message = self._format_alert(signals, date, universe_size)
        subject = self._format_subject(signals, date)

        telegram_ok = self.send_telegram(message)
        email_ok = self.send_email(subject, message)

        if not telegram_ok and not email_ok:
            logger.warning(
                "No notification channels configured or all sends failed — "
                "alert not delivered. Check TELEGRAM_BOT_TOKEN / SMTP_* in .env"
            )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_subject(self, signals: pd.DataFrame, date: date) -> str:
        n = len(signals)
        if n == 0:
            return f"Piedpiper {date} — No signals today"
        signal_word = "signal" if n == 1 else "signals"
        return f"Piedpiper {date} — {n} {signal_word}"

    def _format_alert(
        self,
        signals: pd.DataFrame,
        date: date,
        universe_size: int,
    ) -> str:
        n = len(signals)
        header = f"*Piedpiper — {date}*"

        if n == 0:
            universe_note = (
                f" ({universe_size} scored)" if universe_size else ""
            )
            return (
                f"{header}\n\n"
                f"No qualifying signals today (0{universe_note} cleared the {CONFIDENCE_THRESHOLD:.0%} bar).\n"
                f"Stay flat — this is the system working as designed."
            )

        signal_word = "signal" if n == 1 else "signals"
        lines = [
            f"{header}",
            f"{n} qualifying {signal_word} — act before 9:15 AM IST\n",
        ]

        for _, row in signals.iterrows():
            symbol = row.get("symbol", "?")
            conf = row.get("confidence", float("nan"))
            entry_low = row.get("entry_low", float("nan"))
            entry_high = row.get("entry_high", float("nan"))
            sl = row.get("stop_loss", float("nan"))
            tp = row.get("target", float("nan"))
            size = row.get("position_size", float("nan"))
            gap = row.get("gap_status", "pending")

            # Build gap flag suffix
            gap_suffix = ""
            if gap == "gapped_up_beyond_entry":
                gap_suffix = " ⚠️ GAPPED UP — verify entry"
            elif gap == "gapped_down":
                gap_suffix = " ⚠️ GAPPED DOWN — verify entry"

            block = textwrap.dedent(f"""\
                *{symbol}*{gap_suffix}
                  Conf: {conf:.1%}
                  Entry zone: ₹{entry_low:,.2f} – ₹{entry_high:,.2f}
                  Stop: ₹{sl:,.2f}  |  Target: ₹{tp:,.2f}
                  Size: ₹{size:,.0f}
            """)
            lines.append(block)

        lines.append(
            f"_Threshold: {CONFIDENCE_THRESHOLD:.0%}. "
            f"Zero signals = valid output._"
        )

        return "\n".join(lines)
