"""
Flask dashboard for the NSE trading signal system (Piedpiper).

Single-user local tool. Run with:
    python reporting/dashboard/app.py
    # or
    flask --app reporting/dashboard/app run --port 5001

Routes:
    /             — Overview (portfolio status, regime, VIX, system health)
    /momentum     — Current holdings or CASH, signal history
    /guard        — Daily guard log
    /intraday     — ORB paper trades
    /research     — Static backtest results
    /settings     — Universe + config
    /api/status   — JSON status endpoint
"""
import json
import math
import sys
from pathlib import Path
from datetime import date, datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from loguru import logger

from storage import store
from config import settings

app = Flask(__name__, template_folder="templates", static_folder="static")

store.init_schema()  # ensure tables exist regardless of how the app is started


# ---------------------------------------------------------------------------
# SVG sparkline builder — pure server-side, no JS library
# ---------------------------------------------------------------------------

def _build_sparkline(
    series: pd.Series,
    width: int = 600,
    height: int = 120,
    stroke: str = "#2a7ae2",
    padding: int = 8,
) -> str:
    """Build an inline SVG path string from a numeric pd.Series."""
    values = series.dropna().tolist()
    if len(values) < 2:
        return ""

    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0

    n = len(values)
    x_step = (width - 2 * padding) / (n - 1)

    def _y(v: float) -> float:
        return padding + (height - 2 * padding) * (1 - (v - lo) / span)

    points = [
        f"{padding + i * x_step:.1f},{_y(v):.1f}"
        for i, v in enumerate(values)
    ]
    path_d = "M " + " L ".join(points)

    color = "#27ae60" if values[-1] >= values[0] else "#e74c3c"

    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%;display:block;">'
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_inr(v) -> str:
    """Format as ₹ with comma separation; handle None/NaN/Inf gracefully."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return "—"
        return f"₹{f:,.0f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v) -> str:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return "—"
        return f"{f:.1%}"
    except (TypeError, ValueError):
        return "—"


def _fmt_float(v, decimals: int = 2) -> str:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return "—"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, replacing NaN/NaT with Python None."""
    if df.empty:
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _today() -> date:
    return date.today()


# ---------------------------------------------------------------------------
# Core data helpers
# ---------------------------------------------------------------------------

def _get_regime_data() -> dict:
    """
    Load last 220 trading days of Nifty closes, compute 200d EMA, load latest VIX.
    Returns dict: nifty_close, ema200, is_bull, vix, regime_label, nifty_vs_ema_pct
    """
    defaults = {
        "nifty_close": None,
        "ema200": None,
        "is_bull": None,
        "vix": None,
        "regime_label": "Unknown",
        "nifty_vs_ema_pct": None,
    }
    try:
        with store.db_conn() as conn:
            nifty_df = conn.execute("""
                SELECT dt, close FROM adjusted_ohlcv
                WHERE symbol = 'Nifty 50' AND source = 'eod2_index'
                ORDER BY dt DESC
                LIMIT 220
            """).df()

            vix_row = conn.execute("""
                SELECT close FROM adjusted_ohlcv
                WHERE symbol = 'India VIX' AND source = 'eod2_index'
                ORDER BY dt DESC
                LIMIT 1
            """).fetchone()

        if nifty_df.empty:
            return defaults

        nifty_df = nifty_df.sort_values("dt")
        closes = nifty_df["close"]

        # Exponential moving average (span=200)
        ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
        nifty_close = float(closes.iloc[-1])
        is_bull = nifty_close > ema200
        nifty_vs_ema_pct = (nifty_close - ema200) / ema200 * 100.0

        vix = float(vix_row[0]) if vix_row and vix_row[0] is not None else None

        return {
            "nifty_close": nifty_close,
            "ema200": ema200,
            "is_bull": is_bull,
            "vix": vix,
            "regime_label": "BULL" if is_bull else "BEAR",
            "nifty_vs_ema_pct": nifty_vs_ema_pct,
        }
    except Exception as exc:
        logger.error("_get_regime_data failed: {}", exc)
        return defaults


def _get_latest_signal() -> Optional[dict]:
    """
    Load the most recent row from monthly_signals. Parse JSON target/weights.
    Returns dict or None.
    """
    try:
        with store.db_conn() as conn:
            row = conn.execute("""
                SELECT signal_date, in_market, defensive_mode,
                       nifty_6m_ret, gold_6m_ret, target, weights,
                       prior_holdings, capital, logged_at
                FROM monthly_signals
                ORDER BY signal_date DESC
                LIMIT 1
            """).fetchone()
            if not row:
                return None

        cols = ["signal_date", "in_market", "defensive_mode", "nifty_6m_ret",
                "gold_6m_ret", "target", "weights", "prior_holdings", "capital", "logged_at"]
        result = dict(zip(cols, row))

        # Parse JSON fields
        for field in ("target", "weights", "prior_holdings"):
            val = result.get(field)
            if val and isinstance(val, str):
                try:
                    result[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[field] = []
            elif val is None:
                result[field] = [] if field != "weights" else {}

        return result
    except Exception as exc:
        logger.error("_get_latest_signal failed: {}", exc)
        return None


def _get_all_signals() -> list[dict]:
    """Load all monthly_signals rows ordered by date desc. Parses JSON fields."""
    try:
        with store.db_conn() as conn:
            df = conn.execute("""
                SELECT signal_date, in_market, defensive_mode,
                       nifty_6m_ret, gold_6m_ret, target, weights,
                       prior_holdings, capital, logged_at
                FROM monthly_signals
                ORDER BY signal_date DESC
                LIMIT 24
            """).df()

        if df.empty:
            return []

        records = _to_records(df)
        for rec in records:
            for field in ("target", "weights", "prior_holdings"):
                val = rec.get(field)
                if val and isinstance(val, str):
                    try:
                        rec[field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        rec[field] = [] if field != "weights" else {}
                elif val is None:
                    rec[field] = [] if field != "weights" else {}
        return records
    except Exception as exc:
        logger.error("_get_all_signals failed: {}", exc)
        return []


def _get_current_holdings(signal_date) -> list[dict]:
    """
    Load momentum_entries for a given signal_date, join with latest adjusted_ohlcv
    price per symbol (source='eod2'), compute unrealized P&L.
    """
    try:
        with store.db_conn() as conn:
            entries_df = conn.execute("""
                SELECT signal_date, symbol, entry_price, weight, qty, hard_stop, capital, logged_at
                FROM momentum_entries
                WHERE signal_date = ?
            """, [signal_date]).df()

        if entries_df.empty:
            return []

        symbols = entries_df["symbol"].tolist()

        # Fetch latest price per symbol
        try:
            with store.db_conn() as conn:
                placeholders = ",".join(["?" for _ in symbols])
                prices_df = conn.execute(f"""
                    SELECT symbol, close, dt
                    FROM (
                        SELECT symbol, close, dt,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY dt DESC) AS rn
                        FROM adjusted_ohlcv
                        WHERE symbol IN ({placeholders}) AND source = 'eod2'
                    ) t WHERE rn = 1
                """, symbols).df()
        except Exception:
            prices_df = pd.DataFrame(columns=["symbol", "close", "dt"])

        price_map = {}
        price_date_map = {}
        if not prices_df.empty:
            for _, prow in prices_df.iterrows():
                price_map[prow["symbol"]] = prow["close"]
                price_date_map[prow["symbol"]] = prow["dt"]

        holdings = []
        for _, row in entries_df.iterrows():
            sym = row["symbol"]
            entry_price = row["entry_price"]
            current_price = price_map.get(sym)
            price_date = price_date_map.get(sym)

            unrealized_pnl_pct = None
            unrealized_pnl_abs = None
            if current_price is not None and entry_price and entry_price > 0:
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100.0
                qty = row.get("qty") or 0
                unrealized_pnl_abs = (current_price - entry_price) * qty

            entry_date = row.get("signal_date")
            days_held = None
            if entry_date:
                try:
                    entry_dt = pd.to_datetime(entry_date).date()
                    days_held = (_today() - entry_dt).days
                except Exception:
                    pass

            holdings.append({
                "symbol": sym,
                "entry_date": str(entry_date) if entry_date else None,
                "entry_price": entry_price,
                "current_price": current_price,
                "price_date": str(price_date) if price_date else None,
                "weight": row.get("weight"),
                "qty": row.get("qty"),
                "hard_stop": row.get("hard_stop"),
                "capital": row.get("capital"),
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "unrealized_pnl_abs": unrealized_pnl_abs,
                "days_held": days_held,
            })

        return holdings
    except Exception as exc:
        logger.error("_get_current_holdings failed: {}", exc)
        return []


def _get_db_mtime() -> str:
    """Return the last modified time of the DuckDB file."""
    try:
        db_path = Path(settings.DB_PATH)
        if db_path.exists():
            mtime = datetime.fromtimestamp(db_path.stat().st_mtime)
            return mtime.strftime("%d %b %Y, %H:%M")
        return "—"
    except Exception:
        return "—"


def _vix_label(vix) -> str:
    """Return human label for VIX level."""
    if vix is None:
        return "Unknown"
    if vix < 15:
        return "Low"
    if vix < 20:
        return "Normal"
    if vix < 25:
        return "Elevated"
    if vix < 35:
        return "High"
    return "Danger"


def _signal_action_label(rec: dict) -> str:
    """Derive the action label for a monthly signal record."""
    in_market = rec.get("in_market")
    defensive = rec.get("defensive_mode")
    if in_market:
        return "Entered market"
    if defensive:
        return "Regime exit → LIQUIDBEES"
    return "Stayed in CASH"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def overview():
    """Overview: portfolio status, regime, VIX, system health."""
    import calendar
    regime = _get_regime_data()
    latest = _get_latest_signal()
    all_signals = _get_all_signals()
    db_mtime = _get_db_mtime()

    in_market = latest.get("in_market") if latest else False
    market_since = None
    market_count = 0

    if latest:
        market_since = str(latest.get("signal_date", ""))
        target = latest.get("target") or []
        market_count = len(target) if isinstance(target, list) else 0

    # Determine next rebalance (last trading day of month)
    today = _today()
    last_day = calendar.monthrange(today.year, today.month)[1]
    month_end = date(today.year, today.month, last_day)
    while month_end.weekday() >= 5:
        month_end = month_end.replace(day=month_end.day - 1)

    if today >= month_end:
        if today.month == 12:
            next_last = calendar.monthrange(today.year + 1, 1)[1]
            next_month_end = date(today.year + 1, 1, next_last)
        else:
            next_last = calendar.monthrange(today.year, today.month + 1)[1]
            next_month_end = date(today.year, today.month + 1, next_last)
        while next_month_end.weekday() >= 5:
            next_month_end = next_month_end.replace(day=next_month_end.day - 1)
        next_rebalance = next_month_end
    else:
        next_rebalance = month_end

    if next_rebalance == today:
        next_rebalance_label = "Today at 6:00 PM"
    elif (next_rebalance - today).days == 1:
        next_rebalance_label = "Tomorrow at 6:00 PM"
    else:
        next_rebalance_label = next_rebalance.strftime("%d %b %Y") + " at 6:00 PM"

    # Enrich all_signals with action labels and portfolio display
    for rec in all_signals:
        rec["action_label"] = _signal_action_label(rec)
        tgt = rec.get("target") or []
        if isinstance(tgt, list) and tgt:
            rec["portfolio_display"] = ", ".join(tgt[:4]) + ("…" if len(tgt) > 4 else "")
        else:
            rec["portfolio_display"] = "LIQUIDBEES"
        rec["regime_display"] = "BULL" if (rec.get("in_market") and not rec.get("defensive_mode")) else "BEAR"

    vix = regime.get("vix")
    vix_label = _vix_label(vix)
    vix_badge = "badge-green" if vix and vix < 20 else ("badge-orange" if vix and vix < 35 else "badge-red")

    return render_template(
        "overview.html",
        active_tab="overview",
        in_market=in_market,
        market_since=market_since,
        market_count=market_count,
        regime=regime,
        vix_label=vix_label,
        vix_badge=vix_badge,
        db_mtime=db_mtime,
        next_rebalance_label=next_rebalance_label,
        all_signals=all_signals,
    )


@app.route("/momentum")
def momentum():
    """Momentum portfolio: current holdings or CASH, signal history."""
    latest = _get_latest_signal()
    all_signals = _get_all_signals()
    regime = _get_regime_data()

    in_market = latest.get("in_market") if latest else False
    signal_date = str(latest.get("signal_date", "")) if latest else ""
    holdings = []
    total_invested = 0.0
    total_unrealized_pnl = 0.0
    total_unrealized_pnl_pct = None

    if in_market and latest:
        holdings = _get_current_holdings(latest.get("signal_date"))
        for h in holdings:
            qty = h.get("qty") or 0
            ep = h.get("entry_price") or 0
            total_invested += qty * ep
            if h.get("unrealized_pnl_abs") is not None:
                total_unrealized_pnl += h["unrealized_pnl_abs"]

        if total_invested > 0:
            total_unrealized_pnl_pct = total_unrealized_pnl / total_invested * 100.0

    # Prior holdings from last signal
    prior_holdings_display = []
    if not in_market and latest:
        prior = latest.get("prior_holdings") or []
        if isinstance(prior, list):
            prior_holdings_display = prior

    # Enrich signal history
    for rec in all_signals:
        rec["action_label"] = _signal_action_label(rec)
        tgt = rec.get("target") or []
        rec["portfolio_display"] = ", ".join(tgt) if isinstance(tgt, list) and tgt else "LIQUIDBEES"
        wts = rec.get("weights") or {}
        rec["weights_display"] = wts

    return render_template(
        "momentum.html",
        active_tab="momentum",
        in_market=in_market,
        signal_date=signal_date,
        latest=latest,
        holdings=holdings,
        total_invested=_fmt_inr(total_invested),
        total_unrealized_pnl=_fmt_inr(total_unrealized_pnl),
        total_unrealized_pnl_pct=f"{total_unrealized_pnl_pct:+.2f}%" if total_unrealized_pnl_pct is not None else "—",
        pnl_positive=(total_unrealized_pnl >= 0),
        prior_holdings_display=prior_holdings_display,
        all_signals=all_signals,
        regime=regime,
        fmt_inr=_fmt_inr,
        fmt_pct=_fmt_pct,
    )


@app.route("/guard")
def guard():
    """Daily guard log: regime, VIX, guard decisions."""
    regime = _get_regime_data()

    decisions = []
    last_guard_run = None
    try:
        with store.db_conn() as conn:
            df = conn.execute("""
                SELECT decision_date, symbol, action, reason, detail,
                       trigger_price, trail_stop, entry_price,
                       unrealized_pnl_pct, paper, logged_at
                FROM momentum_daily_decisions
                ORDER BY logged_at DESC
                LIMIT 200
            """).df()

        if not df.empty:
            decisions = _to_records(df)
            last_guard_run = str(decisions[0].get("logged_at", "")) if decisions else None
    except Exception as exc:
        logger.error("guard: failed to load decisions: {}", exc)

    vix = regime.get("vix")
    vix_label = _vix_label(vix)
    vix_badge = "badge-green" if vix and vix < 20 else ("badge-orange" if vix and vix < 35 else "badge-red")

    return render_template(
        "guard.html",
        active_tab="guard",
        regime=regime,
        vix_label=vix_label,
        vix_badge=vix_badge,
        decisions=decisions,
        last_guard_run=last_guard_run,
        today=_today().strftime("%d %b %Y"),
    )


@app.route("/intraday")
def intraday_view():
    """Intraday ORB paper trading — today's trades and recent history."""
    today_dt = _today()
    capital = settings.INTRADAY_CAPITAL

    today_df = store.load_intraday_trades(from_date=today_dt, to_date=today_dt)

    closed_today = today_df.dropna(subset=["net_pnl"]) if not today_df.empty else pd.DataFrame()
    net_pnl = float(closed_today["net_pnl"].sum()) if not closed_today.empty else 0.0
    total_charges = float(closed_today["charges"].sum()) if not closed_today.empty else 0.0
    wins = int((closed_today["net_pnl"] > 0).sum()) if not closed_today.empty else 0
    losses = int((closed_today["net_pnl"] <= 0).sum()) if not closed_today.empty else 0

    today_rows = _to_records(today_df)

    all_df = store.load_intraday_trades()
    all_closed = all_df.dropna(subset=["net_pnl"]) if not all_df.empty else pd.DataFrame()
    all_time_pnl = float(all_closed["net_pnl"].sum()) if not all_closed.empty else 0.0
    total_trades = len(all_closed)

    hist = (
        all_closed[all_closed["trade_date"].astype(str) != str(today_dt)]
        if not all_closed.empty
        else pd.DataFrame()
    )
    history_rows = (
        _to_records(hist.sort_values("trade_date", ascending=False).head(30))
        if not hist.empty
        else []
    )

    return render_template(
        "intraday.html",
        active_tab="intraday",
        run_date=today_dt.strftime("%A, %d %b %Y"),
        capital=capital,
        today_rows=today_rows,
        trades_today=len(today_rows),
        net_pnl=net_pnl,
        total_charges=total_charges,
        wins=wins,
        losses=losses,
        all_time_pnl=all_time_pnl,
        total_trades=total_trades,
        history_rows=history_rows,
    )


@app.route("/intraday/close/<symbol>", methods=["POST"])
def intraday_close(symbol):
    """Trigger early manual close of an open intraday position."""
    import subprocess
    venv_py = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python3"
    monitor = Path(__file__).parent.parent.parent / "scripts" / "intraday_monitor.py"
    py = str(venv_py) if venv_py.exists() else sys.executable
    subprocess.Popen([py, str(monitor), "--close", symbol])
    logger.info("Manual close triggered for {}", symbol)
    return redirect(url_for("intraday_view"))


@app.route("/intraday/close-all", methods=["POST"])
def intraday_close_all():
    """Trigger early manual close of ALL open intraday positions."""
    import subprocess
    venv_py = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python3"
    monitor = Path(__file__).parent.parent.parent / "scripts" / "intraday_monitor.py"
    py = str(venv_py) if venv_py.exists() else sys.executable
    subprocess.Popen([py, str(monitor), "--close", "ALL"])
    logger.info("Manual close ALL triggered")
    return redirect(url_for("intraday_view"))


@app.route("/research")
def research():
    """Static research page showing backtest results."""
    return render_template("research.html", active_tab="research")


@app.route("/settings")
def settings_view():
    """Universe & Settings."""
    universe_path = Path(settings.DATA_DIR) / "universe_cache.parquet"
    universe: list[dict] = []

    if universe_path.exists():
        try:
            df = pd.read_parquet(universe_path)
            if "avg_turnover_cr" not in df.columns:
                df["avg_turnover_cr"] = None
            if "in_index" not in df.columns:
                df["in_index"] = "Nifty 200"
            universe = df[["symbol", "sector", "avg_turnover_cr", "in_index"]].to_dict("records")
        except Exception as exc:
            logger.error("settings_view: failed to load universe cache: {}", exc)

    config_items = {
        "Universe": "500 NSE stocks (Nifty 500)",
        "Score": "Composite momentum (12m×40% + 6m×30% + 3m×20% + 1m×10%)",
        "Sizing": "Inverse volatility weights",
        "Portfolio Size": "Top 10 stocks",
        "Filter": "Stock above its own 200d EMA | Market above Nifty 200d EMA",
        "Rebalance": "Last trading day of each month",
        "Sector Cap": "Max 3 stocks per sector",
        "Regime Exit": "Nifty 50 < 200d EMA → move to LIQUIDBEES",
        "Hard Stop": "-15% from entry price (daily guard)",
        "VIX Emergency Exit": "India VIX > 35",
        "DB Path": str(settings.DB_PATH),
        "Intraday Capital": _fmt_inr(settings.INTRADAY_CAPITAL),
        "Paper Mode": "Yes — no real orders placed",
    }

    return render_template(
        "settings.html",
        active_tab="settings",
        universe=universe,
        n_universe=len(universe),
        config_items=config_items,
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """Return system status as JSON."""
    try:
        regime = _get_regime_data()
        latest = _get_latest_signal()

        in_market = latest.get("in_market") if latest else False
        signal_date = str(latest.get("signal_date", "")) if latest else None
        target = (latest.get("target") or []) if latest else []

        return jsonify({
            "as_of": _today().isoformat(),
            "regime": regime.get("regime_label"),
            "nifty_close": regime.get("nifty_close"),
            "ema200": regime.get("ema200"),
            "nifty_vs_ema_pct": regime.get("nifty_vs_ema_pct"),
            "is_bull": regime.get("is_bull"),
            "vix": regime.get("vix"),
            "in_market": in_market,
            "signal_date": signal_date,
            "portfolio": target if isinstance(target, list) else [],
            "portfolio_count": len(target) if isinstance(target, list) else 0,
        })
    except Exception as exc:
        logger.error("api_status failed: {}", exc)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
