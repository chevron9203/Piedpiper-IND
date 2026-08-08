"""
Flask dashboard for the NSE trading signal system.

Single-user local tool. Run with:
    python reporting/dashboard/app.py
    # or
    flask --app reporting/dashboard/app run --port 5001

Routes:
    /                 — Today's signals
    /ledger-a         — Virtual portfolio (Ledger A)
    /ledger-b         — Real trades (Ledger B)
    /ledger-b/log     — POST: log a real trade
    /ledger-b/decline — POST: mark signal as declined
    /comparison       — Ledger A vs Ledger B side-by-side
    /backtest         — Walk-forward results
    /universe         — Universe & settings
    /api/signals/<date_str> — JSON signals endpoint
"""
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
from portfolio.ledger_b import LedgerB
from config import settings

app = Flask(__name__, template_folder="templates", static_folder="static")

store.init_schema()   # ensure tables exist regardless of how the app is started

_ledger_b = LedgerB()


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
    """
    Build an inline SVG path string from a numeric pd.Series.
    Returns an empty string if there are fewer than 2 points.
    """
    values = series.dropna().tolist()
    if len(values) < 2:
        return ""

    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0

    n = len(values)
    x_step = (width - 2 * padding) / (n - 1)

    def _y(v: float) -> float:
        # flip: high value → top of SVG
        return padding + (height - 2 * padding) * (1 - (v - lo) / span)

    points = [
        f"{padding + i * x_step:.1f},{_y(v):.1f}"
        for i, v in enumerate(values)
    ]
    path_d = "M " + " L ".join(points)

    # Determine line colour by last-vs-first trend
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


def _load_ledger_a_data() -> dict:
    """
    Read ledger_a table from DuckDB and compute summary metrics.
    Returns a context dict ready for the template.
    """
    with store.db_conn() as conn:
        open_pos = conn.execute(
            "SELECT * FROM ledger_a WHERE status = 'open' ORDER BY entry_date DESC"
        ).df()
        closed_trades = conn.execute(
            "SELECT * FROM ledger_a WHERE status != 'open' ORDER BY exit_date DESC"
        ).df()

    # Equity history: reconstruct from start capital + cumulative net_pnl of closed trades
    start_cap = settings.STARTING_VIRTUAL_CAPITAL
    eq_series = pd.Series(dtype=float)
    sparkline_svg = ""

    if not closed_trades.empty and "exit_date" in closed_trades.columns:
        ct = closed_trades.copy()
        ct["exit_date"] = pd.to_datetime(ct["exit_date"], errors="coerce")
        ct = ct.dropna(subset=["exit_date", "net_pnl"])
        if not ct.empty:
            daily_pnl = ct.groupby("exit_date")["net_pnl"].sum().sort_index()
            eq_series = (start_cap + daily_pnl.cumsum())
            sparkline_svg = _build_sparkline(eq_series)

    # Metrics
    net_pnl = float(closed_trades["net_pnl"].sum()) if not closed_trades.empty and "net_pnl" in closed_trades.columns else 0.0
    current_equity = start_cap + net_pnl

    hit_rate_val = 0.0
    sharpe_val = 0.0
    max_dd_val = 0.0

    if not closed_trades.empty and "net_pnl" in closed_trades.columns:
        ct_nn = closed_trades.dropna(subset=["net_pnl"])
        if not ct_nn.empty:
            hit_rate_val = float((ct_nn["net_pnl"] > 0).mean())

    if len(eq_series) >= 2:
        from backtest.metrics import sharpe_ratio, max_drawdown
        daily_ret = eq_series.pct_change().dropna()
        sharpe_val = sharpe_ratio(daily_ret)
        max_dd_val = max_drawdown(eq_series)

    # Unrealised P&L for open positions (no live price — use entry as proxy)
    if not open_pos.empty:
        open_pos["unrealised_pnl"] = 0.0  # placeholder without live prices
        open_pos["days_held"] = (
            pd.Timestamp(_today()) - pd.to_datetime(open_pos["entry_date"], errors="coerce")
        ).dt.days

    return {
        "current_equity":  _fmt_inr(current_equity),
        "sharpe":          _fmt_float(sharpe_val),
        "max_drawdown":    _fmt_pct(max_dd_val),
        "hit_rate":        _fmt_pct(hit_rate_val),
        "n_open":          len(open_pos),
        "net_pnl":         _fmt_inr(net_pnl),
        "sparkline_svg":   sparkline_svg,
        "open_positions":  open_pos,
        "closed_trades":   closed_trades,
    }


def _load_ledger_b_data() -> dict:
    """Load Ledger B open/closed trades."""
    open_trades = _ledger_b.get_open_trades()
    closed_trades = _ledger_b.get_closed_trades()
    return {
        "open_trades":   open_trades,
        "closed_trades": closed_trades,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def today():
    """Tab 1: Today's Signals."""
    run_date = _today()
    signals_df = store.load_signals(run_date)

    # Enrich with gap check status if gap_check column present
    gap_status_col = "gap_status"
    if gap_status_col not in signals_df.columns and not signals_df.empty:
        signals_df[gap_status_col] = "pending"

    # Universe scored count (approximate from signals table for the date)
    with store.db_conn() as conn:
        scored_count_row = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM features WHERE dt = ?", [run_date]
        ).fetchone()
    universe_scored = scored_count_row[0] if scored_count_row else 0

    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return render_template(
        "today.html",
        active_tab="today",
        run_date=run_date.strftime("%A, %d %B %Y"),
        signals=_to_records(signals_df),
        n_signals=len(signals_df),
        universe_scored=universe_scored,
        last_updated=last_updated,
    )


@app.route("/ledger-a")
def ledger_a():
    """Tab 2: Ledger A — virtual portfolio."""
    page = int(request.args.get("page", 1))
    per_page = 25
    sort_by = request.args.get("sort", "exit_date")
    sort_dir = request.args.get("dir", "desc")

    ctx = _load_ledger_a_data()

    # Paginate closed trades
    closed = ctx["closed_trades"]
    if not closed.empty and sort_by in closed.columns:
        closed = closed.sort_values(sort_by, ascending=(sort_dir == "asc"))
    total_closed = len(closed)
    total_pages = max(1, (total_closed + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    closed_page = closed.iloc[start : start + per_page]

    ctx.update({
        "active_tab":    "ledger_a",
        "closed_trades": closed_page,
        "total_closed":  total_closed,
        "page":          page,
        "total_pages":   total_pages,
        "sort_by":       sort_by,
        "sort_dir":      sort_dir,
    })
    return render_template("ledger_a.html", **ctx)


@app.route("/ledger-b")
def ledger_b_view():
    """Tab 3: Ledger B — real trades."""
    run_date = _today()
    today_signals = store.load_signals(run_date)

    # Filter out signals already logged or declined today
    with store.db_conn() as conn:
        already_logged = conn.execute(
            "SELECT symbol FROM ledger_b WHERE signal_date = ?", [run_date]
        ).df()

    logged_symbols = set(already_logged["symbol"].tolist()) if not already_logged.empty else set()
    pending_signals = (
        today_signals[~today_signals["symbol"].isin(logged_symbols)]
        if not today_signals.empty
        else pd.DataFrame()
    )

    ctx = _load_ledger_b_data()
    ctx.update({
        "active_tab":      "ledger_b",
        "run_date":        run_date.isoformat(),
        "pending_signals": pending_signals.to_dict("records") if not pending_signals.empty else [],
    })
    return render_template("ledger_b.html", **ctx)


@app.route("/ledger-b/log", methods=["POST"])
def log_trade():
    """Handle trade logging from Ledger B."""
    form = request.form
    try:
        symbol = form["symbol"].strip().upper()
        fill_price = float(form["fill_price"])
        quantity = int(form["quantity"])
        gtt_stop = float(form["gtt_stop"])
        gtt_target = float(form["gtt_target"])
        notes = form.get("notes", "").strip()
        signal_date_str = form.get("signal_date", "")
        signal_date = date.fromisoformat(signal_date_str) if signal_date_str else _today()
        entry_date = _today()

        _ledger_b.log_trade(
            symbol=symbol,
            signal_date=signal_date,
            entry_date=entry_date,
            entry_price=fill_price,
            quantity=quantity,
            gtt_stop=gtt_stop,
            gtt_target=gtt_target,
            notes=notes,
        )
        logger.info("Dashboard: logged trade for {}", symbol)
    except Exception as exc:
        logger.error("Dashboard: trade log failed: {}", exc)

    return redirect(url_for("ledger_b_view"))


@app.route("/ledger-b/decline", methods=["POST"])
def decline_signal():
    """Mark one or more signals as declined."""
    form = request.form
    symbols = form.getlist("symbol")
    reason = form.get("reason", "").strip()
    signal_date_str = form.get("signal_date", "")
    signal_date = date.fromisoformat(signal_date_str) if signal_date_str else _today()

    for symbol in symbols:
        symbol = symbol.strip().upper()
        if symbol:
            _ledger_b.log_declined(symbol=symbol, signal_date=signal_date, reason=reason)
            logger.info("Dashboard: declined signal for {}", symbol)

    return redirect(url_for("ledger_b_view"))


@app.route("/comparison")
def comparison():
    """Tab 4: Ledger A vs Ledger B comparison."""
    # Ledger A data from DB
    with store.db_conn() as conn:
        la_closed = conn.execute(
            "SELECT symbol, entry_date, entry_price, exit_date, exit_price, "
            "gross_pnl, net_pnl, exit_reason FROM ledger_a WHERE status != 'open'"
        ).df()

    la_net_pnl = float(la_closed["net_pnl"].sum()) if not la_closed.empty else 0.0
    la_equity  = settings.STARTING_VIRTUAL_CAPITAL + la_net_pnl
    la_trades  = len(la_closed)

    # Compute Ledger A Sharpe from equity curve
    la_sharpe_str = "—"
    if not la_closed.empty:
        try:
            ct = la_closed.copy()
            ct["exit_date"] = pd.to_datetime(ct["exit_date"], errors="coerce")
            ct = ct.dropna(subset=["exit_date", "net_pnl"])
            if len(ct) >= 5:
                eq_ser = (settings.STARTING_VIRTUAL_CAPITAL
                          + ct.groupby("exit_date")["net_pnl"].sum().sort_index().cumsum())
                if len(eq_ser) >= 2:
                    from backtest.metrics import sharpe_ratio
                    la_sharpe_str = _fmt_float(sharpe_ratio(eq_ser.pct_change().dropna()))
        except Exception:
            pass

    # Ledger B data
    lb_closed   = _ledger_b.get_closed_trades()
    lb_declined = _ledger_b.get_declined_signals()
    lb_net_pnl  = float(lb_closed["net_pnl"].sum()) if not lb_closed.empty else 0.0
    lb_trades   = len(lb_closed)

    # A took but B declined
    if not la_closed.empty and not lb_declined.empty:
        lb_declined["signal_date"] = pd.to_datetime(lb_declined["signal_date"], errors="coerce")
        la_closed["entry_date"] = pd.to_datetime(la_closed["entry_date"], errors="coerce")
        a_not_b = la_closed[la_closed["symbol"].isin(lb_declined["symbol"])]
    else:
        a_not_b = pd.DataFrame()

    # Matched trades (same symbol in both)
    matched = pd.DataFrame()
    if not la_closed.empty and not lb_closed.empty:
        la_subset = la_closed[["symbol", "entry_price", "net_pnl"]].rename(
            columns={"entry_price": "a_entry", "net_pnl": "a_net_pnl"}
        )
        lb_subset = lb_closed[["symbol", "entry_price", "net_pnl"]].rename(
            columns={"entry_price": "b_entry", "net_pnl": "b_net_pnl"}
        )
        matched = la_subset.merge(lb_subset, on="symbol", how="inner")
        if not matched.empty:
            matched["fill_diff"] = matched["b_entry"] - matched["a_entry"]
            matched["pnl_diff"] = matched["b_net_pnl"] - matched["a_net_pnl"]

    net_alpha = lb_net_pnl - la_net_pnl

    return render_template(
        "comparison.html",
        active_tab="comparison",
        la_equity=_fmt_inr(la_equity),
        la_sharpe=la_sharpe_str,
        la_trades=la_trades,
        lb_net_pnl=_fmt_inr(lb_net_pnl),
        lb_trades=lb_trades,
        net_alpha=_fmt_inr(net_alpha),
        net_alpha_sign="positive" if net_alpha >= 0 else "negative",
        a_not_b=_to_records(a_not_b),
        matched_trades=_to_records(matched),
    )


@app.route("/backtest")
def backtest_view():
    """Tab 5: Backtest & Research — walk-forward results."""
    results_path = Path(settings.DATA_DIR) / "backtest_results.parquet"

    results_available = False
    metrics_gross: dict = {}
    metrics_net: dict = {}
    trades_df = pd.DataFrame()
    n_folds = 0
    date_range = ""

    if results_path.exists():
        try:
            bt = pd.read_parquet(results_path)
            results_available = True

            # The parquet stores metrics as rows with a 'metric' and 'value' column
            # plus 'label' (gross/net) and optionally trades.
            if "label" in bt.columns and "metric" in bt.columns:
                gross_rows = bt[bt["label"] == "gross"]
                net_rows = bt[bt["label"] == "net"]
                metrics_gross = dict(zip(gross_rows["metric"], gross_rows["value"]))
                metrics_net = dict(zip(net_rows["metric"], net_rows["value"]))

            # Trades stored in a separate file by convention
            trades_path = Path(settings.DATA_DIR) / "backtest_trades.parquet"
            if trades_path.exists():
                trades_df = pd.read_parquet(trades_path)
                n_folds = int(trades_df["split_id"].nunique()) if "split_id" in trades_df.columns else 0
                if "entry_date" in trades_df.columns:
                    ed = pd.to_datetime(trades_df["entry_date"], errors="coerce").dropna()
                    if not ed.empty:
                        date_range = f"{ed.min().date()} → {ed.max().date()}"
        except Exception as exc:
            logger.error("backtest_view: failed to load results: {}", exc)

    def _fmts(d: dict, k: str, pct: bool = False) -> str:
        v = d.get(k)
        if v is None:
            return "—"
        return _fmt_pct(v) if pct else _fmt_float(v)

    return render_template(
        "backtest.html",
        active_tab="backtest",
        results_available=results_available,
        n_folds=n_folds,
        date_range=date_range,
        # Gross metrics
        gross_sharpe=_fmts(metrics_gross, "sharpe"),
        gross_sortino=_fmts(metrics_gross, "sortino"),
        gross_maxdd=_fmts(metrics_gross, "max_drawdown", pct=True),
        gross_cagr=_fmts(metrics_gross, "cagr", pct=True),
        gross_hitrate=_fmts(metrics_gross, "hit_rate", pct=True),
        gross_wl=_fmts(metrics_gross, "win_loss_ratio"),
        gross_trades=metrics_gross.get("n_trades", "—"),
        gross_pnl=_fmt_inr(metrics_gross.get("total_gross_pnl")),
        # Net metrics
        net_sharpe=_fmts(metrics_net, "sharpe"),
        net_sortino=_fmts(metrics_net, "sortino"),
        net_maxdd=_fmts(metrics_net, "max_drawdown", pct=True),
        net_cagr=_fmts(metrics_net, "cagr", pct=True),
        net_hitrate=_fmts(metrics_net, "hit_rate", pct=True),
        net_wl=_fmts(metrics_net, "win_loss_ratio"),
        net_trades=metrics_net.get("n_trades", "—"),
        net_pnl=_fmt_inr(metrics_net.get("total_net_pnl")),
        gross_pnl_pos=(metrics_gross.get("total_gross_pnl") or 0) >= 0,
        net_pnl_pos=(metrics_net.get("total_net_pnl") or 0) >= 0,
        trades=_to_records(trades_df.head(50)) if not trades_df.empty else [],
    )


@app.route("/universe")
def universe_view():
    """Tab 6: Universe & Settings."""
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
            logger.error("universe_view: failed to load universe cache: {}", exc)

    config_items = {
        "Confidence Threshold":         f"{settings.CONFIDENCE_THRESHOLD:.0%}",
        "Slippage (Nifty 100)":         "0.10%",
        "Slippage (Midcap 100)":        "0.20%",
        "Max Concurrent Positions":     str(settings.MAX_CONCURRENT_POSITIONS),
        "Max Capital Deployed":         f"{settings.MAX_CAPITAL_DEPLOYED_PCT:.0%}",
        "Max Positions per Sector":     str(settings.MAX_POSITIONS_PER_SECTOR),
        "Starting Virtual Capital":     _fmt_inr(settings.STARTING_VIRTUAL_CAPITAL),
        "Take-Profit Barrier":          f"{settings.BARRIER_TAKE_PROFIT_PCT:.0%}",
        "Stop-Loss Barrier":            f"{settings.BARRIER_STOP_LOSS_PCT:.0%}",
        "Max Holding Days":             str(settings.BARRIER_MAX_HOLDING_DAYS),
        "Min Avg Daily Turnover (Cr)":  f"₹{settings.MIN_AVG_DAILY_TURNOVER_CR:.0f} Cr",
        "Feature Version":              settings.FEATURE_VERSION,
        "Fill Assumption":              settings.FILL_ASSUMPTION,
        "Retrain Cadence (trading days)": str(settings.RETRAIN_CADENCE_TRADING_DAYS),
        "Walk-Forward Min Train (years)": str(settings.MIN_TRAIN_WINDOW_YEARS),
        "Embargo Days":                 str(settings.EMBARGO_DAYS),
        "Opening Gap Flag %":           f"{settings.OPENING_GAP_FLAG_PCT:.1%}",
        "Max Drawdown Kill Switch":     f"{settings.MAX_DRAWDOWN_KILL:.0%}",
    }

    return render_template(
        "universe.html",
        active_tab="universe",
        universe=universe,
        n_universe=len(universe),
        config_items=config_items,
    )


# ---------------------------------------------------------------------------
# Intraday ORB
# ---------------------------------------------------------------------------

@app.route("/intraday")
def intraday_view():
    """Intraday ORB paper trading — today's trades and recent history."""
    today_dt = _today()
    capital  = settings.INTRADAY_CAPITAL

    # Today's trades (keep original for calculations, clean copy for template)
    today_df = store.load_intraday_trades(from_date=today_dt, to_date=today_dt)

    closed_today  = today_df.dropna(subset=["net_pnl"]) if not today_df.empty else pd.DataFrame()
    net_pnl       = float(closed_today["net_pnl"].sum())      if not closed_today.empty else 0.0
    total_charges = float(closed_today["charges"].sum())      if not closed_today.empty else 0.0
    wins          = int((closed_today["net_pnl"] > 0).sum())  if not closed_today.empty else 0
    losses        = int((closed_today["net_pnl"] <= 0).sum()) if not closed_today.empty else 0

    today_rows = _to_records(today_df)

    # All-time metrics
    all_df       = store.load_intraday_trades()
    all_closed   = all_df.dropna(subset=["net_pnl"]) if not all_df.empty else pd.DataFrame()
    all_time_pnl = float(all_closed["net_pnl"].sum()) if not all_closed.empty else 0.0
    total_trades = len(all_closed)

    # Recent history (last 30 closed, excluding today)
    hist = all_closed[all_closed["trade_date"].astype(str) != str(today_dt)] if not all_closed.empty else pd.DataFrame()
    history_rows = _to_records(hist.sort_values("trade_date", ascending=False).head(30)) if not hist.empty else []

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
    monitor  = Path(__file__).parent.parent.parent / "scripts" / "intraday_monitor.py"
    py = str(venv_py) if venv_py.exists() else sys.executable
    subprocess.Popen([py, str(monitor), "--close", symbol])
    logger.info("Manual close triggered for {}", symbol)
    return redirect(url_for("intraday_view"))


@app.route("/intraday/close-all", methods=["POST"])
def intraday_close_all():
    """Trigger early manual close of ALL open intraday positions."""
    import subprocess
    venv_py = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python3"
    monitor  = Path(__file__).parent.parent.parent / "scripts" / "intraday_monitor.py"
    py = str(venv_py) if venv_py.exists() else sys.executable
    subprocess.Popen([py, str(monitor), "--close", "ALL"])
    logger.info("Manual close ALL triggered")
    return redirect(url_for("intraday_view"))


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.route("/api/signals/<date_str>")
def api_signals(date_str: str):
    """Return signals for a given date as JSON (YYYY-MM-DD)."""
    try:
        dt = date.fromisoformat(date_str)
    except ValueError:
        return jsonify({"error": f"Invalid date format: {date_str!r}. Use YYYY-MM-DD."}), 400

    df = store.load_signals(dt)
    if df.empty:
        return jsonify({"date": date_str, "signals": [], "count": 0})

    return jsonify({
        "date":    date_str,
        "signals": _to_records(df),
        "count":   len(df),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
