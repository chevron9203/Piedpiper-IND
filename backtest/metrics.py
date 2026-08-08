"""
Backtest performance metrics for the NSE trading signal system.

All ratio calculations use daily returns. Risk-free rate defaults to 6.5%
(approximate Indian 10-year G-Sec yield as of 2026).
"""
import numpy as np
import pandas as pd
from loguru import logger


def sharpe_ratio(returns: pd.Series, annualise: bool = True, rf: float = 0.065) -> float:
    """
    Sharpe ratio on daily returns.

    Parameters
    ----------
    returns  : daily return series (decimal, not percent)
    annualise: if True, multiplies by sqrt(252)
    rf       : annual risk-free rate (default 6.5%)
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = rf / 252
    excess = returns - daily_rf
    sr = excess.mean() / excess.std(ddof=1)
    if annualise:
        sr *= np.sqrt(252)
    return float(sr)


def sortino_ratio(returns: pd.Series, rf: float = 0.065) -> float:
    """
    Sortino ratio — penalises only downside deviation, annualised.

    Parameters
    ----------
    returns : daily return series (decimal)
    rf      : annual risk-free rate
    """
    if returns.empty:
        return 0.0
    daily_rf = rf / 252
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return 0.0
    downside_std = np.sqrt((downside ** 2).mean())  # semi-deviation
    if downside_std == 0:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(252))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a positive fraction (e.g. 0.20 = 20%).

    Parameters
    ----------
    equity_curve : portfolio value over time
    """
    if equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    drawdowns = (equity_curve - roll_max) / roll_max
    return float(abs(drawdowns.min()))


def cagr(equity_curve: pd.Series) -> float:
    """
    Compound Annual Growth Rate from first to last value.
    Uses 252 trading days per year.

    Returns 0.0 if fewer than 2 data points.
    """
    if len(equity_curve) < 2:
        return 0.0
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    if start <= 0:
        return 0.0
    n_days = len(equity_curve) - 1
    years = n_days / 252
    if years <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1)


def hit_rate(trades: pd.DataFrame) -> float:
    """
    Fraction of completed trades where net_pnl > 0.

    Parameters
    ----------
    trades : DataFrame with 'net_pnl' column
    """
    if trades.empty or "net_pnl" not in trades.columns:
        return 0.0
    closed = trades.dropna(subset=["net_pnl"])
    if closed.empty:
        return 0.0
    return float((closed["net_pnl"] > 0).mean())


def avg_win_loss_ratio(trades: pd.DataFrame) -> float:
    """
    Ratio of average winning net_pnl to absolute average losing net_pnl.
    Only considers trades with non-zero P&L. Returns 0.0 if no losers.

    Parameters
    ----------
    trades : DataFrame with 'net_pnl' column
    """
    if trades.empty or "net_pnl" not in trades.columns:
        return 0.0
    closed = trades.dropna(subset=["net_pnl"])
    winners = closed[closed["net_pnl"] > 0]["net_pnl"]
    losers = closed[closed["net_pnl"] < 0]["net_pnl"]
    if losers.empty or winners.empty:
        return 0.0
    return float(winners.mean() / abs(losers.mean()))


def turnover(trades: pd.DataFrame, avg_capital: float) -> float:
    """
    Annualised turnover = total traded value (entry side) / avg_capital / years.

    Parameters
    ----------
    trades      : DataFrame with 'entry_price' and 'quantity' columns
    avg_capital : average capital deployed over the period
    """
    if trades.empty or avg_capital <= 0:
        return 0.0
    required = {"entry_price", "quantity", "entry_date"}
    if not required.issubset(trades.columns):
        logger.warning("turnover: missing columns, need entry_price/quantity/entry_date")
        return 0.0
    total_traded = (trades["entry_price"] * trades["quantity"]).sum()
    dates = pd.to_datetime(trades["entry_date"].dropna())
    if dates.empty or len(dates) < 2:
        years = 1.0
    else:
        years = max((dates.max() - dates.min()).days / 365.25, 1 / 252)
    return float(total_traded / avg_capital / years)


def compute_all_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    label: str = "",
) -> dict:
    """
    Compute all standard metrics in one call.

    Parameters
    ----------
    equity_curve : daily portfolio value (DatetimeIndex)
    trades       : completed trades DataFrame.
                   Must have: gross_pnl, net_pnl, entry_price, quantity, entry_date
    label        : string prefix for logging / dict keys (e.g. "gross" / "net")

    Returns
    -------
    dict with keys: sharpe, sortino, max_drawdown, cagr, hit_rate,
                    win_loss_ratio, turnover, total_return, n_trades,
                    total_gross_pnl, total_net_pnl
    """
    if equity_curve.empty:
        logger.warning("compute_all_metrics[{}]: empty equity curve", label)
        return {}

    daily_returns = equity_curve.pct_change().dropna()
    avg_cap = equity_curve.mean()

    metrics = {
        "sharpe":          sharpe_ratio(daily_returns),
        "sortino":         sortino_ratio(daily_returns),
        "max_drawdown":    max_drawdown(equity_curve),
        "cagr":            cagr(equity_curve),
        "hit_rate":        hit_rate(trades),
        "win_loss_ratio":  avg_win_loss_ratio(trades),
        "turnover":        turnover(trades, avg_cap),
        "total_return":    float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1),
        "n_trades":        len(trades),
        "total_gross_pnl": float(trades["gross_pnl"].sum()) if "gross_pnl" in trades.columns else 0.0,
        "total_net_pnl":   float(trades["net_pnl"].sum()) if "net_pnl" in trades.columns else 0.0,
    }

    if label:
        logger.info(
            "[{}] Sharpe={:.2f} | Sortino={:.2f} | MaxDD={:.1%} | CAGR={:.1%} "
            "| HitRate={:.1%} | W/L={:.2f} | Trades={}",
            label,
            metrics["sharpe"],
            metrics["sortino"],
            metrics["max_drawdown"],
            metrics["cagr"],
            metrics["hit_rate"],
            metrics["win_loss_ratio"],
            metrics["n_trades"],
        )

    return metrics
