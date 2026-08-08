"""
DuckDB-backed storage for all persistent data.

Tables:
  raw_ohlcv      — unadjusted prices from SmartAPI (never overwritten)
  adjusted_ohlcv — adjusted prices from EOD2 (refreshed daily)
  features       — computed feature vectors per symbol per date
  signals        — daily signal output with confidence scores
  ledger_a       — virtual portfolio positions and trades
  ledger_b       — actual portfolio trades and declined signals
  ca_log         — corporate action adjustment audit log

No silent overwrites of raw data. Raw inserts use INSERT OR IGNORE.
All tables include an 'ingested_at' timestamp for auditability.
"""
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd
from loguru import logger
from config import settings


def get_db() -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB database file."""
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(settings.DB_PATH))


@contextmanager
def db_conn():
    """Context manager for database connections."""
    conn = get_db()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    """Create all tables if they don't already exist."""
    with db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_ohlcv (
                symbol       VARCHAR NOT NULL,
                dt           DATE NOT NULL,
                open         DOUBLE,
                high         DOUBLE,
                low          DOUBLE,
                close        DOUBLE,
                volume       BIGINT,
                ingested_at  TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (symbol, dt)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS adjusted_ohlcv (
                symbol       VARCHAR NOT NULL,
                dt           DATE NOT NULL,
                open         DOUBLE,
                high         DOUBLE,
                low          DOUBLE,
                close        DOUBLE,
                volume       BIGINT,
                source       VARCHAR DEFAULT 'eod2',
                ingested_at  TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (symbol, dt)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol       VARCHAR NOT NULL,
                dt           DATE NOT NULL,
                version      VARCHAR NOT NULL,
                feature_data JSON,
                computed_at  TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (symbol, dt, version)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                dt              DATE NOT NULL,
                symbol          VARCHAR NOT NULL,
                direction       VARCHAR,
                confidence      DOUBLE,
                entry_low       DOUBLE,
                entry_high      DOUBLE,
                stop_loss       DOUBLE,
                target          DOUBLE,
                max_hold_days   INTEGER,
                position_size   DOUBLE,
                model_version   VARCHAR,
                generated_at    TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (dt, symbol)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ledger_a (
                trade_id        VARCHAR PRIMARY KEY,
                symbol          VARCHAR NOT NULL,
                entry_date      DATE,
                entry_price     DOUBLE,
                quantity        INTEGER,
                stop_loss       DOUBLE,
                target          DOUBLE,
                exit_date       DATE,
                exit_price      DOUBLE,
                exit_reason     VARCHAR,
                gross_pnl       DOUBLE,
                net_pnl         DOUBLE,
                confidence      DOUBLE,
                sector          VARCHAR,
                status          VARCHAR DEFAULT 'open',
                created_at      TIMESTAMP DEFAULT current_timestamp
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ledger_b (
                trade_id        VARCHAR PRIMARY KEY,
                symbol          VARCHAR NOT NULL,
                signal_date     DATE,
                action          VARCHAR NOT NULL,
                entry_date      DATE,
                entry_price     DOUBLE,
                quantity        INTEGER,
                gtt_stop        DOUBLE,
                gtt_target      DOUBLE,
                exit_date       DATE,
                exit_price      DOUBLE,
                exit_reason     VARCHAR,
                gross_pnl       DOUBLE,
                net_pnl         DOUBLE,
                notes           VARCHAR,
                logged_at       TIMESTAMP DEFAULT current_timestamp
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ca_log (
                symbol          VARCHAR NOT NULL,
                ca_date         DATE NOT NULL,
                ca_type         VARCHAR,
                ratio           DOUBLE,
                adjustment_applied BOOLEAN,
                logged_at       TIMESTAMP DEFAULT current_timestamp
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intraday_ohlcv (
                symbol       VARCHAR NOT NULL,
                dt           TIMESTAMP NOT NULL,
                interval     VARCHAR NOT NULL,
                open         DOUBLE,
                high         DOUBLE,
                low          DOUBLE,
                close        DOUBLE,
                volume       BIGINT,
                ingested_at  TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (symbol, dt, interval)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intraday_trades (
                trade_id        VARCHAR PRIMARY KEY,
                strategy        VARCHAR NOT NULL DEFAULT 'orb',
                symbol          VARCHAR NOT NULL,
                trade_date      DATE NOT NULL,
                direction       VARCHAR NOT NULL,        -- 'LONG' | 'SHORT'
                entry_time      TIMESTAMP,
                entry_price     DOUBLE,
                qty             INTEGER,
                stop_loss       DOUBLE,
                target          DOUBLE,
                exit_time       TIMESTAMP,
                exit_price      DOUBLE,
                exit_reason     VARCHAR,                 -- 'target'|'stop'|'squareoff'|'manual'
                gross_pnl       DOUBLE,
                charges         DOUBLE,
                net_pnl         DOUBLE,
                angel_entry_id  VARCHAR,                 -- Angel One order_id
                angel_exit_id   VARCHAR,
                paper_trade     BOOLEAN DEFAULT true,
                created_at      TIMESTAMP DEFAULT current_timestamp
            )
        """)
    logger.info("Database schema initialised at {}", settings.DB_PATH)


# ── Raw OHLCV ─────────────────────────────────────────────────────────────────

def upsert_raw_ohlcv(symbol: str, df: pd.DataFrame) -> int:
    """
    Insert raw OHLCV rows. Uses INSERT OR IGNORE — never overwrites existing raw data.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0
    df = df.copy().reset_index()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"datetime": "dt", "date": "dt"}, inplace=True)
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    df["symbol"] = symbol

    with db_conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM raw_ohlcv WHERE symbol = ?", [symbol]).fetchone()[0]
        conn.register("_raw_insert", df[["symbol", "dt", "open", "high", "low", "close", "volume"]])
        conn.execute("""
            INSERT OR IGNORE INTO raw_ohlcv (symbol, dt, open, high, low, close, volume)
            SELECT symbol, dt, open, high, low, close, volume FROM _raw_insert
        """)
        after = conn.execute("SELECT COUNT(*) FROM raw_ohlcv WHERE symbol = ?", [symbol]).fetchone()[0]
    inserted = after - before
    logger.debug("raw_ohlcv: {} rows inserted for {}", inserted, symbol)
    return inserted


def load_raw_ohlcv(symbol: str, from_date: Optional[date] = None,
                   to_date: Optional[date] = None) -> pd.DataFrame:
    with db_conn() as conn:
        query = "SELECT dt, open, high, low, close, volume FROM raw_ohlcv WHERE symbol = ?"
        params = [symbol]
        if from_date:
            query += " AND dt >= ?"
            params.append(from_date)
        if to_date:
            query += " AND dt <= ?"
            params.append(to_date)
        query += " ORDER BY dt"
        df = conn.execute(query, params).df()
    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt")


# ── Adjusted OHLCV ────────────────────────────────────────────────────────────

def upsert_adjusted_ohlcv(symbol: str, df: pd.DataFrame, source: str = "eod2") -> int:
    if df.empty:
        return 0
    df = df.copy().reset_index()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"datetime": "dt", "date": "dt"}, inplace=True)
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    df["symbol"] = symbol
    df["source"] = source

    with db_conn() as conn:
        conn.register("_adj_insert", df[["symbol", "dt", "open", "high", "low", "close", "volume", "source"]])
        conn.execute("""
            INSERT OR REPLACE INTO adjusted_ohlcv (symbol, dt, open, high, low, close, volume, source)
            SELECT symbol, dt, open, high, low, close, volume, source FROM _adj_insert
        """)
        count = conn.execute(
            "SELECT COUNT(*) FROM adjusted_ohlcv WHERE symbol = ?", [symbol]
        ).fetchone()[0]
    logger.debug("adjusted_ohlcv: {} total rows for {}", count, symbol)
    return count


def load_adjusted_ohlcv(symbol: str, from_date: Optional[date] = None,
                         to_date: Optional[date] = None) -> pd.DataFrame:
    with db_conn() as conn:
        query = "SELECT dt, open, high, low, close, volume FROM adjusted_ohlcv WHERE symbol = ?"
        params = [symbol]
        if from_date:
            query += " AND dt >= ?"
            params.append(from_date)
        if to_date:
            query += " AND dt <= ?"
            params.append(to_date)
        query += " ORDER BY dt"
        df = conn.execute(query, params).df()
    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt")


# ── Features ──────────────────────────────────────────────────────────────────

def upsert_features(symbol: str, dt: date, version: str, feature_dict: dict) -> None:
    import json
    with db_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO features (symbol, dt, version, feature_data)
            VALUES (?, ?, ?, ?)
        """, [symbol, dt, version, json.dumps(feature_dict)])


def load_features(symbols: list[str], from_date: date, to_date: date,
                  version: str) -> pd.DataFrame:
    import json
    with db_conn() as conn:
        placeholders = ",".join("?" * len(symbols))
        df = conn.execute(f"""
            SELECT symbol, dt, feature_data FROM features
            WHERE symbol IN ({placeholders}) AND dt >= ? AND dt <= ? AND version = ?
            ORDER BY symbol, dt
        """, symbols + [from_date, to_date, version]).df()
    if df.empty:
        return df
    features_expanded = df["feature_data"].apply(json.loads).apply(pd.Series)
    result = pd.concat([df[["symbol", "dt"]], features_expanded], axis=1)
    result["dt"] = pd.to_datetime(result["dt"])
    return result


# ── Signals ───────────────────────────────────────────────────────────────────

def save_signals(signals: list[dict]) -> None:
    if not signals:
        return
    df = pd.DataFrame(signals)
    with db_conn() as conn:
        conn.register("_sig_insert", df)
        conn.execute("""
            INSERT OR REPLACE INTO signals
                (dt, symbol, direction, confidence, entry_low, entry_high,
                 stop_loss, target, max_hold_days, position_size, model_version)
            SELECT dt, symbol, direction, confidence, entry_low, entry_high,
                   stop_loss, target, max_hold_days, position_size, model_version
            FROM _sig_insert
        """)


def load_signals(dt: date) -> pd.DataFrame:
    with db_conn() as conn:
        return conn.execute(
            "SELECT * FROM signals WHERE dt = ? ORDER BY confidence DESC", [dt]
        ).df()


# ── Intraday OHLCV ────────────────────────────────────────────────────────────

def upsert_intraday_ohlcv(symbol: str, df: pd.DataFrame, interval: str) -> int:
    """
    Insert intraday OHLCV rows. Uses INSERT OR IGNORE — never overwrites existing data.
    df must have a DatetimeIndex (with time component) and columns: open, high, low, close, volume.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0
    d = df.copy().reset_index()
    d.columns = [c.lower() for c in d.columns]
    d.rename(columns={"datetime": "dt", "date": "dt", "index": "dt"}, inplace=True)
    d["dt"] = pd.to_datetime(d["dt"])
    d["symbol"]   = symbol
    d["interval"] = interval

    with db_conn() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM intraday_ohlcv WHERE symbol = ? AND interval = ?",
            [symbol, interval]
        ).fetchone()[0]
        conn.register("_intra_insert", d[["symbol", "dt", "interval", "open", "high", "low", "close", "volume"]])
        conn.execute("""
            INSERT OR IGNORE INTO intraday_ohlcv
                (symbol, dt, interval, open, high, low, close, volume)
            SELECT symbol, dt, interval, open, high, low, close, volume
            FROM _intra_insert
        """)
        after = conn.execute(
            "SELECT COUNT(*) FROM intraday_ohlcv WHERE symbol = ? AND interval = ?",
            [symbol, interval]
        ).fetchone()[0]
    inserted = after - before
    logger.debug("intraday_ohlcv: {} rows inserted for {} @ {}", inserted, symbol, interval)
    return inserted


def load_intraday_ohlcv(
    symbol: str,
    interval: str,
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    with db_conn() as conn:
        query  = "SELECT dt, open, high, low, close, volume FROM intraday_ohlcv WHERE symbol = ? AND interval = ?"
        params = [symbol, interval]
        if from_dt:
            query += " AND dt >= ?"
            params.append(from_dt)
        if to_dt:
            query += " AND dt <= ?"
            params.append(to_dt)
        query += " ORDER BY dt"
        df = conn.execute(query, params).df()
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt")


# ── Intraday Trades ───────────────────────────────────────────────────────────

def log_intraday_trade(trade: dict) -> None:
    """Insert or replace one intraday trade record."""
    with db_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO intraday_trades
                (trade_id, strategy, symbol, trade_date, direction, entry_time, entry_price,
                 qty, stop_loss, target, exit_time, exit_price, exit_reason,
                 gross_pnl, charges, net_pnl, angel_entry_id, angel_exit_id, paper_trade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            trade["trade_id"],
            trade.get("strategy", "orb"),
            trade["symbol"],
            trade["trade_date"],
            trade["direction"],
            trade.get("entry_time"),
            trade.get("entry_price"),
            trade.get("qty"),
            trade.get("stop_loss"),
            trade.get("target"),
            trade.get("exit_time"),
            trade.get("exit_price"),
            trade.get("exit_reason"),
            trade.get("gross_pnl"),
            trade.get("charges"),
            trade.get("net_pnl"),
            trade.get("angel_entry_id"),
            trade.get("angel_exit_id"),
            trade.get("paper_trade", True),
        ])


def load_intraday_trades(
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    strategy: Optional[str] = None,
    paper_only: bool = False,
) -> pd.DataFrame:
    """Load intraday trade log with optional filters."""
    with db_conn() as conn:
        query  = "SELECT * FROM intraday_trades WHERE 1=1"
        params: list = []
        if from_date:
            query += " AND trade_date >= ?"; params.append(from_date)
        if to_date:
            query += " AND trade_date <= ?"; params.append(to_date)
        if strategy:
            query += " AND strategy = ?"; params.append(strategy)
        if paper_only:
            query += " AND paper_trade = true"
        query += " ORDER BY entry_time"
        return conn.execute(query, params).df()


if __name__ == "__main__":
    init_schema()
    logger.info("Schema ready")
