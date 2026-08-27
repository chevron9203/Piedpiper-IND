"""
Central config. All locked plan decisions live here — not scattered across modules.
Load once at import time; fail fast if required vars are missing.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


import warnings as _warnings


def _require(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise EnvironmentError(f"Required env var {key!r} is not set. Check .env.")
    return v


def _optional(key: str) -> str:
    """Returns env var or '' — warns instead of raising, so dashboard runs without creds."""
    v = os.getenv(key, "")
    if not v:
        _warnings.warn(
            f"Credential {key!r} not set — live trading/auth will fail if attempted.",
            stacklevel=2,
        )
    return v


# ── Angel One credentials ─────────────────────────────────────────────────────
ANGELONE_API_KEY     = _optional("ANGELONE_API_KEY")
ANGELONE_CLIENT_CODE = _optional("ANGELONE_CLIENT_CODE")
ANGELONE_MPIN        = _optional("ANGELONE_MPIN")
ANGELONE_TOTP_SECRET = _optional("ANGELONE_TOTP_SECRET")

# ── Storage ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data_store"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "piedpiper.duckdb"))
EOD2_DATA_DIR = Path(os.getenv("EOD2_DATA_DIR", DATA_DIR / "eod2"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATA_DIR.mkdir(parents=True, exist_ok=True)
EOD2_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Notifications ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

# ── Universe ──────────────────────────────────────────────────────────────────
NIFTY500_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
NIFTY200_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
NIFTY100_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
UNIVERSE_CSV_URL  = NIFTY200_CSV_URL   # active pool — change to NIFTY500 to expand

SECTORAL_INDEX_URLS = {
    "Bank":    "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv",
    "IT":      "https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv",
    "Auto":    "https://www.niftyindices.com/IndexConstituent/ind_niftyautolist.csv",
    "Pharma":  "https://www.niftyindices.com/IndexConstituent/ind_niftypharmalist.csv",
    "FMCG":   "https://www.niftyindices.com/IndexConstituent/ind_niftyfmcglist.csv",
    "Metal":   "https://www.niftyindices.com/IndexConstituent/ind_niftymetallist.csv",
    "Realty":  "https://www.niftyindices.com/IndexConstituent/ind_niftyrealtylist.csv",
    "Energy":  "https://www.niftyindices.com/IndexConstituent/ind_niftyenergylist.csv",
    "Infra":   "https://www.niftyindices.com/IndexConstituent/ind_niftyinfralist.csv",
    "Media":   "https://www.niftyindices.com/IndexConstituent/ind_niftymedialist.csv",
    "PSUBank": "https://www.niftyindices.com/IndexConstituent/ind_niftypsubanklist.csv",
}

MIN_AVG_DAILY_TURNOVER_CR = 50.0      # crores — minimum liquidity filter
CORPORATE_ACTION_EXCLUSION_MONTHS = 3  # exclude stocks with recent CA from initial universe
UNIVERSE_SECTOR_CAP_PCT = 0.25         # max 25% of universe from one sector

# ── SmartAPI ──────────────────────────────────────────────────────────────────
INDIA_VIX_TOKEN = "99926017"           # verify against current instrument master in Phase 0
INDIA_VIX_SYMBOL = "India VIX"
NIFTY50_TOKEN = "99926000"
NIFTY50_SYMBOL = "Nifty 50"

SMARTAPI_MAX_DAYS_PER_REQUEST = 2000   # ONE_DAY interval; verify in Phase 0
SMARTAPI_REQUEST_DELAY_SEC = 2.0       # AB1021 fires after ~3 rapid requests; 2s keeps us safely under the burst limit

# ── Feature engineering ───────────────────────────────────────────────────────
FEATURE_VERSION = "v1"
RETURN_HORIZONS = [1, 5, 10, 20, 60]   # calendar days for multi-horizon return features
ROLLING_WINDOWS = [5, 10, 20, 60]      # generic rolling windows

# ── Triple-barrier labeling ───────────────────────────────────────────────────
BARRIER_TAKE_PROFIT_PCT = 0.06         # 6% net-of-cost upper barrier
BARRIER_STOP_LOSS_PCT = 0.03           # 3% net-of-cost lower barrier
BARRIER_MAX_HOLDING_DAYS = 20          # time barrier (trading days)
MIN_HOLD_TRADING_DAYS = 5              # minimum bars before a signal_exit is allowed

# ── Walk-forward / training ───────────────────────────────────────────────────
RETRAIN_CADENCE_TRADING_DAYS = 63      # quarterly (~63 trading days) — LOCKED
MIN_TRAIN_WINDOW_YEARS = 3             # minimum lookback before first model
EMBARGO_DAYS = 10                      # embargo buffer between train/test splits

# ── Portfolio constraints (LOCKED) ───────────────────────────────────────────
MAX_CONCURRENT_POSITIONS = 5
MAX_CAPITAL_DEPLOYED_PCT = 0.80
MAX_POSITIONS_PER_SECTOR = 2
STARTING_VIRTUAL_CAPITAL = 100_000.0  # ₹1 lakh
INTRADAY_CAPITAL         = float(os.getenv("INTRADAY_CAPITAL", "200000"))  # ORB daily capital (₹2L — confirmed via backtest Aug 2026)

# ── Signal gating ────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55            # tuned during backtesting; do not adjust after deployment
OPENING_GAP_FLAG_PCT = 0.005           # 0.5% — flag if open deviates beyond this
VIX_ENTRY_BLOCK_THRESHOLD = 20.0       # India VIX close above this → no new entries that day

# ── Success metrics (LOCKED) ─────────────────────────────────────────────────
TARGET_SHARPE_PAPER = 1.0              # minimum on live Ledger A, not backtest
MAX_DRAWDOWN_KILL = 0.20               # 20% drawdown triggers pause-and-reassess
MIN_BACKTEST_YEARS = 5
MIN_PAPER_TRADING_MONTHS = 3

# ── Backtest ──────────────────────────────────────────────────────────────────
FILL_ASSUMPTION = "next_bar_open"      # LOCKED — no VWAP

# ── NSE market hours (IST) ───────────────────────────────────────────────────
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"
DAILY_RUN_TIME = "16:00"               # after-close run, before midnight re-auth expiry
