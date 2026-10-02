"""Paths and the frozen v1 system definition (chosen on DEV 2013-2022, audited)."""
from __future__ import annotations
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent           # repo root / ~/smartpiper
DATA = BASE/"data_store"
RAW = DATA/"nse_raw"
LIVE = Path(os.environ.get("SMART_LIVE", DATA/"smart_live"))   # everything the service writes
MODELS = LIVE/"models"
LOGS = BASE/"logs"

STORE = LIVE/"feature_store.parquet"      # unranked feature rows, one per (decision date, sym)
LABELS = LIVE/"labels.parquet"            # forward returns per row for each horizon
STATE = LIVE/"state.json"                 # listing-age counts etc. (rolled forward daily)
LEDGER = LIVE/"ledger.json"               # paper book: holdings, pending orders, cash, NAV
NAV = LIVE/"nav.csv"                      # daily paper NAV
SIGNAL = LIVE/"signal.json"               # latest weekly decision
ALERTS = LIVE/"alerts.jsonl"              # watcher output

VERSION = "v1.1"     # v1.1: traded-price (not split-adjusted) for price floor + log_price; no insider features
DROP = ("prom_net", "ins_net", "ins_buyers")   # NSE PIT insider feed empty since 2026-05
STEP = 5
WINDOW = 450                              # trading days of prices kept in memory
HORIZONS = (10, 21, 63)
SEEDS = (7, 8, 9)
MOM_W = 0.50                              # v1.1: 50/50 chosen on DEV after the price-level leak fix
BOOK = dict(top_n=20, exit_rank=100, band=0.25, max_corr=0.5)
CORR_WIN = 126
CAPITAL = 200_000.0                       # paper capital (Rs)
THREADS = int(os.environ.get("SMART_THREADS", "1"))   # the box has 1 CPU; Mac bootstrap uses 8
