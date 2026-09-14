#!/bin/bash
# Install all Piedpiper cron jobs on the Oracle Free Tier box.
# Run once on the server: bash scripts/setup_cron.sh
# Requires: project at ~/piedpiper, venv at ~/piedpiper/.venv

set -e
PROJ="/home/ubuntu/piedpiper"
PYTHON="$PROJ/.venv/bin/python3"

if [ ! -f "$PYTHON" ]; then echo "ERROR: venv not found at $PYTHON"; exit 1; fi
echo "Using Python: $PYTHON"
mkdir -p "$PROJ/logs"

CRON_BLOCK="CRON_TZ=Asia/Kolkata

# ── Piedpiper — validated momentum system (PAPER) ──────────────────────────────
# EOD data chain — runs after NSE bhavcopy posts (~19:00 IST)
0  19 * * 1-5  cd $PROJ/data_store/eod2/src && $PYTHON init.py >> $PROJ/logs/eod2_update.log 2>&1
15 19 * * 1-5  cd $PROJ && $PYTHON scripts/ingest_data.py >> logs/ingest_data.log 2>&1
# EOD monitor + performance tracker (after data is fresh)
30 19 * * 1-5  cd $PROJ && $PYTHON scripts/momentum_monitor.py >> logs/momentum_monitor.log 2>&1
35 19 * * 1-5  cd $PROJ && $PYTHON scripts/performance_tracker.py >> logs/perf_tracker.log 2>&1
# Monthly signals — 1st of each month 08:00 IST
# S1 (multi-asset momentum + gold + US)
0  8  1 * *   cd $PROJ && $PYTHON scripts/momentum_live.py >> logs/momentum_live.log 2>&1
# S4 (pure MID-cap, regime-gated) + S5 (always invested) — runs 5 min after S1
5  8  1 * *   cd $PROJ && $PYTHON scripts/momentum_live_pure.py >> logs/momentum_live_pure.log 2>&1
# ── End Piedpiper ──────────────────────────────────────────────────────────────
"

# Install into crontab (idempotent: strip old Piedpiper block, add fresh)
( crontab -l 2>/dev/null | grep -v "Piedpiper\|momentum_live\|momentum_monitor\|performance_tracker\|ingest_data\|eod2_update\|End Piedpiper\|CRON_TZ" ; echo "$CRON_BLOCK" ) | crontab -

echo ""
echo "Cron installed. Verify with: crontab -l"
echo ""
echo "Schedule (Asia/Kolkata / IST):"
echo "  1st of month 08:00  → momentum_live.py       (S1 signal)"
echo "  1st of month 08:05  → momentum_live_pure.py  (S4 + S5 signals)"
echo "  Mon-Fri     19:00   → eod2 init.py            (NSE data update)"
echo "  Mon-Fri     19:15   → ingest_data.py          (DuckDB ingest)"
echo "  Mon-Fri     19:30   → momentum_monitor.py     (portfolio check)"
echo "  Mon-Fri     19:35   → performance_tracker.py  (NAV record)"
echo ""
echo "Dashboard (run once as a persistent process, NOT via cron):"
echo "  nohup .venv/bin/python3 scripts/paper_dashboard.py >> logs/dashboard.log 2>&1 &"
echo "  Access via SSH tunnel: ssh -L 5002:localhost:5002 ubuntu@<box-ip>"
