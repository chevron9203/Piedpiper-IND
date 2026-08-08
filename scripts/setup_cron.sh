#!/bin/bash
# Setup all Piedpiper cron jobs.
# Run once: bash scripts/setup_cron.sh
# System must be in Asia/Kolkata (IST) timezone — confirmed on this machine.

PROJ="/Users/abc/Documents/ozi-code/piedpiper"
PYTHON=$(which python3)

# Resolve virtualenv python if it exists
if [ -f "$PROJ/.venv/bin/python3" ]; then
    PYTHON="$PROJ/.venv/bin/python3"
fi

echo "Using Python: $PYTHON"
echo "Project: $PROJ"

# Build the cron block
CRON_BLOCK="
# ── Piedpiper automated trading ──────────────────────────────────────────
# Intraday ORB: fetch opening range (9:15-9:30 candle) and compute signals
# Runs at 9:31 AM IST Mon-Fri  (opening range candle completes at 9:30)
31 9 * * 1-5 cd $PROJ && $PYTHON scripts/intraday_run.py --capital 50000 >> logs/intraday_run.log 2>&1

# Intraday monitor: check LTP vs target/stop every 15 min, trail stop to BE, auto-exit at target
# Runs every 15 min from 9:45 AM to 2:45 PM IST Mon-Fri
*/15 9-14 * * 1-5 cd $PROJ && $PYTHON scripts/intraday_monitor.py >> logs/intraday_monitor.log 2>&1

# Intraday square-off: close all paper/live positions before Angel One auto-SQO at 3:15 PM
# Runs at 3:10 PM IST Mon-Fri
10 15 * * 1-5 cd $PROJ && $PYTHON scripts/intraday_squareoff.py >> logs/intraday_squareoff.log 2>&1

# Monthly momentum: check every trading day evening; only fires on last trading day of month
# Runs at 6:00 PM IST Mon-Fri
0 18 * * 1-5 cd $PROJ && $PYTHON scripts/monthly_run.py >> logs/monthly_run.log 2>&1
# ── End Piedpiper ─────────────────────────────────────────────────────────
"

# Make sure logs dir exists
mkdir -p "$PROJ/logs"

# Install into crontab (remove old piedpiper block if present, add fresh)
( crontab -l 2>/dev/null | grep -v "Piedpiper\|intraday_run\|intraday_monitor\|intraday_squareoff\|monthly_run\|End Piedpiper" ; echo "$CRON_BLOCK" ) | crontab -

echo ""
echo "Cron jobs installed. Current crontab:"
echo "─────────────────────────────────────────"
crontab -l
echo "─────────────────────────────────────────"
echo ""
echo "Schedule (IST):"
echo "  9:31 AM  Mon-Fri → intraday_run.py      (ORB signals + paper orders)"
echo "  */15min  Mon-Fri → intraday_monitor.py  (LTP check, target exit, trail stop)"
echo "  3:10 PM  Mon-Fri → intraday_squareoff.py (close all positions)"
echo "  6:00 PM  Mon-Fri → monthly_run.py        (fires only on month-end)"
echo ""
echo "Manual early close (outside cron, run anytime during market hours):"
echo "  python scripts/intraday_monitor.py --close RELIANCE"
echo "  python scripts/intraday_monitor.py --close ALL"
echo ""
echo "Logs: $PROJ/logs/"
echo "  tail -f logs/intraday_run.log"
echo "  tail -f logs/intraday_squareoff.log"
echo "  tail -f logs/monthly_run.log"
