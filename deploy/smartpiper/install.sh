#!/usr/bin/env bash
# Installs smartpiper as a SEPARATE service. Touches nothing of ~/piedpiper, its venv,
# its crontab or its dashboard. Re-runnable.
set -euo pipefail
cd /home/ubuntu/smartpiper
mkdir -p logs data_store/smart_live
command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV="$HOME/.local/bin/uv"; [ -x "$UV" ] || UV=uv
[ -x .venv/bin/python ] || "$UV" venv --python 3.12 .venv
"$UV" pip install --python .venv/bin/python -q -r deploy/smartpiper/requirements.txt
chmod 600 .env 2>/dev/null || true
sudo cp deploy/smartpiper/smartpiper-*.service deploy/smartpiper/smartpiper-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
echo "installed. enable with: sudo systemctl enable --now smartpiper-daily.timer smartpiper-monthly.timer smartpiper-watcher.service"
