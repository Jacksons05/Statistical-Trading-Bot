#!/usr/bin/env bash
# Cron wrapper for one tick of examples/paper_trade_polymarket.py.
#
# Two things a bare `python examples/paper_trade_polymarket.py` in a crontab
# line would get wrong:
#
# Overlap. A run takes ~2.5 minutes (mostly paginating the gamma API), and a
# slow network day can push that further. DuckDB does not support concurrent
# writers to one file -- two cron fires racing on the same database corrupt
# or error rather than merging -- so a run that is still going when the next
# one fires must be skipped, not started alongside it. flock on a dedicated
# lock file (held via a file descriptor, not the `flock command...` form,
# so a lock failure and a script failure stay distinguishable) does that.
#
# Unbounded logs. Left alone, appending forever eventually fills the disk.
# The log is truncated to its last MAX_LOG_BYTES before each run rather than
# rotated by an external tool, to keep this self-contained.
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

REPO_DIR="/home/jackson/Statistical-Trading-Bot"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
DB_PATH="$REPO_DIR/paper_polymarket.duckdb"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/paper_trade_polymarket.log"
LOCK_FILE="$REPO_DIR/.paper_trade_polymarket.lock"
MAX_LOG_BYTES=$((20 * 1024 * 1024))  # 20 MB

mkdir -p "$LOG_DIR"

if [ -f "$LOG_FILE" ] && \
   [ "$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)" -gt "$MAX_LOG_BYTES" ]; then
    tail -c "$MAX_LOG_BYTES" "$LOG_FILE" > "$LOG_FILE.tmp"
    mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') skip: previous run still in progress" \
        >> "$LOG_FILE"
    exit 0
fi

{
    echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
    "$VENV_PYTHON" "$REPO_DIR/examples/paper_trade_polymarket.py" --db "$DB_PATH"
    echo
} >> "$LOG_FILE" 2>&1
