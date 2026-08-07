#!/usr/bin/env bash
# Finalize the worker, download batch-summary.json, and stop the session.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab

KEEP=false
[[ "${1:-}" == "--keep" ]] && KEEP=true
remote_exec finish_worker.py 300
LOCAL_RESULTS="${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}"
mkdir -p "$LOCAL_RESULTS"
colab download -s "$SESSION" "$REMOTE_ROOT/results/batch-summary.json" "$LOCAL_RESULTS/batch-summary.json"
if [[ "$KEEP" == false ]]; then
  colab stop -s "$SESSION"
fi
