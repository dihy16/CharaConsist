#!/usr/bin/env bash
# Run one configured worker task and download/extract its resulting archive.
# Usage: 05_run_task.sh <lambda|component|role|binding|routing> <prompt.txt> <value> [seed]
# Binding values use beta:gamma, for example 1:0.5.
# Routing values use mode:beta:gamma, for example hard:1:0.5.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab

MODE="${1:-}"
PROMPT="${2:-}"
VALUE="${3:-}"
[[ -n "$MODE" && -n "$PROMPT" && -n "$VALUE" ]] || fail "Usage: $0 <lambda|component|role|binding|routing> <prompt.txt> <value> [seed]"
SEED="$(task_seed_or_default "${4:-}")"

case "$MODE" in lambda|component|role|binding|routing) ;; *) fail "Mode must be lambda, component, role, binding, or routing." ;; esac
LOCAL_TASK="$(mktemp)"
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOCAL_TASK" "$LOG_FILE"' EXIT
PYTHONPATH="$REPO_ROOT" python3 "$CONSOLE_DIR/write_manual_task.py" \
  --output "$LOCAL_TASK" --mode "$MODE" --prompt "$PROMPT" --value "$VALUE" --seed "$SEED"
RELATIVE_OUTPUT="$(python3 "$CONSOLE_DIR/read_manual_task.py" "$LOCAL_TASK" relative_output)"

colab upload -s "$SESSION" "$LOCAL_TASK" /content/characonsist_console_task.json
remote_exec run_manual_task.py | tee "$LOG_FILE"
if ! grep -q 'CHARACONSIST_TASK_RESULT=.*"status": "\(success\|skipped\)"' "$LOG_FILE"; then
  fail "Worker did not report a successful task; no archive was downloaded."
fi

LOCAL_RESULTS="${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}"
ARCHIVE="$(mktemp)"
trap 'rm -f "$LOCAL_TASK" "$LOG_FILE" "$ARCHIVE"' EXIT
mkdir -p "$LOCAL_RESULTS"
colab download -s "$SESSION" "$REMOTE_ROOT/result_archives/$RELATIVE_OUTPUT.tar.gz" "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$LOCAL_RESULTS"
printf 'Downloaded: %s/%s\n' "$LOCAL_RESULTS" "$RELATIVE_OUTPUT"
