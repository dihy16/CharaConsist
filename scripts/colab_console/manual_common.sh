#!/usr/bin/env bash
# Shared helpers for the step-by-step manual Colab scripts.

set -euo pipefail

CONSOLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CONSOLE_DIR/../.." && pwd)"
MANUAL_CONFIG="${CHARACONSIST_MANUAL_CONFIG:-$CONSOLE_DIR/manual_config.json}"

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command is unavailable: $1"
}

load_manual_config() {
  [[ -f "$MANUAL_CONFIG" ]] || fail "Missing $MANUAL_CONFIG. Copy manual_config.example.json and edit it first."
  require_command python3
  SESSION="$(python3 "$CONSOLE_DIR/read_manual_config.py" "$MANUAL_CONFIG" session)"
  GPU="$(python3 "$CONSOLE_DIR/read_manual_config.py" "$MANUAL_CONFIG" gpu)"
  REMOTE_ROOT="$(python3 "$CONSOLE_DIR/read_manual_config.py" "$MANUAL_CONFIG" remote_root)"
  MODEL_PATH="$(python3 "$CONSOLE_DIR/read_manual_config.py" "$MANUAL_CONFIG" model_path)"
}

require_colab() {
  require_command colab
}

remote_exec() {
  local script="$1"
  local timeout="${2:-7200}"
  colab exec -s "$SESSION" --timeout "$timeout" -f "$CONSOLE_DIR/$script"
}

task_seed_or_default() {
  if [[ -n "${1:-}" ]]; then
    printf '%s\n' "$1"
  else
    python3 "$CONSOLE_DIR/read_manual_config.py" "$MANUAL_CONFIG" seeds | cut -d, -f1
  fi
}
