#!/usr/bin/env bash
# Render a downloaded role-action control/intervention pair. Usage: 06_compare_role_action.sh <prompt.txt> [seed]

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
PROMPT="${1:-}"
[[ -n "$PROMPT" ]] || fail "Usage: $0 <prompt.txt> [seed]"
python3 "$REPO_ROOT/compare_role_action_results.py" --results-root "${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}" --prompt "$(basename "${PROMPT%.txt}")" --seed "${2:-2025}"
