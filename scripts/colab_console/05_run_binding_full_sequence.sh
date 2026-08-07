#!/usr/bin/env bash
# Restore the complete transfer sequence only after matched semantic improvement.
# Usage: 05_run_binding_full_sequence.sh <beta:gamma>

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
CONDITION="${1:-}"
[[ -n "$CONDITION" ]] || fail "Usage: $0 <beta:gamma>"
RESULTS="${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}"
SCORES="$REPO_ROOT/comparisons/action_binding/2b_final_action_binding/manual_scores.csv"
python3 "$REPO_ROOT/scripts/check_binding_scores.py" --scores "$SCORES" --condition "$CONDITION" \
  || fail "The selected condition has not demonstrated matched semantic improvement."
for SEED in 2025 2026 2027 2028 2029; do
  "$CONSOLE_DIR/05_run_task.sh" binding 2b_transfer_action_binding.txt "$CONDITION" "$SEED"
  "$CONSOLE_DIR/05_run_task.sh" binding 2b_transfer_action_binding.txt 0:0 "$SEED"
done
