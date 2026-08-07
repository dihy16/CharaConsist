#!/usr/bin/env bash
# Run off/hard routing x zero/weak action bias across five matched seeds.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config

for MODE in off hard; do
  for CONDITION in 0:0 1:0.5; do
    for SEED in 2025 2026 2027 2028 2029; do
      "$CONSOLE_DIR/05_run_task.sh" routing 2b_final_action_binding.txt "$MODE:$CONDITION" "$SEED"
    done
  done
done

RESULTS="${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}"
python3 "$REPO_ROOT/compare_entity_routing_results.py" --results-root "$RESULTS"
PYTHONPATH="$REPO_ROOT" python3 "$REPO_ROOT/scripts/audit_entity_routing.py" \
  --results-root "$RESULTS"
