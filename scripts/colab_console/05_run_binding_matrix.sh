#!/usr/bin/env bash
# Run four matched conditions x five seeds after both mechanical gates pass.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
RESULTS="${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}"
AUDIT="$RESULTS/action_binding_ablation/mechanical_audit_seed_2025.json"
python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); assert data["status"] == "pass", data' "$AUDIT" \
  || fail "Mechanical audit has not passed."

for CONDITION in 0:0 1:0 1:0.5 2:1; do
  for SEED in 2025 2026 2027 2028 2029; do
    "$CONSOLE_DIR/05_run_task.sh" binding 2b_final_action_binding.txt "$CONDITION" "$SEED"
  done
done
python3 "$REPO_ROOT/compare_action_binding_results.py" --results-root "$RESULTS"
