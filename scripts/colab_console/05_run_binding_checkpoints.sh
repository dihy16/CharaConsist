#!/usr/bin/env bash
# Run zero-equivalence and weak-contrastive checkpoints at seed 2025.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
RESULTS="${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}"
APPROVAL="$RESULTS/action_binding_ablation/.maps_approved_seed_2025"
[[ -f "$APPROVAL" ]] || fail "Inspect and approve the C1/C2 maps first."
python3 -c 'import numpy, PIL' 2>/dev/null || fail \
  "WSL python3 needs NumPy and Pillow. Run: python3 -m venv .venv && source .venv/bin/activate && python -m pip install numpy pillow"

"$CONSOLE_DIR/05_run_task.sh" binding 2b_final_baseline.txt 0:0 2025
"$CONSOLE_DIR/05_run_task.sh" binding 2b_final_action_binding.txt 0:0 2025
"$CONSOLE_DIR/05_run_task.sh" binding 2b_final_action_binding.txt 1:0.5 2025

BASE="$RESULTS/action_binding_ablation"
python3 "$REPO_ROOT/scripts/audit_action_binding.py" \
  --reference "$BASE/beta_0p00_gamma_0p00/seed_2025/bg_fg/2b_final_baseline/prompt_0" \
  --zero "$BASE/beta_0p00_gamma_0p00/seed_2025/bg_fg/2b_final_action_binding/prompt_0" \
  --weak "$BASE/beta_1p00_gamma_0p50/seed_2025/bg_fg/2b_final_action_binding/prompt_0" \
  --output "$BASE/mechanical_audit_seed_2025.json"
