#!/usr/bin/env bash
# Record the required human confirmation after inspecting frozen character maps.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
RESULTS="${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}"
REPORT="$RESULTS/action_binding_ablation/beta_0p00_gamma_0p00/seed_2025/bg_fg/2b_final_action_binding/prompt_0/action_binding/map_preflight.json"
python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); assert data["status"] == "pass", data' "$REPORT"
APPROVAL="$RESULTS/action_binding_ablation/.maps_approved_seed_2025"
printf 'C1=man C2=woman; manually approved\n' > "$APPROVAL"
printf 'Recorded map approval: %s\n' "$APPROVAL"
