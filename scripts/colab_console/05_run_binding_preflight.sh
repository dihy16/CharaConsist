#!/usr/bin/env bash
# Generate the tagged zero-strength run and its frozen C1/C2 map overlays.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
"$CONSOLE_DIR/05_run_task.sh" binding 2b_final_action_binding.txt 0:0 2025
printf '%s\n' "Inspect results_colab/action_binding_ablation/beta_0p00_gamma_0p00/seed_2025/bg_fg/2b_final_action_binding/prompt_0/action_binding/*_overlay.png"
printf '%s\n' "Then run 05_approve_binding_maps.sh only if C1 locates the man and C2 locates the woman."
