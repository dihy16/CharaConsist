#!/usr/bin/env bash
# Render a downloaded lambda sweep. Usage: 06_compare_lambda.sh <prompt.txt> [lambdas] [seed]

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
PROMPT="${1:-}"
[[ -n "$PROMPT" ]] || fail "Usage: $0 <prompt.txt> [lambdas] [seed]"
LAMBDAS="${2:-0,0.5,1}"
SEED="${3:-2025}"
python3 "$REPO_ROOT/visualize_lambda_results.py" --results-root "${CHARACONSIST_RESULTS_DIR:-$REPO_ROOT/results_colab}" --prompt-file "$PROMPT" --lambdas "$LAMBDAS" --seed "$SEED"
