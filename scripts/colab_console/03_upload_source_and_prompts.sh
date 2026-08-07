#!/usr/bin/env bash
# Archive local source/prompts, upload them, and extract them on the Colab VM.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manual_common.sh"
load_manual_config
require_colab

PROMPTS_DIR="${1:-$REPO_ROOT/prompts/stress_test}"
[[ -d "$PROMPTS_DIR" ]] || fail "Prompt directory not found: $PROMPTS_DIR"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

SOURCE_ARCHIVE="$WORK_DIR/source.tar.gz"
PROMPTS_ARCHIVE="$WORK_DIR/prompts.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' -C "$REPO_ROOT" -czf "$SOURCE_ARCHIVE" \
  inference.py make_story_image.py run_colab_remote.py run_batch_inference.py \
  run_colab_worker.py compare_component_results.py compare_role_action_results.py \
  visualize_lambda_results.py requirements-colab.txt characonsist models
tar -C "$PROMPTS_DIR" -czf "$PROMPTS_ARCHIVE" .

remote_exec prepare_remote_root.py 300
colab upload -s "$SESSION" "$SOURCE_ARCHIVE" "$REMOTE_ROOT/source.tar.gz"
colab upload -s "$SESSION" "$PROMPTS_ARCHIVE" "$REMOTE_ROOT/prompts.tar.gz"
remote_exec extract_source.py
printf 'Uploaded %s prompt file(s).\n' "$(find "$PROMPTS_DIR" -type f -name '*.txt' | wc -l | tr -d '[:space:]')"
