#!/usr/bin/env bash
# Run CharaConsist prompt batches on a Google Colab CLI session.
#
# The FLUX model is intentionally not uploaded: place it in the Colab runtime
# first (for example by mounting Drive), then pass its *remote* path.
#
# Usage:
#   bash run_colab.sh <prompts_folder> [session_name] --model-path <remote_path> [options]
#
# Example:
#   bash run_colab.sh prompts/stress_test my-session \
#     --model-path /content/drive/MyDrive/models/flux-dev --gpu A100

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_colab.sh <prompts_folder> [session_name] --model-path <remote_path> [options]

Options:
  --session <name>       Session name (alternative to the optional positional name)
  --gpu <A100|H100>      Required accelerator (mode 0 needs about 37 GB VRAM)
  --model-path <path>    Required path to FLUX.1-dev on the Colab VM
  --model-repo <repo>    Hugging Face fallback (default: black-forest-labs/FLUX.1-dev)
  --timeout <seconds>    Timeout for dependency setup and inference (default: 7200)
  --output-dir <path>    Local directory for downloaded results (default: results_colab)
  --keep                 Leave the Colab session running after completion or failure
  -h, --help             Show this help
EOF
}

PROMPTS_FOLDER=""
SESSION_NAME=""
GPU=""
MODEL_PATH=""
MODEL_REPO="black-forest-labs/FLUX.1-dev"
EXEC_TIMEOUT=7200
LOCAL_OUTPUT_DIR="results_colab"
KEEP_SESSION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      [[ $# -ge 2 ]] || { echo "Error: --session needs a value" >&2; exit 2; }
      SESSION_NAME="$2"
      shift 2
      ;;
    --gpu)
      [[ $# -ge 2 ]] || { echo "Error: --gpu needs a value" >&2; exit 2; }
      GPU="$2"
      shift 2
      ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "Error: --model-path needs a value" >&2; exit 2; }
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-repo)
      [[ $# -ge 2 ]] || { echo "Error: --model-repo needs a value" >&2; exit 2; }
      MODEL_REPO="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "Error: --output-dir needs a value" >&2; exit 2; }
      LOCAL_OUTPUT_DIR="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "Error: --timeout needs a value" >&2; exit 2; }
      EXEC_TIMEOUT="$2"
      shift 2
      ;;
    --keep)
      KEEP_SESSION=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$PROMPTS_FOLDER" ]]; then
        PROMPTS_FOLDER="$1"
      elif [[ -z "$SESSION_NAME" ]]; then
        SESSION_NAME="$1"
      else
        echo "Error: unexpected argument '$1'" >&2
        usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

[[ -n "$PROMPTS_FOLDER" ]] || { usage >&2; exit 2; }
[[ -n "$MODEL_PATH" ]] || { echo "Error: --model-path is required" >&2; usage >&2; exit 2; }
[[ "$EXEC_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Error: --timeout must be a positive number of seconds" >&2; exit 2; }
[[ -n "$GPU" ]] || { echo "Error: --gpu is required; choose A100 or H100 for init mode 0" >&2; exit 2; }
[[ "$GPU" =~ ^(A100|H100)$ ]] || { echo "Error: unsupported GPU '$GPU'; init mode 0 requires A100 or H100" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer an already-exported token. Otherwise load the project's trusted .env
# file so users do not need to export HF_TOKEN for every invocation.
HF_TOKEN_SOURCE="environment"
if [[ -z "${HF_TOKEN:-}" && -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
  HF_TOKEN_SOURCE="$SCRIPT_DIR/.env"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  # Avoid carrying a Windows CRLF terminator into Hugging Face authentication.
  HF_TOKEN="${HF_TOKEN%$'\r'}"
  echo "  Hugging Face token: loaded from $HF_TOKEN_SOURCE"
fi

PROMPTS_FOLDER="$(cd "$PROMPTS_FOLDER" && pwd)" || { echo "Error: prompts folder not found" >&2; exit 1; }
[[ -d "$PROMPTS_FOLDER" ]] || { echo "Error: '$PROMPTS_FOLDER' is not a directory" >&2; exit 1; }

PROMPT_COUNT="$(find "$PROMPTS_FOLDER" -type f -name '*.txt' | wc -l | tr -d '[:space:]')"
[[ "$PROMPT_COUNT" -gt 0 ]] || { echo "Error: no .txt files found in '$PROMPTS_FOLDER'" >&2; exit 1; }

SESSION_NAME="${SESSION_NAME:-characonsist-$(date +%s)}"
REMOTE_ROOT="/content/CharaConsist"
REMOTE_PROMPTS="$REMOTE_ROOT/prompts_batch"
REMOTE_RESULTS="$REMOTE_ROOT/results"
WORK_DIR="$(mktemp -d)"
SOURCE_ARCHIVE="$WORK_DIR/characonsist-source.tar.gz"
PROMPTS_ARCHIVE="$WORK_DIR/prompts.tar.gz"
LOCAL_FAILURES_FILE="$WORK_DIR/local-result-delivery-failures.tsv"
SESSION_CREATED=false
RESULTS_DOWNLOADED=false
PENDING_REMOTE_ARCHIVE=""
PENDING_LOCAL_ARCHIVE=""
: > "$LOCAL_FAILURES_FILE"

cleanup() {
  local exit_code=$?
  if [[ "$SESSION_CREATED" == true && "$KEEP_SESSION" != true ]]; then
    if [[ -n "$PENDING_REMOTE_ARCHIVE" && -n "$PENDING_LOCAL_ARCHIVE" ]]; then
      echo "Attempting to recover the pending finalized result archive..."
      mkdir -p "$(dirname "$PENDING_LOCAL_ARCHIVE")" "$LOCAL_OUTPUT_DIR"
      if colab download -s "$SESSION_NAME" "$PENDING_REMOTE_ARCHIVE" "$PENDING_LOCAL_ARCHIVE"; then
        tar -xzf "$PENDING_LOCAL_ARCHIVE" -C "$LOCAL_OUTPUT_DIR" \
          || echo "Warning: could not extract the recovered result archive." >&2
      else
        echo "Warning: could not download the pending result archive during cleanup." >&2
      fi
    fi
    merge_remote_summary \
      || echo "Warning: could not download the batch summary during cleanup." >&2
    echo "Stopping Colab session '$SESSION_NAME'..."
    colab stop -s "$SESSION_NAME" || echo "Warning: could not stop '$SESSION_NAME'; run: colab stop -s $SESSION_NAME" >&2
  elif [[ "$SESSION_CREATED" == true ]]; then
    echo "Session '$SESSION_NAME' is still running (stop it with: colab stop -s $SESSION_NAME)"
  fi
  rm -rf "$WORK_DIR"
  return "$exit_code"
}
trap cleanup EXIT

echo "CharaConsist batch inference on Colab CLI"
echo "  Session: $SESSION_NAME"
echo "  Prompts: $PROMPT_COUNT file(s) from $PROMPTS_FOLDER"
echo "  Remote model: $MODEL_PATH"
echo "  Model download fallback: $MODEL_REPO"
echo "  Initialization mode: 0 (single GPU)"
echo "  Attention masks: enabled"
echo "  Colab command timeout: ${EXEC_TIMEOUT}s"

# Upload only project source. Model weights stay at the supplied remote path.
tar -C "$SCRIPT_DIR" -czf "$SOURCE_ARCHIVE" inference.py prompt_utils.py point_visualization.py make_story_image.py run_colab_remote.py run_batch_inference.py run_colab_worker.py requirements-colab.txt models
tar -C "$PROMPTS_FOLDER" -czf "$PROMPTS_ARCHIVE" .

NEW_COMMAND=(colab new -s "$SESSION_NAME")
[[ -n "$GPU" ]] && NEW_COMMAND+=(--gpu "$GPU")
"${NEW_COMMAND[@]}"
SESSION_CREATED=true

colab_exec() {
  # google-colab-cli receives Python code on stdin; passing it as a positional
  # argument is rejected by current releases.
  printf '%s\n' "$1" | colab exec -s "$SESSION_NAME" --timeout "$EXEC_TIMEOUT"
}

# `colab new` can report READY a few seconds before the Jupyter kernel API is
# responsive. The CLI's kernel-start HTTP request has its own fixed 10-second
# read timeout, independent of --timeout, so warm it up with bounded retries.
wait_for_colab_kernel() {
  local attempt
  for attempt in 1 2 3 4 5 6; do
    echo "Checking Colab kernel readiness ($attempt/6)..."
    if colab_exec "from pathlib import Path; Path('$REMOTE_ROOT').mkdir(parents=True, exist_ok=True); print('Colab kernel is ready')"; then
      return 0
    fi
    if [[ "$attempt" -lt 6 ]]; then
      echo "Kernel endpoint is not ready yet; retrying in $((attempt * 5)) seconds..."
      sleep $((attempt * 5))
    fi
  done
  echo "Error: Colab kernel did not become ready after 6 attempts" >&2
  return 1
}

wait_for_colab_kernel

# A Drive-backed model path is not visible in a fresh VM until Drive is
# mounted. `colab drivemount` is an interactive browser authorization flow;
# its kernel WebSocket can occasionally disconnect after a successful VM
# allocation. Retry on a fresh CLI connection before giving up the session.
mount_google_drive() {
  local attempt mount_output
  for attempt in 1 2 3; do
    echo "Mounting Google Drive for model path: $MODEL_PATH ($attempt/3)..."
    echo "  If an authorization URL appears, open it, grant Drive access, then complete the CLI prompt."

    # Do not capture this interactive command in command substitution.  That
    # buffers the authorization URL and can detach its stdin, leaving the user
    # unable to authorize Drive until the command fails.  Let the CLI inherit
    # this terminal so its URL and any prompt are immediately usable.
    if colab drivemount -s "$SESSION_NAME"; then
      :
    else
      echo "Drive mount attempt $attempt failed (a transient Colab WebSocket disconnect is recoverable)." >&2
    fi

    # The CLI can exit zero even when the notebook cell reports an error, so
    # use an explicit kernel marker rather than the process exit status.
    mount_output="$(colab_exec "from pathlib import Path; print('CHARACONSIST_DRIVE_MOUNT_OK' if Path('/content/drive').is_dir() else 'CHARACONSIST_DRIVE_MOUNT_MISSING')" 2>&1 || true)"
    printf '%s\n' "$mount_output"
    if [[ "$mount_output" == *"CHARACONSIST_DRIVE_MOUNT_OK"* ]]; then
      return 0
    fi
    echo "Drive mount is not reachable after attempt $attempt." >&2

    if [[ "$attempt" -lt 3 ]]; then
      echo "Retrying Drive mount in $((attempt * 5)) seconds without releasing '$SESSION_NAME'..."
      sleep $((attempt * 5))
    fi
  done
  echo "Error: Google Drive could not be mounted after 3 attempts. The session will be cleaned up." >&2
  return 1
}

if [[ "$MODEL_PATH" == /content/drive/* ]]; then
  mount_google_drive
fi

colab upload -s "$SESSION_NAME" "$SOURCE_ARCHIVE" "$REMOTE_ROOT/source.tar.gz"
colab upload -s "$SESSION_NAME" "$PROMPTS_ARCHIVE" "$REMOTE_ROOT/prompts.tar.gz"

# Extract source before checking whether the remote model requires recovery.
colab_exec "import pathlib, subprocess; root=pathlib.Path('$REMOTE_ROOT'); subprocess.run(['tar', '-xzf', str(root/'source.tar.gz'), '-C', str(root)], check=True)"

# Do not upload a Hugging Face token for a complete Drive snapshot. The CLI can
# return zero for a failed cell, so select behavior from an explicit marker.
MODEL_STATUS_OUTPUT="$(colab_exec "import pathlib, sys; root=pathlib.Path('$REMOTE_ROOT'); sys.path.insert(0, str(root)); from run_colab_remote import model_snapshot_complete; ready=model_snapshot_complete(pathlib.Path('$MODEL_PATH')); print(f'CHARACONSIST_MODEL_READY={int(ready)}')" 2>&1 || true)"
printf '%s\n' "$MODEL_STATUS_OUTPUT"
if [[ "$MODEL_STATUS_OUTPUT" == *"CHARACONSIST_MODEL_READY=1"* ]]; then
  echo "Remote model snapshot is complete; Hugging Face fallback is not needed."
elif [[ "$MODEL_STATUS_OUTPUT" == *"CHARACONSIST_MODEL_READY=0"* ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Error: remote model snapshot is incomplete and HF_TOKEN is not configured." >&2
    exit 1
  fi
  echo "Remote model snapshot is incomplete; uploading a temporary Hugging Face credential for recovery."
  LOCAL_HF_TOKEN_FILE="$WORK_DIR/hf-token"
  (umask 077 && printf '%s' "$HF_TOKEN" > "$LOCAL_HF_TOKEN_FILE")
  colab upload -s "$SESSION_NAME" "$LOCAL_HF_TOKEN_FILE" "$REMOTE_ROOT/.hf_token"
else
  echo "Error: could not determine whether the remote model snapshot is complete." >&2
  exit 1
fi

encode_for_python() {
  printf '%s' "$1" | base64 | tr -d '\n'
}

download_prompt_archive() {
  local relative_output="$1"
  local remote_archive="$REMOTE_ROOT/result_archives/$relative_output.tar.gz"
  local local_archive="$WORK_DIR/result_archives/$relative_output.tar.gz"
  mkdir -p "$(dirname "$local_archive")" "$LOCAL_OUTPUT_DIR"
  PENDING_REMOTE_ARCHIVE="$remote_archive"
  PENDING_LOCAL_ARCHIVE="$local_archive"
  if colab download -s "$SESSION_NAME" "$remote_archive" "$local_archive"; then
    if tar -xzf "$local_archive" -C "$LOCAL_OUTPUT_DIR"; then
      RESULTS_DOWNLOADED=true
      PENDING_REMOTE_ARCHIVE=""
      PENDING_LOCAL_ARCHIVE=""
      echo "Finalized result downloaded to: $LOCAL_OUTPUT_DIR/bg_fg/$relative_output"
      return 0
    fi
    echo "Warning: could not extract finalized result archive" >&2
    return 1
  fi
  echo "Warning: could not download finalized result archive" >&2
  return 1
}

record_local_failure() {
  local relative_prompt="$1"
  local reason="$2"
  printf '%s\t%s\n' "$relative_prompt" "$reason" >> "$LOCAL_FAILURES_FILE"
}

merge_remote_summary() {
  local remote_summary="$WORK_DIR/remote-batch-summary.json"
  mkdir -p "$LOCAL_OUTPUT_DIR"
  if ! colab download -s "$SESSION_NAME" "$REMOTE_RESULTS/batch-summary.json" "$remote_summary"; then
    return 1
  fi
  PYTHONPATH="$SCRIPT_DIR" python3 - "$remote_summary" "$LOCAL_FAILURES_FILE" "$LOCAL_OUTPUT_DIR/batch-summary.json" <<'PY'
import json
import sys
from pathlib import Path

from run_batch_inference import merge_delivery_failures

remote_path, failures_path, destination = map(Path, sys.argv[1:])
delivery_failures = []
if failures_path.is_file():
    for line in failures_path.read_text(encoding="utf-8").splitlines():
        prompt_file, reason = line.split("\t", 1)
        delivery_failures.append({
            "prompt_file": prompt_file,
            "exit_code": 1,
            "source": "local_result_delivery",
            "error": reason,
        })

summary = json.loads(remote_path.read_text(encoding="utf-8"))
destination.write_text(
    json.dumps(merge_delivery_failures(summary, delivery_failures), indent=2) + "\n",
    encoding="utf-8",
)
PY
}

local_result_is_current() {
  PYTHONPATH="$SCRIPT_DIR" python3 - "$1" "$2" "$MODEL_PATH" <<'PY'
import sys
from pathlib import Path
from run_batch_inference import inference_settings, marker_matches, success_record
import argparse

prompt, output, model_path = map(Path, sys.argv[1:])
config = argparse.Namespace(model_path=str(model_path), init_mode=0, gpu_ids=[0], save_mask=True, save_points=True, height=1024, width=1024, seed=2025)
raise SystemExit(0 if marker_matches(output, success_record(prompt, inference_settings(config))) else 1)
PY
}

ROOT_B64="$(encode_for_python "$REMOTE_ROOT")"
MODEL_B64="$(encode_for_python "$MODEL_PATH")"
REPO_B64="$(encode_for_python "$MODEL_REPO")"
echo "Initializing one persistent Colab inference worker..."
colab_exec "import base64, sys; root=base64.b64decode('$ROOT_B64').decode(); sys.path.insert(0, root); import run_colab_worker as worker; print('CHARACONSIST_WORKER_RESULT='+__import__('json').dumps(worker.start(root, base64.b64decode('$MODEL_B64').decode(), base64.b64decode('$REPO_B64').decode(), 0)))"

BATCH_FAILURES=0
while IFS= read -r -d '' prompt_file; do
  relative_prompt="${prompt_file#"$PROMPTS_FOLDER"/}"
  relative_output="${relative_prompt%.txt}"
  local_output="$LOCAL_OUTPUT_DIR/bg_fg/$relative_output"
  relative_b64="$(encode_for_python "$relative_prompt")"

  if local_result_is_current "$prompt_file" "$local_output"; then
    echo "Skipping locally verified result: $relative_prompt"
    colab_exec "import base64, json, run_colab_worker as worker; print('CHARACONSIST_PROMPT_RESULT='+json.dumps(worker.record_local_skip(base64.b64decode('$relative_b64').decode())))"
    continue
  fi

  echo "Running prompt file: $relative_prompt"
  RESULTS_DOWNLOADED=false
  PENDING_REMOTE_ARCHIVE="$REMOTE_ROOT/result_archives/$relative_output.tar.gz"
  PENDING_LOCAL_ARCHIVE="$WORK_DIR/result_archives/$relative_output.tar.gz"
  if ! worker_output="$(colab_exec "import base64, json, run_colab_worker as worker; print('CHARACONSIST_PROMPT_RESULT='+json.dumps(worker.run_one(base64.b64decode('$relative_b64').decode())))" 2>&1)"; then
    printf '%s\n' "$worker_output" >&2
    echo "Error: Colab worker became unavailable; finalized earlier results will be recovered during cleanup." >&2
    exit 1
  fi
  printf '%s\n' "$worker_output"

  if [[ "$worker_output" == *'"status": "success"'* || "$worker_output" == *'"status": "skipped"'* ]]; then
    if ! download_prompt_archive "$relative_output"; then
      record_local_failure "$relative_prompt" "finalized result archive could not be downloaded or extracted"
      BATCH_FAILURES=1
      PENDING_REMOTE_ARCHIVE=""
      PENDING_LOCAL_ARCHIVE=""
      merge_remote_summary || true
      echo "Result delivery failed for '$relative_prompt'; continuing so it can be retried next run." >&2
    elif ! local_result_is_current "$prompt_file" "$local_output"; then
      record_local_failure "$relative_prompt" "downloaded result did not pass marker verification for the current settings"
      BATCH_FAILURES=1
      PENDING_REMOTE_ARCHIVE=""
      PENDING_LOCAL_ARCHIVE=""
      merge_remote_summary || true
      echo "Downloaded result for '$relative_prompt' is retained but marked bad; continuing so it can be retried next run." >&2
    else
      echo "Verified local result before continuing: $relative_prompt"
    fi
  else
    BATCH_FAILURES=1
    PENDING_REMOTE_ARCHIVE=""
    PENDING_LOCAL_ARCHIVE=""
    merge_remote_summary || true
    echo "Prompt file failed; continuing with the next file." >&2
  fi
done < <(find "$PROMPTS_FOLDER" -type f -name '*.txt' -print0 | sort -z)

finish_output="$(colab_exec "import json, run_colab_worker as worker; print('CHARACONSIST_WORKER_RESULT='+json.dumps(worker.finish()))" 2>&1 || true)"
printf '%s\n' "$finish_output"
RESULTS_DOWNLOADED=false
merge_remote_summary || exit 1

if [[ "$finish_output" != *'"failed": 0'* || "$BATCH_FAILURES" != 0 ]]; then
  echo "Error: one or more prompt files or local result deliveries failed; valid results were downloaded." >&2
  exit 1
fi
