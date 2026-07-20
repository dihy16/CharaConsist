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
#     --model-path /content/drive/MyDrive/models/flux-dev --gpu L4

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_colab.sh <prompts_folder> [session_name] --model-path <remote_path> [options]

Options:
  --session <name>       Session name (alternative to the optional positional name)
  --gpu <T4|L4|A100|H100>
  --model-path <path>    Required path to FLUX.1-dev on the Colab VM
  --model-repo <repo>    Hugging Face fallback (default: black-forest-labs/FLUX.1-dev)
  --init-mode <0|1|2|3>  inference.py initialization mode (default: 1, CPU offload)
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
INIT_MODE=1
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
    --init-mode)
      [[ $# -ge 2 ]] || { echo "Error: --init-mode needs a value" >&2; exit 2; }
      INIT_MODE="$2"
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
[[ "$INIT_MODE" =~ ^[0-3]$ ]] || { echo "Error: --init-mode must be 0, 1, 2, or 3" >&2; exit 2; }
[[ "$EXEC_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Error: --timeout must be a positive number of seconds" >&2; exit 2; }
[[ -z "$GPU" || "$GPU" =~ ^(T4|L4|A100|H100)$ ]] || { echo "Error: unsupported GPU '$GPU'" >&2; exit 2; }

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
SESSION_CREATED=false

cleanup() {
  rm -rf "$WORK_DIR"
  if [[ "$SESSION_CREATED" == true && "$KEEP_SESSION" != true ]]; then
    echo "Stopping Colab session '$SESSION_NAME'..."
    colab stop -s "$SESSION_NAME" || echo "Warning: could not stop '$SESSION_NAME'; run: colab stop -s $SESSION_NAME" >&2
  elif [[ "$SESSION_CREATED" == true ]]; then
    echo "Session '$SESSION_NAME' is still running (stop it with: colab stop -s $SESSION_NAME)"
  fi
}
trap cleanup EXIT

echo "CharaConsist batch inference on Colab CLI"
echo "  Session: $SESSION_NAME"
echo "  Prompts: $PROMPT_COUNT file(s) from $PROMPTS_FOLDER"
echo "  Remote model: $MODEL_PATH"
echo "  Model download fallback: $MODEL_REPO"
echo "  Attention masks: enabled"
echo "  Colab command timeout: ${EXEC_TIMEOUT}s"

# Upload only project source. Model weights stay at the supplied remote path.
tar -C "$SCRIPT_DIR" -czf "$SOURCE_ARCHIVE" inference.py make_story_image.py run_colab_remote.py requirements.txt models
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
# mounted. colab drivemount handles the required browser authorization flow.
if [[ "$MODEL_PATH" == /content/drive/* ]]; then
  echo "Mounting Google Drive for model path: $MODEL_PATH"
  colab drivemount -s "$SESSION_NAME"
fi

colab upload -s "$SESSION_NAME" "$SOURCE_ARCHIVE" "$REMOTE_ROOT/source.tar.gz"
colab upload -s "$SESSION_NAME" "$PROMPTS_ARCHIVE" "$REMOTE_ROOT/prompts.tar.gz"

# Transfer the Hugging Face credential as a temporary file so it is not
# embedded in Colab command history. The remote runner deletes it immediately.
if [[ -n "${HF_TOKEN:-}" ]]; then
  LOCAL_HF_TOKEN_FILE="$WORK_DIR/hf-token"
  (umask 077 && printf '%s' "$HF_TOKEN" > "$LOCAL_HF_TOKEN_FILE")
  colab upload -s "$SESSION_NAME" "$LOCAL_HF_TOKEN_FILE" "$REMOTE_ROOT/.hf_token"
fi

# Extract and launch the maintainable remote runner in one kernel execution.
# Stream the child output explicitly: Jupyter does not consistently forward
# inherited subprocess streams. Record its exit code because some colab-cli
# versions return zero even when the executed cell raises. `deque` consumes
# the line generator without accumulating the full model-download log.
colab_exec "import pathlib, subprocess, sys; from collections import deque; root=pathlib.Path('$REMOTE_ROOT'); subprocess.run(['tar', '-xzf', str(root/'source.tar.gz'), '-C', str(root)], check=True); prompts=root/'prompts_batch'; prompts.mkdir(exist_ok=True); subprocess.run(['tar', '-xzf', str(root/'prompts.tar.gz'), '-C', str(prompts)], check=True); process=subprocess.Popen([sys.executable, str(root/'run_colab_remote.py'), '--root', str(root), '--prompts-dir', str(prompts), '--model-path', '$MODEL_PATH', '--model-repo', '$MODEL_REPO', '--init-mode', '$INIT_MODE'], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1); deque((print(line, end='', flush=True) for line in process.stdout), maxlen=0); exit_code=process.wait(); (root/'run-exit-code.txt').write_text(str(exit_code), encoding='utf-8')"

LOCAL_EXIT_CODE_FILE="$WORK_DIR/run-exit-code.txt"
colab download -s "$SESSION_NAME" "$REMOTE_ROOT/run-exit-code.txt" "$LOCAL_EXIT_CODE_FILE"
REMOTE_EXIT_CODE="$(tr -d '[:space:]' < "$LOCAL_EXIT_CODE_FILE")"
if [[ "$REMOTE_EXIT_CODE" != 0 ]]; then
  echo "Error: remote setup or inference failed with exit code $REMOTE_EXIT_CODE" >&2
  exit "$REMOTE_EXIT_CODE"
fi

mkdir -p "$LOCAL_OUTPUT_DIR"
colab download -s "$SESSION_NAME" "$REMOTE_RESULTS" "$LOCAL_OUTPUT_DIR"
echo "Results downloaded to: $LOCAL_OUTPUT_DIR"
