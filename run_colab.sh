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
  --init-mode <0|1|2|3>  inference.py initialization mode (default: 1, CPU offload)
  --output-dir <path>    Local directory for downloaded results (default: results_colab)
  --keep                 Leave the Colab session running after completion or failure
  -h, --help             Show this help
EOF
}

PROMPTS_FOLDER=""
SESSION_NAME=""
GPU=""
MODEL_PATH=""
INIT_MODE=1
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
[[ -z "$GPU" || "$GPU" =~ ^(T4|L4|A100|H100)$ ]] || { echo "Error: unsupported GPU '$GPU'" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
echo "  Attention masks: enabled"

# Upload only project source. Model weights stay at the supplied remote path.
tar -C "$SCRIPT_DIR" -czf "$SOURCE_ARCHIVE" inference.py make_story_image.py requirements.txt models
tar -C "$PROMPTS_FOLDER" -czf "$PROMPTS_ARCHIVE" .

NEW_COMMAND=(colab new -s "$SESSION_NAME")
[[ -n "$GPU" ]] && NEW_COMMAND+=(--gpu "$GPU")
"${NEW_COMMAND[@]}"
SESSION_CREATED=true

# A new Colab VM has neither this project's pinned Python dependencies nor its
# source tree. The Colab CLI installs a local requirements file on the session.
colab install -s "$SESSION_NAME" -r "$SCRIPT_DIR/requirements.txt"
colab exec -s "$SESSION_NAME" "from pathlib import Path; Path('$REMOTE_ROOT').mkdir(parents=True, exist_ok=True)"
colab upload -s "$SESSION_NAME" "$SOURCE_ARCHIVE" "$REMOTE_ROOT/source.tar.gz"
colab upload -s "$SESSION_NAME" "$PROMPTS_ARCHIVE" "$REMOTE_ROOT/prompts.tar.gz"

# Extract archives and verify the externally supplied model before starting an
# expensive inference job. colab exec runs Python, not a shell command.
colab exec -s "$SESSION_NAME" "import pathlib, subprocess; root=pathlib.Path('$REMOTE_ROOT'); subprocess.run(['tar', '-xzf', str(root / 'source.tar.gz'), '-C', str(root)], check=True); (root / 'prompts_batch').mkdir(exist_ok=True); subprocess.run(['tar', '-xzf', str(root / 'prompts.tar.gz'), '-C', str(root / 'prompts_batch')], check=True); model=pathlib.Path('$MODEL_PATH'); assert model.is_dir(), f'Model path does not exist or is not a directory: {model}'"

# Run every prompt file in the uploaded folder, preserving nested folders in
# the results. --save_mask writes the foreground attention-mask overlays.
colab exec -s "$SESSION_NAME" "import pathlib, subprocess, sys; root=pathlib.Path('$REMOTE_ROOT'); prompts=root/'prompts_batch'; files=sorted(prompts.rglob('*.txt')); assert files, f'No .txt files found in {prompts}'; [subprocess.run([sys.executable, 'inference.py', '--init_mode', '$INIT_MODE', '--prompts_file', str(prompt), '--model_path', '$MODEL_PATH', '--out_dir', str(root/'results'/'bg_fg'/prompt.relative_to(prompts).with_suffix('')), '--save_mask'], cwd=root, check=True) for prompt in files]"

mkdir -p "$LOCAL_OUTPUT_DIR"
colab download -s "$SESSION_NAME" "$REMOTE_RESULTS" "$LOCAL_OUTPUT_DIR"
echo "Results downloaded to: $LOCAL_OUTPUT_DIR"
