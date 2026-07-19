#!/bin/bash
# CharaConsist Batch Inference Runner using Colab CLI
# Usage: ./run_colab.sh <prompts_folder> [session_name] [--gpu T4|L4|A100|H100] [--keep]
#
# Example:
#   ./run_colab.sh prompts/stress_test my-session --gpu L4
#   ./run_colab.sh prompts/stress_test my-session --gpu L4 --keep

set -e

# Parse arguments
if [ -z "$1" ]; then
  echo "Usage: $0 <prompts_folder> [session_name] [--gpu T4|L4|A100|H100] [--keep]"
  exit 1
fi

PROMPTS_FOLDER="$1"
SESSION_NAME="${2:-characonsist-$(date +%s)}"
GPU_FLAG=""
KEEP_SESSION=false

# Parse optional flags
shift
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU_FLAG="--gpu $2"
      shift 2
      ;;
    --keep)
      KEEP_SESSION=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

echo "=========================================="
echo "CharaConsist Batch Inference on Colab CLI"
echo "=========================================="
echo "Session: $SESSION_NAME"
echo "Prompts folder: $PROMPTS_FOLDER"
[ -n "$GPU_FLAG" ] && echo "GPU: $GPU_FLAG"
echo "Keep session: $KEEP_SESSION"
echo ""

# Validate prompts folder
if [ ! -d "$PROMPTS_FOLDER" ]; then
  echo "Error: Prompts folder '$PROMPTS_FOLDER' not found"
  exit 1
fi

# Count prompt files
PROMPT_COUNT=$(find "$PROMPTS_FOLDER" -maxdepth 1 -name "*.txt" | wc -l)
if [ "$PROMPT_COUNT" -eq 0 ]; then
  echo "Error: No .txt files found in $PROMPTS_FOLDER"
  exit 1
fi
echo "Found $PROMPT_COUNT prompt file(s)"
echo ""

# Step 1: Create Colab session
echo "[1/4] Creating Colab session '$SESSION_NAME'..."
colab new -s "$SESSION_NAME" $GPU_FLAG
echo "✓ Session created"
echo ""

# Step 2: Upload prompts and code to the session
echo "[2/4] Uploading files to session..."
# Create a remote prompts directory
colab exec -s "$SESSION_NAME" "import os; os.makedirs('/root/CharaConsist/prompts_batch', exist_ok=True)"

# Upload each prompt file
for prompts_file in "$PROMPTS_FOLDER"/*.txt; do
  if [ -f "$prompts_file" ]; then
    basename=$(basename "$prompts_file")
    echo "  Uploading: $basename"
    colab upload -s "$SESSION_NAME" "$prompts_file" "/root/CharaConsist/prompts_batch/$basename"
  fi
done
echo "✓ Files uploaded"
echo ""

# Step 3: Run inference for each prompt file
echo "[3/4] Running inference..."
COUNTER=0
for prompts_file in "$PROMPTS_FOLDER"/*.txt; do
  if [ -f "$prompts_file" ]; then
    ((COUNTER++))
    basename=$(basename "$prompts_file" .txt)
    echo "  [$COUNTER/$PROMPT_COUNT] Processing: $basename"
    
    # Execute inference on the remote session
    colab exec -s "$SESSION_NAME" <<EOF
import subprocess
import sys

prompt_file = "/root/CharaConsist/prompts_batch/$(basename "$prompts_file")"
output_dir = f"/root/CharaConsist/results/bg_fg/$basename"

try:
    subprocess.run([
        sys.executable, "inference.py",
        "--init_mode", "0",
        "--prompts_file", prompt_file,
        "--model_path", "../model/flux-dev",
        "--out_dir", output_dir
    ], cwd="/root/CharaConsist", check=True)
    print(f"✓ Completed: $basename")
except subprocess.CalledProcessError as e:
    print(f"✗ Failed: $basename", file=sys.stderr)
    sys.exit(1)
EOF
  fi
done
echo "✓ All inference jobs completed"
echo ""

# Step 4: Download results
echo "[4/4] Downloading results..."
colab download -s "$SESSION_NAME" "/root/CharaConsist/results" "./results_colab"
echo "✓ Results downloaded to ./results_colab"
echo ""

# Step 5: Cleanup
if [ "$KEEP_SESSION" = true ]; then
  echo "✓ Session '$SESSION_NAME' is still running (use 'colab stop -s $SESSION_NAME' to stop)"
else
  echo "Cleaning up: stopping session '$SESSION_NAME'..."
  colab stop -s "$SESSION_NAME"
  echo "✓ Session stopped"
fi

echo ""
echo "=========================================="
echo "✓ Batch inference completed successfully!"
echo "=========================================="
