if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <gpu_id> <prompts_folder>"
  exit 1
fi

GPU_ID=$1
PROMPTS_FOLDER=$2

CUDA_VISIBLE_DEVICES=$GPU_ID python run_batch_inference.py \
  --root . \
  --prompts-dir "$PROMPTS_FOLDER" \
  --model-path ../model/flux-dev \
  --init-mode 0 \
  --results-dir results/bg_fg \
  --summary results/batch-summary.json
