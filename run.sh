if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <gpu_id> <prompts_folder>"
  exit 1
fi

GPU_ID=$1
PROMPTS_FOLDER=$2

for prompts_file in "$PROMPTS_FOLDER"/*.txt; do
  # Skip if no .txt files found
  [ -e "$prompts_file" ] || { echo "No .txt files found in $PROMPTS_FOLDER"; exit 1; }

  echo "Processing: $prompts_file"
  basename=$(basename "$prompts_file" .txt)
  CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --init_mode 0 \
    --prompts_file "$prompts_file" \
    --model_path ../model/flux-dev \
    --out_dir "results/bg_fg/$basename"

  echo "Done: $prompts_file"
done

echo "All files processed."