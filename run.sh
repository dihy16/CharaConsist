if [ -z "$1" ]; then
  echo "Usage: $0 <gpu_id>"
  exit 1
fi
CUDA_VISIBLE_DEVICES=$1 python inference.py \
--init_mode 0 \
--prompts_file prompts/long_story.txt \
--model_path ../model/flux-dev \
--out_dir results/bg_fg