import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument("--init_mode", type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument("--prompts_file", type=str, default="")
parser.add_argument("--model_path", type=str, default="/path/to/FLUX.1-dev")
parser.add_argument("--out_dir", type=str, default="results")
parser.add_argument("--use_interpolate", action='store_true')
parser.add_argument(
    "--action_gate_strength",
    type=float,
    default=1.0,
    help="Action-attention suppression for adaptive token merge (0 disables, 1 is full gating).",
)
parser.add_argument("--share_bg", action='store_true')
parser.add_argument("--save_mask", action='store_true')
parser.add_argument("--save_points", action='store_true')
parser.add_argument("--save_action_maps", action='store_true')
parser.add_argument("--save_all_steps", action='store_true')
parser.add_argument("--mix_mode", action='store_true')
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--seed", type=int, default=2025)
import os
import torch
import numpy as np

from models.attention_processor_characonsist import (
    reset_attn_processor,
    set_text_spans,
    reset_size,
    reset_id_bank,
)
from models.pipeline_characonsist import CharaConsistPipeline
from prompt_utils import build_prompt_and_spans
from point_visualization import save_dense_correspondence, tensor_to_numpy
from action_visualization import save_action_attention_artifacts


def configure_cuda(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def init_model_mode_0(config):
    pipe = CharaConsistPipeline.from_pretrained(config.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda:0")
    return pipe

def init_model_mode_1(config):
    pipe = CharaConsistPipeline.from_pretrained(config.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

def init_model_mode_2(config):
    from diffusers import FluxTransformer2DModel
    from transformers import T5EncoderModel
    transformer = FluxTransformer2DModel.from_pretrained(
        config.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="balanced")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        config.model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, device_map="balanced")
    pipe = CharaConsistPipeline.from_pretrained(
        config.model_path,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16, 
        device_map="balanced")
    return pipe

def init_model_mode_3(config):
    pipe = CharaConsistPipeline.from_pretrained(config.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return pipe


MODEL_INIT_FUNCS = {
    0: init_model_mode_0,
    1: init_model_mode_1,
    2: init_model_mode_2,
    3: init_model_mode_3
}

def modify_prompt_and_get_length(bg, fg, act, pipe):
    """Backward-compatible wrapper returning the new cumulative spans."""
    return build_prompt_and_spans(bg, fg, act, pipe)
            
def load_prompt_file(pipe, file_path):
    with open(file_path, "r") as f:
        all_lines = f.readlines()
    all_prompt_info = []
    curr_prompts, curr_bg_len, curr_action_start, curr_real_len = [], [], [], []
    for line_number, line in enumerate(all_lines, start=1):
        prompt = line.strip()
        if len(prompt) > 0:
            prompt_parts = prompt.split("#", 2)
            if len(prompt_parts) != 3:
                raise ValueError(
                    f"Invalid prompt at line {line_number}: expected background#foreground#action."
                )
            bg, fg, act = prompt_parts
            prompt, bg_len, action_start, real_len = modify_prompt_and_get_length(
                bg, fg, act, pipe
            )
            curr_prompts.append(prompt)
            curr_bg_len.append(bg_len)
            curr_action_start.append(action_start)
            curr_real_len.append(real_len)
        else:
            if curr_prompts:
                all_prompt_info.append(
                    (curr_prompts, curr_bg_len, curr_action_start, curr_real_len)
                )
            curr_prompts, curr_bg_len, curr_action_start, curr_real_len = [], [], [], []
    if len(curr_prompts) > 0:
        all_prompt_info.append(
            (curr_prompts, curr_bg_len, curr_action_start, curr_real_len)
        )
    return all_prompt_info

def load_mix_prompt_file(pipe, file_path):
    scenes = load_prompt_file(pipe, file_path)
    story_prompts, story_bg_lens = [], []
    story_action_starts, story_real_lens, story_meta_info = [], [], []
    for scene_ind, (prompts, bg_lens, action_starts, real_lens) in enumerate(scenes):
        for prompt_ind, prompt in enumerate(prompts):
            story_prompts.append(prompt)
            story_bg_lens.append(bg_lens[prompt_ind])
            story_action_starts.append(action_starts[prompt_ind])
            story_real_lens.append(real_lens[prompt_ind])
            story_meta_info.append(
                dict(
                    update_bg=(scene_ind > 0 and prompt_ind == 0),
                )
            )
    return (
        story_prompts,
        story_bg_lens,
        story_action_starts,
        story_real_lens,
        story_meta_info,
    )

from PIL import Image
from make_story_image import parse_prompt_scenes, save_story_visualization

def overlay_mask_on_image(image, mask, color, output_path):
    img_array = np.array(image).astype(np.float32) * 0.5
    mask_zero = np.zeros_like(img_array)

    mask_resized = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_resized.resize(image.size, Image.NEAREST)
    mask_resized = np.array(mask_resized)
    mask_resized = mask_resized[:, :, None]
    color = np.array(color, dtype=np.float32).reshape(1, 1, -1)
    mask_resized_color = mask_resized * color
    img_array = img_array + mask_resized_color * 0.5
    mask_zero = mask_zero + mask_resized_color
    out_img = np.concatenate([img_array, mask_zero], axis=1)
    out_img[out_img>255] = 255
    out_img = out_img.astype(np.uint8)
    Image.fromarray(out_img).save(output_path)

def visualize_argmax_indices(image, argmax_indices, output_path):
    img_array = np.array(image).astype(np.float32) * 0.5
    h, w = img_array.shape[:2]
    indices = argmax_indices.cpu().numpy().flatten()
    if len(indices) == 0:
        return
    feat_h, feat_w = h // 16, w // 16
    q_y = indices // feat_w
    q_x = indices % feat_w
    R = (q_x / feat_w) * 255.0
    G = (q_y / feat_h) * 255.0
    B = np.full_like(R, 128.0)
    color_map = np.stack([R, G, B], axis=-1).astype(np.uint8)
    color_map_2d = color_map.reshape(feat_h, feat_w, 3)
    mask_resized = Image.fromarray(color_map_2d).resize((w, h), Image.NEAREST)
    mask_resized = np.array(mask_resized).astype(np.float32)
    out_img = np.concatenate([img_array + mask_resized * 0.5, mask_resized], axis=1)
    out_img[out_img > 255] = 255
    out_img = out_img.astype(np.uint8)
    Image.fromarray(out_img).save(output_path)


def snapshot_point_tracking(spatial_kwargs):
    """Freeze the paper's unmodified pre-run correspondence before the final pass."""
    required = ("argmax_indices", "max_sim", "id_fg_mask", "curr_fg_mask")
    missing = [key for key in required if key not in spatial_kwargs]
    if missing:
        raise RuntimeError(f"Missing pre-run point-tracking tensors: {', '.join(missing)}")
    return {
        key: tensor_to_numpy(spatial_kwargs[key])
        for key in required
    }


def save_action_map(image, spatial_kwargs, output_dir, stem):
    """Export the normalized foreground-masked action scores from step 10."""
    if "action_scores" not in spatial_kwargs:
        raise RuntimeError("Missing action_scores required for action-map export.")
    save_action_attention_artifacts(
        image,
        spatial_kwargs["action_scores"],
        output_dir,
        stem,
    )

def run_prompt_file(pipe, args):
    pipe_kwargs = dict(
        height = args.height,
        width = args.width,
        use_interpolate = args.use_interpolate,
        action_gate_strength = args.action_gate_strength,
        share_bg = args.share_bg,
        save_all_steps = args.save_all_steps
    )

    if args.mix_mode:
        prompts, bg_lens, action_starts, real_lens, meta_info = load_mix_prompt_file(
            pipe, args.prompts_file
        )
        if len(prompts) == 0:
            raise ValueError("No prompts found in prompts_file.")

        os.makedirs(args.out_dir, exist_ok=True)
        if args.save_mask:
            mask_out_dir = os.path.join(args.out_dir, "mask")
            os.makedirs(mask_out_dir, exist_ok=True)

        id_prompt = prompts[0]
        frm_prompts = prompts[1:]

        print("#" * 50)
        print("Generating ID image ...")
        set_text_spans(pipe, bg_lens[0], action_starts[0], real_lens[0])
        id_images, id_spatial_kwargs = pipe(
            id_prompt, is_id=True, generator=torch.Generator("cpu").manual_seed(args.seed), **pipe_kwargs
        )
        id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
        id_images[0].save(f"{args.out_dir}/id.jpg")
        if getattr(args, "save_action_maps", False):
            save_action_map(
                id_images[0], id_spatial_kwargs, os.path.join(args.out_dir, "action_attention"), "id"
            )
        if args.save_mask:
            overlay_mask_on_image(
                id_images[0], id_fg_mask[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_mask.jpg"
            )
            if args.save_all_steps and "all_fg_masks" in id_spatial_kwargs:
                os.makedirs(f"{mask_out_dir}/id_all_steps", exist_ok=True)
                for step_i, mask_t in id_spatial_kwargs["all_fg_masks"].items():
                    overlay_mask_on_image(
                        id_images[0], mask_t[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_all_steps/id_mask_step_{step_i:02d}.jpg"
                    )

        spatial_kwargs = dict(id_fg_mask=id_fg_mask, id_bg_mask=~id_fg_mask)
        print("#" * 50)
        print("Generating frame images ...")
        for ind, prompt in enumerate(frm_prompts):
            curr_pipe_kwargs = dict(pipe_kwargs)
            curr_pipe_kwargs.update(meta_info[ind + 1])
            set_text_spans(
                pipe,
                bg_lens[ind + 1],
                action_starts[ind + 1],
                real_lens[ind + 1],
            )
            pre_images, spatial_kwargs = pipe(
                prompt,
                is_pre_run=True,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                spatial_kwargs=spatial_kwargs,
                **curr_pipe_kwargs,
            )
            point_snapshot = snapshot_point_tracking(spatial_kwargs) if args.save_points else None
            pre_images[0].save(f"{args.out_dir}/{ind}_pre.jpg")
            images, spatial_kwargs = pipe(
                prompt,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                spatial_kwargs=spatial_kwargs,
                **curr_pipe_kwargs,
            )
            images[0].save(f"{args.out_dir}/{ind}.jpg")
            if getattr(args, "save_action_maps", False):
                save_action_map(
                    images[0], spatial_kwargs, os.path.join(args.out_dir, "action_attention"), str(ind)
                )
            if point_snapshot is not None:
                points_out_dir = os.path.join(args.out_dir, "points")
                save_dense_correspondence(
                    images[0],
                    point_snapshot,
                    os.path.join(points_out_dir, f"{ind}_dense.jpg"),
                    os.path.join(points_out_dir, f"{ind}_dense.json"),
                )
            if args.save_mask:
                overlay_mask_on_image(
                    images[0],
                    spatial_kwargs["curr_fg_mask"][0].cpu().numpy(),
                    (255, 0, 0),
                    f"{mask_out_dir}/{ind}_mask.jpg",
                )
            if args.save_all_steps:
                os.makedirs(f"{mask_out_dir}/{ind}_all_steps", exist_ok=True)
                if "all_fg_masks" in spatial_kwargs:
                    for step_i, mask_t in spatial_kwargs["all_fg_masks"].items():
                        overlay_mask_on_image(
                            images[0], mask_t[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/{ind}_all_steps/mask_step_{step_i:02d}.jpg"
                        )
                if "all_argmax_indices" in spatial_kwargs:
                    for step_i, indices_t in spatial_kwargs["all_argmax_indices"].items():
                        visualize_argmax_indices(images[0], indices_t, f"{mask_out_dir}/{ind}_all_steps/match_step_{step_i:02d}.jpg")
        reset_id_bank(pipe)
        mix_prompt_parts = [part for scene in parse_prompt_scenes(args.prompts_file) for part in scene]
        save_story_visualization(args.out_dir, mix_prompt_parts)
    else:
        # Load prompts for standard batch mode
        all_prompt_info = load_prompt_file(pipe, args.prompts_file)
        all_prompt_scenes = parse_prompt_scenes(args.prompts_file)

        for prompt_ind, (prompts, bg_lens, action_starts, real_lens) in enumerate(all_prompt_info):
            out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}")
            os.makedirs(out_dir, exist_ok=True)
            if args.save_mask:
                mask_out_dir = os.path.join(args.out_dir, f"prompt_{prompt_ind}", "mask")
                os.makedirs(mask_out_dir, exist_ok=True)
            id_prompt = prompts[0]
            frm_prompts = prompts[1:]

            # ID Gen
            print("#" * 50)
            print("Generating ID image ...")
            set_text_spans(pipe, bg_lens[0], action_starts[0], real_lens[0])
            id_images, id_spatial_kwargs = pipe(
                id_prompt, is_id=True, generator = torch.Generator("cpu").manual_seed(args.seed), **pipe_kwargs)
            id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
            id_images[0].save(f"{out_dir}/id.jpg")
            if getattr(args, "save_action_maps", False):
                save_action_map(
                    id_images[0], id_spatial_kwargs, os.path.join(out_dir, "action_attention"), "id"
                )
            if args.save_mask:
                overlay_mask_on_image(id_images[0], id_fg_mask[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_mask.jpg")
                if args.save_all_steps and "all_fg_masks" in id_spatial_kwargs:
                    os.makedirs(f"{mask_out_dir}/id_all_steps", exist_ok=True)
                    for step_i, mask_t in id_spatial_kwargs["all_fg_masks"].items():
                        overlay_mask_on_image(id_images[0], mask_t[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/id_all_steps/id_mask_step_{step_i:02d}.jpg")

            # Frame Gen
            spatial_kwargs = dict(id_fg_mask = id_fg_mask, id_bg_mask = ~id_fg_mask)
            print("#" * 50)
            print("Generating frame images ...")
            for ind, prompt in enumerate(frm_prompts):    
                set_text_spans(
                    pipe,
                    bg_lens[1:][ind],
                    action_starts[1:][ind],
                    real_lens[1:][ind],
                )
                pre_images, spatial_kwargs = pipe(
                    prompt, is_pre_run=True, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs) 
                point_snapshot = snapshot_point_tracking(spatial_kwargs) if args.save_points else None
                pre_images[0].save(f"{out_dir}/{ind}_pre.jpg")       
                images, spatial_kwargs = pipe(
                    prompt, generator = torch.Generator("cpu").manual_seed(args.seed), spatial_kwargs=spatial_kwargs, **pipe_kwargs)
                images[0].save(f"{out_dir}/{ind}.jpg")
                if getattr(args, "save_action_maps", False):
                    save_action_map(
                        images[0], spatial_kwargs, os.path.join(out_dir, "action_attention"), str(ind)
                    )
                if point_snapshot is not None:
                    points_out_dir = os.path.join(out_dir, "points")
                    save_dense_correspondence(
                        images[0],
                        point_snapshot,
                        os.path.join(points_out_dir, f"{ind}_dense.jpg"),
                        os.path.join(points_out_dir, f"{ind}_dense.json"),
                    )
                if args.save_mask:
                    overlay_mask_on_image(images[0], spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/{ind}_mask.jpg")
                if args.save_all_steps:
                    os.makedirs(f"{mask_out_dir}/{ind}_all_steps", exist_ok=True)
                    if "all_fg_masks" in spatial_kwargs:
                        for step_i, mask_t in spatial_kwargs["all_fg_masks"].items():
                            overlay_mask_on_image(images[0], mask_t[0].cpu().numpy(), (255, 0, 0), f"{mask_out_dir}/{ind}_all_steps/mask_step_{step_i:02d}.jpg")
                    if "all_argmax_indices" in spatial_kwargs:
                        for step_i, indices_t in spatial_kwargs["all_argmax_indices"].items():
                            visualize_argmax_indices(images[0], indices_t, f"{mask_out_dir}/{ind}_all_steps/match_step_{step_i:02d}.jpg")
            reset_id_bank(pipe)
            if prompt_ind < len(all_prompt_scenes):
                save_story_visualization(out_dir, all_prompt_scenes[prompt_ind])


def initialize_pipeline(args):
    configure_cuda(args.gpu_ids)
    pipe = MODEL_INIT_FUNCS[args.init_mode](args)
    reset_attn_processor(pipe, size=(args.height // 16, args.width // 16))
    return pipe


def reset_runtime_state(pipe, args):
    """Discard per-file identity and attention state without reloading weights."""
    reset_id_bank(pipe)
    reset_attn_processor(pipe, size=(args.height // 16, args.width // 16))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parser.parse_args()
    pipe = initialize_pipeline(args)
    try:
        run_prompt_file(pipe, args)
    finally:
        reset_runtime_state(pipe, args)


if __name__ == "__main__":
    main()
