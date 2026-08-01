"""Interactive foreground-mask and point-correspondence visualizer.

Run either with ``--model_path`` to generate a fresh pair with FLUX, or with
``--run_dir`` to inspect the ``--save_points`` artifacts from inference.py.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


SIMILARITY_THRESHOLD = 0.5
DEFAULT_SIZE = 1024


@dataclass
class CorrespondenceSession:
    identity_image: Image.Image
    current_image: Image.Image
    identity_mask: np.ndarray
    current_mask: np.ndarray
    identity_x: np.ndarray
    identity_y: np.ndarray
    similarity: np.ndarray
    valid: np.ndarray
    threshold: float


def _as_grid(metadata, name):
    grid = metadata.get(name)
    if not isinstance(grid, dict):
        raise ValueError(f"Dense correspondence is missing {name} metadata.")
    height, width = grid.get("height"), grid.get("width")
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError(f"Dense correspondence has an invalid {name} grid.")
    return height, width


def load_dense_correspondence(path):
    """Load a schema-v1 dense correspondence JSON into click-ready grids."""
    path = Path(path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dense correspondence {path}: {exc}") from exc
    if metadata.get("schema_version") != 1:
        raise ValueError("Only dense correspondence schema_version 1 is supported.")

    current_h, current_w = _as_grid(metadata, "current_grid")
    identity_h, identity_w = _as_grid(metadata, "identity_grid")
    matches = metadata.get("matches")
    if not isinstance(matches, list) or len(matches) != current_h * current_w:
        raise ValueError("Dense correspondence must contain exactly one match per current token.")

    identity_x = np.zeros((current_h, current_w), dtype=np.int32)
    identity_y = np.zeros((current_h, current_w), dtype=np.int32)
    similarity = np.zeros((current_h, current_w), dtype=np.float32)
    current_mask = np.zeros((current_h, current_w), dtype=bool)
    identity_mask = np.zeros((identity_h, identity_w), dtype=bool)
    valid = np.zeros((current_h, current_w), dtype=bool)
    seen = np.zeros((current_h, current_w), dtype=bool)

    for record in matches:
        try:
            current_x, current_y = record["current"]
            matched_x, matched_y = record["identity"]
            score = float(record["similarity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Dense correspondence contains a malformed match record.") from exc
        if not (0 <= current_x < current_w and 0 <= current_y < current_h):
            raise ValueError("A current token coordinate is outside the current grid.")
        if not (0 <= matched_x < identity_w and 0 <= matched_y < identity_h):
            raise ValueError("An identity token coordinate is outside the identity grid.")
        if seen[current_y, current_x]:
            raise ValueError("Dense correspondence contains duplicate current token coordinates.")
        seen[current_y, current_x] = True
        identity_x[current_y, current_x] = matched_x
        identity_y[current_y, current_x] = matched_y
        similarity[current_y, current_x] = score
        current_mask[current_y, current_x] = bool(record.get("current_foreground", False))
        identity_mask[matched_y, matched_x] |= bool(record.get("identity_foreground", False))
        valid[current_y, current_x] = bool(record.get("valid", False))
    if not seen.all():
        raise ValueError("Dense correspondence is missing current token coordinates.")

    return {
        "identity_x": identity_x,
        "identity_y": identity_y,
        "similarity": similarity,
        "current_mask": current_mask,
        "identity_mask": identity_mask,
        "valid": valid,
        "threshold": float(metadata.get("similarity_threshold", SIMILARITY_THRESHOLD)),
    }


def discover_artifact_frames(run_dir):
    """Return numbered frames that have both a final image and dense JSON."""
    run_dir = Path(run_dir)
    if not (run_dir / "id.jpg").is_file():
        raise ValueError(f"{run_dir} does not contain id.jpg.")
    points_dir = run_dir / "points"
    frames = []
    if points_dir.is_dir():
        for json_path in points_dir.glob("*_dense.json"):
            frame_name = json_path.name.removesuffix("_dense.json")
            if frame_name.isdigit() and (run_dir / f"{frame_name}.jpg").is_file():
                frames.append(int(frame_name))
    if not frames:
        raise ValueError(f"{run_dir} has no numbered image plus points/*_dense.json pair.")
    return sorted(frames)


def _overlay_mask(image, mask, color=(255, 0, 0)):
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask_image, dtype=bool)[..., None]
    tint = np.asarray(color, dtype=np.float32)
    base = np.where(mask_array, base * 0.45 + tint * 0.55, base)
    return Image.fromarray(base.astype(np.uint8))


def _marker_position(grid_x, grid_y, grid_width, grid_height, image):
    return (
        round((grid_x + 0.5) * image.width / grid_width),
        round((grid_y + 0.5) * image.height / grid_height),
    )


def select_correspondence(session: Optional[CorrespondenceSession], x, y):
    """Render the most recent click and return paired images plus status text."""
    if session is None:
        return None, None, "Generate or load a frame before selecting a point."
    current_h, current_w = session.current_mask.shape
    identity_h, identity_w = session.identity_mask.shape
    current_x = min(max(int(x * current_w / session.current_image.width), 0), current_w - 1)
    current_y = min(max(int(y * current_h / session.current_image.height), 0), current_h - 1)
    matched_x = int(session.identity_x[current_y, current_x])
    matched_y = int(session.identity_y[current_y, current_x])
    score = float(session.similarity[current_y, current_x])
    is_valid = bool(session.valid[current_y, current_x])

    current = session.current_image.convert("RGB").copy()
    identity = session.identity_image.convert("RGB").copy()
    current_draw, identity_draw = ImageDraw.Draw(current), ImageDraw.Draw(identity)
    current_point = _marker_position(current_x, current_y, current_w, current_h, current)
    identity_point = _marker_position(matched_x, matched_y, identity_w, identity_h, identity)
    radius = max(5, min(current.width, current.height) // 60)
    color = (0, 220, 80) if is_valid else (255, 150, 0)
    for draw, point in ((current_draw, current_point), (identity_draw, identity_point)):
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), outline=color, width=4)
    status = f"Similarity: {score:.4f}. "
    status += "Valid foreground correspondence." if is_valid else (
        f"Not a valid foreground correspondence at threshold {session.threshold:g}."
    )
    return identity, current, status


def load_artifact_session(run_dir, frame):
    run_dir = Path(run_dir)
    dense = load_dense_correspondence(run_dir / "points" / f"{frame}_dense.json")
    identity = Image.open(run_dir / "id.jpg").convert("RGB")
    current = Image.open(run_dir / f"{frame}.jpg").convert("RGB")
    return CorrespondenceSession(
        identity, current, dense["identity_mask"], dense["current_mask"],
        dense["identity_x"], dense["identity_y"], dense["similarity"], dense["valid"], dense["threshold"],
    )


def _prompt_and_lengths(pipe, background, foreground, action):
    prompt = f"{background.strip()} {foreground.strip()} {action.strip()}".strip()
    if not all((background.strip(), foreground.strip(), action.strip())):
        raise ValueError("Background, foreground, and action must all be non-empty.")

    def token_length(text):
        mask = pipe.tokenizer_2(
            text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        ).attention_mask
        return max(0, int(mask.sum().item()) - 1)

    return prompt, token_length(background.strip() + " "), token_length(prompt)


def create_live_session(pipe, first_background, first_foreground, first_action,
                        second_background, second_foreground, second_action, seed):
    """Generate the documented pair and freeze its matching tensors for the UI."""
    import torch
    from attention_processor import set_text_len

    first_prompt, first_bg_len, first_real_len = _prompt_and_lengths(
        pipe, first_background, first_foreground, first_action
    )
    second_prompt, second_bg_len, second_real_len = _prompt_and_lengths(
        pipe, second_background, second_foreground, second_action
    )
    set_text_len(pipe, first_bg_len, first_real_len)
    identity_images, identity_data = pipe(
        first_prompt, is_id=True, generator=torch.Generator("cpu").manual_seed(int(seed))
    )
    set_text_len(pipe, second_bg_len, second_real_len)
    current_images, current_data = pipe(
        second_prompt, generator=torch.Generator("cpu").manual_seed(int(seed))
    )

    to_numpy = lambda tensor: tensor.detach().cpu().numpy()
    identity_mask = to_numpy(identity_data["curr_fg_mask"])[0].astype(bool)
    current_mask = to_numpy(current_data["curr_fg_mask"])[0].astype(bool)
    indices = to_numpy(current_data["argmax_indices"]).reshape(current_mask.shape)
    scores = to_numpy(current_data["max_sim"]).reshape(current_mask.shape).astype(np.float32)
    identity_h, identity_w = identity_mask.shape
    identity_y, identity_x = np.divmod(indices, identity_w)
    valid = current_mask & identity_mask[identity_y, identity_x] & (scores > SIMILARITY_THRESHOLD)
    return CorrespondenceSession(
        identity_images[0], current_images[0], identity_mask, current_mask,
        identity_x.astype(np.int32), identity_y.astype(np.int32), scores, valid, SIMILARITY_THRESHOLD,
    )


def load_live_pipeline(args):
    if args.height != DEFAULT_SIZE or args.width != DEFAULT_SIZE:
        raise ValueError("point_and_mask currently supports only 1024x1024 (its attention grid is 64x64).")
    if args.gpu_id is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from attention_processor import reset_attn_processor
    from pipeline import MaskPointPipeline

    if args.init_mode == "direct":
        if not torch.cuda.is_available():
            raise RuntimeError("Direct mode requires a CUDA-capable PyTorch installation.")
        pipe = MaskPointPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda:0")
    else:
        pipe = MaskPointPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        if args.init_mode == "cpu-offload":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
    reset_attn_processor(pipe)
    return pipe


def build_app(pipe=None, run_dir=None):
    """Create the Gradio application. Importing this module does not require Gradio."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the optional UI dependency with: pip install gradio==3.36.1") from exc

    artifact_frames = discover_artifact_frames(run_dir) if run_dir is not None else []
    example = [
        "in a modern gym with equipment in the background", "a muscular man in his 30s wearing a black tank top and shorts", "lifting dumbbells, focused expression",
        "in a modern gym with equipment in the background", "a muscular man in his 30s wearing a black tank top and shorts", "drinking from a bottle, side view", 2025,
    ]

    with gr.Blocks(title="CharaConsist Point Matching") as app:
        gr.Markdown("# CharaConsist mask and point matching\nClick **The 2nd image** to inspect its identity correspondence.")
        session_state = gr.State(value=None)
        with gr.Row():
            identity_image = gr.Image(label="The 1st image", type="pil")
            identity_mask = gr.Image(label="1st foreground mask", type="pil")
            current_image = gr.Image(label="The 2nd image (click here)", type="pil", interactive=True)
            current_mask = gr.Image(label="2nd foreground mask", type="pil")
        similarity = gr.Textbox(label="Similarity of clicked points pair", interactive=False)

        def present(session, status):
            return (
                session, session.identity_image, _overlay_mask(session.identity_image, session.identity_mask),
                session.current_image, _overlay_mask(session.current_image, session.current_mask), status,
            )

        if pipe is not None:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Prompt of the 1st image")
                    first_background = gr.Textbox(label="Background", value=example[0])
                    first_foreground = gr.Textbox(label="Foreground", value=example[1])
                    first_action = gr.Textbox(label="Foreground action", value=example[2])
                with gr.Column():
                    gr.Markdown("### Prompt of the 2nd image")
                    second_background = gr.Textbox(label="Background", value=example[3])
                    second_foreground = gr.Textbox(label="Foreground", value=example[4])
                    second_action = gr.Textbox(label="Foreground action", value=example[5])
            seed = gr.Number(label="Seed", value=example[6], precision=0)
            generate = gr.Button("Generate!", variant="primary")

            def generate_pair(*values):
                try:
                    session = create_live_session(pipe, *values)
                    return present(session, "Click the 2nd image to inspect a correspondence.")
                except (RuntimeError, ValueError) as exc:
                    raise gr.Error(str(exc))

            generate.click(
                generate_pair,
                [first_background, first_foreground, first_action, second_background, second_foreground, second_action, seed],
                [session_state, identity_image, identity_mask, current_image, current_mask, similarity],
            )
        else:
            gr.Markdown("### Saved run viewer\nCorrespondence was captured during the unmodified pre-run and is overlaid on the final saved frame.")
            frame_select = gr.Dropdown(choices=artifact_frames, value=artifact_frames[0], label="Frame")

            def load_frame(frame):
                try:
                    session = load_artifact_session(run_dir, int(frame))
                    return present(session, "Select a point in the 2nd image to inspect the saved correspondence.")
                except (RuntimeError, ValueError, OSError) as exc:
                    raise gr.Error(str(exc))

            frame_select.change(load_frame, [frame_select], [session_state, identity_image, identity_mask, current_image, current_mask, similarity])
            initial = load_artifact_session(run_dir, artifact_frames[0])
            app.load(lambda: present(initial, "Select a point in the 2nd image to inspect the saved correspondence."), outputs=[session_state, identity_image, identity_mask, current_image, current_mask, similarity])

        def handle_click(session, event: gr.SelectData):
            x, y = event.index
            return select_correspondence(session, x, y)

        current_image.select(handle_click, [session_state], [identity_image, current_image, similarity])
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model_path", help="Local FLUX.1-dev model directory for live generation.")
    mode.add_argument("--run_dir", help="Completed inference prompt directory to inspect offline.")
    parser.add_argument("--init_mode", choices=("direct", "cpu-offload", "sequential-offload"), default="direct")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir:
        app = build_app(run_dir=args.run_dir)
    else:
        pipe = load_live_pipeline(args)
        app = build_app(pipe=pipe)
    app.queue(concurrency_count=1).launch(server_name=args.server_name, server_port=args.port)


if __name__ == "__main__":
    # Direct execution places this file's directory on sys.path, which keeps the
    # standalone point_and_mask imports working as documented.
    main()
