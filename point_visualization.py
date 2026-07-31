"""Paper-aligned dense point-correspondence diagnostics."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def build_dense_correspondence(argmax_indices, max_sim, id_fg_mask, curr_fg_mask, similarity_threshold=0.5):
    """Convert current-to-ID token matches into serializable records and colors."""
    indices = np.asarray(argmax_indices).reshape(-1).astype(np.int64)
    similarities = np.asarray(max_sim).reshape(-1).astype(np.float32)
    id_mask = np.asarray(id_fg_mask).squeeze().astype(bool)
    curr_mask = np.asarray(curr_fg_mask).squeeze().astype(bool)
    if id_mask.ndim != 2 or curr_mask.ndim != 2:
        raise ValueError("Point masks must each resolve to a 2D feature grid.")
    if indices.size != curr_mask.size or similarities.size != curr_mask.size:
        raise ValueError("Correspondence tensors must contain one entry per current-image token.")
    if indices.size and (indices.min() < 0 or indices.max() >= id_mask.size):
        raise ValueError("A matched identity-token index is outside the identity feature grid.")

    curr_h, curr_w = curr_mask.shape
    id_h, id_w = id_mask.shape
    curr_y, curr_x = np.divmod(np.arange(indices.size), curr_w)
    id_y, id_x = np.divmod(indices, id_w)
    valid = curr_mask.reshape(-1) & id_mask.reshape(-1)[indices] & (similarities > similarity_threshold)

    denom_x = max(id_w - 1, 1)
    denom_y = max(id_h - 1, 1)
    colors = np.stack(
        [id_x / denom_x * 255.0, id_y / denom_y * 255.0, np.full(indices.size, 128.0)],
        axis=-1,
    )
    colors[~valid] *= 0.2
    color_map = colors.astype(np.uint8).reshape(curr_h, curr_w, 3)
    records = [
        {
            "current": [int(x), int(y)],
            "identity": [int(ix), int(iy)],
            "similarity": float(similarity),
            "current_foreground": bool(curr_fg),
            "identity_foreground": bool(id_mask.reshape(-1)[match_index]),
            "valid": bool(is_valid),
        }
        for x, y, ix, iy, similarity, curr_fg, match_index, is_valid in zip(
            curr_x,
            curr_y,
            id_x,
            id_y,
            similarities,
            curr_mask.reshape(-1),
            indices,
            valid,
        )
    ]
    metadata = {
        "schema_version": 1,
        "extraction_pass": "unmodified_pre_run",
        "sampling_step_index": 10,
        "sampling_timestep_ordinal": 11,
        "similarity_threshold": similarity_threshold,
        "current_grid": {"height": curr_h, "width": curr_w},
        "identity_grid": {"height": id_h, "width": id_w},
        "matches": records,
    }
    return color_map, metadata


def save_dense_correspondence(image, snapshot, image_path, json_path, similarity_threshold=0.5):
    color_map, metadata = build_dense_correspondence(
        snapshot["argmax_indices"],
        snapshot["max_sim"],
        snapshot["id_fg_mask"],
        snapshot["curr_fg_mask"],
        similarity_threshold,
    )
    rendered = Image.fromarray(color_map).resize(image.size, Image.Resampling.NEAREST)
    comparison = Image.new("RGB", (image.width * 2, image.height))
    comparison.paste(image.convert("RGB"), (0, 0))
    comparison.paste(rendered, (image.width, 0))
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    comparison.save(image_path)
    Path(json_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
