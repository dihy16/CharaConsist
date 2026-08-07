"""Artifacts and mechanical checks for character-conditioned action binding."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from models.action_binding import validate_character_maps
from models.action_binding import normalize_character_map


def _heatmap(values, size):
    values = values.detach().float().cpu().numpy()[0]
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(values * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((1.0 - np.abs(values - 0.5) * 2.0) * 180, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def save_action_binding_preflight(image, character_maps, foreground_mask, output_dir):
    """Save frozen C1/C2 maps, overlays, tensors, and automated sanity report."""
    output_dir = Path(output_dir) / "action_binding"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = validate_character_maps(character_maps, foreground_mask)
    arrays = {f"character_{key}": value.detach().float().cpu().numpy() for key, value in character_maps.items()}
    arrays["foreground_mask"] = foreground_mask.detach().cpu().numpy()
    np.savez_compressed(output_dir / "frozen_character_maps.npz", **arrays)
    for entity_id, values in character_maps.items():
        heat = _heatmap(values, image.size)
        heat.save(output_dir / f"character_{entity_id}_heatmap.png")
        Image.blend(image.convert("RGB"), heat, 0.45).save(
            output_dir / f"character_{entity_id}_overlay.png"
        )
    (output_dir / "map_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def save_action_binding_trace(output_dir, seed, beta, gamma, frames):
    output_dir = Path(output_dir) / "action_binding"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "beta": float(beta),
        "gamma": float(gamma),
        "frames": frames,
    }
    (output_dir / "action_binding_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def save_action_binding_action_maps(image, action_maps, foreground_mask, output_dir, frame):
    """Save the observed A1/A2 attention maps for manual localization checks."""
    output_dir = Path(output_dir) / "action_binding"
    output_dir.mkdir(parents=True, exist_ok=True)
    for entity_id, raw in action_maps.items():
        values = normalize_character_map(raw, foreground_mask)
        heat = _heatmap(values, image.size)
        heat.save(output_dir / f"frame_{frame}_action_{entity_id}_heatmap.png")
        Image.blend(image.convert("RGB"), heat, 0.45).save(
            output_dir / f"frame_{frame}_action_{entity_id}_overlay.png"
        )
