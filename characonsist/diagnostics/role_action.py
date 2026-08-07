"""Artifacts proving that role-action routing executed and where it acted."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _tensor_map(value):
    array = value.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return np.clip(array, 0.0, 1.0)


def save_role_action_maps(image, role_maps, output_dir, frame):
    """Save raw maps and simple red heat-map overlays for one generated frame."""
    map_dir = Path(output_dir) / "role_action" / f"frame_{frame}"
    map_dir.mkdir(parents=True, exist_ok=True)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    for name, value in role_maps.items():
        role_map = _tensor_map(value)
        np.save(map_dir / f"{name}.npy", role_map)
        heat = Image.fromarray(np.uint8(role_map * 255)).resize(
            image.size, Image.Resampling.BILINEAR
        )
        heat_array = np.asarray(heat, dtype=np.float32) / 255.0
        colored = np.zeros_like(base)
        colored[..., 0] = 255.0 * heat_array
        overlay = np.uint8(np.clip(base * 0.65 + colored * 0.35, 0, 255))
        Image.fromarray(overlay).save(map_dir / f"{name}_overlay.jpg")


def save_role_action_trace(output_dir, seed, strength, frames):
    """Write a prompt-level execution trace for the controlled ablation."""
    payload = {
        "seed": int(seed),
        "role_action_bias_strength": float(strength),
        "action_gate_strength_required": 0.0,
        "frames": frames,
    }
    path = Path(output_dir) / "role_action_trace.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
