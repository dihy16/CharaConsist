"""Save normalized action-attention scores as reusable visual artifacts."""

from pathlib import Path

import numpy as np
from PIL import Image


def action_score_array(action_scores):
    """Return one finite, normalized 2-D action-score array."""
    if hasattr(action_scores, "detach"):
        action_scores = action_scores.detach().cpu().numpy()
    scores = np.asarray(action_scores, dtype=np.float32)
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 2:
        raise ValueError(
            "action_scores must have shape (height, width) or (1, height, width), "
            f"got {scores.shape}."
        )
    if not np.isfinite(scores).all():
        raise ValueError("action_scores must contain only finite values.")
    tolerance = 1e-6
    if scores.min() < -tolerance or scores.max() > 1.0 + tolerance:
        raise ValueError("action_scores must be normalized to the range [0, 1].")
    return np.clip(scores, 0.0, 1.0)


def colorize_action_scores(scores):
    """Convert normalized scores to an RGB blue-to-red heatmap."""
    scores = action_score_array(scores)
    red = np.clip(1.5 - np.abs(4.0 * scores - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * scores - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * scores - 1.0), 0.0, 1.0)
    return np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


def save_action_attention_artifacts(image, action_scores, output_dir, stem):
    """Save raw scores, a full-size heatmap, and an image overlay."""
    scores = action_score_array(action_scores)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{stem}_scores.npy", scores)
    heatmap = Image.fromarray(colorize_action_scores(scores), mode="RGB")
    heatmap = heatmap.resize(image.size, Image.Resampling.BILINEAR)
    heatmap.save(output_dir / f"{stem}_heatmap.png")

    base = image.convert("RGB")
    overlay = Image.blend(base, heatmap, alpha=0.45)
    overlay.save(output_dir / f"{stem}_overlay.jpg", quality=95)

