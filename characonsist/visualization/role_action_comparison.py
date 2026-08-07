"""Render matched role-action control/intervention stories and diagnostic maps."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from characonsist.experiments.conditions import role_bias_label


def _fit(image, width):
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _finished_frame_paths(folder):
    paths = [folder / "id.jpg"]
    paths.extend(sorted(folder.glob("[0-9]*.jpg"), key=lambda path: int(path.stem)))
    return [path for path in paths if path.is_file() and not path.stem.endswith("_pre")]


def _output_audit(folders):
    baseline = {path.name: path for path in _finished_frame_paths(folders[0])}
    intervention = {path.name: path for path in _finished_frame_paths(folders[1])}
    records = []
    for name in sorted(set(baseline) & set(intervention), key=lambda value: (value != "id.jpg", value)):
        before = np.asarray(Image.open(baseline[name]).convert("RGB"), dtype=np.float32)
        after = np.asarray(Image.open(intervention[name]).convert("RGB"), dtype=np.float32)
        if before.shape != after.shape:
            records.append({"frame": name, "shape_match": False})
            continue
        delta = np.abs(after - before)
        records.append({
            "frame": name,
            "shape_match": True,
            "pixel_mae": float(delta.mean()),
            "changed_pixel_fraction": float(np.any(delta > 0, axis=-1).mean()),
        })
    return records


def _role_map_audit(folders):
    records = []
    baseline_root = folders[0] / "role_action"
    intervention_root = folders[1] / "role_action"
    for before_path in sorted(baseline_root.glob("frame_*/*.npy")):
        relative = before_path.relative_to(baseline_root)
        after_path = intervention_root / relative
        if not after_path.is_file():
            records.append({"map": str(relative), "available_in_both": False})
            continue
        before = np.load(before_path)
        after = np.load(after_path)
        records.append({
            "map": str(relative),
            "available_in_both": True,
            "shape_match": before.shape == after.shape,
            "mae": float(np.abs(before - after).mean()) if before.shape == after.shape else None,
        })
    return records


def render_role_action_comparison(results_root, prompt_stem, seed=2025, output=None):
    root = Path(results_root)
    labels = (role_bias_label(0.0), role_bias_label(1.0))
    folders = [
        root / "role_action_ablation" / label / f"seed_{seed}" / "bg_fg" / prompt_stem / "prompt_0"
        for label in labels
    ]
    story_paths = [folder / "story.jpg" for folder in folders]
    missing = [str(path) for path in story_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing matched story result(s): " + ", ".join(missing))

    width = 1600
    stories = [_fit(Image.open(path).convert("RGB"), width) for path in story_paths]
    header = 54
    canvas = Image.new("RGB", (width, sum(item.height for item in stories) + 2 * header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y = 0
    for label, story in zip(labels, stories):
        draw.text((16, y + 14), f"{label} | seed {seed}", fill="black", font=font)
        y += header
        canvas.paste(story, (0, y))
        y += story.height

    destination = Path(output) if output else root / "comparisons" / prompt_stem / "role_action_story.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95)
    audit = {
        "prompt": prompt_stem,
        "seed": seed,
        "conditions": list(labels),
        "outputs": _output_audit(folders),
        "pre_run_role_maps": _role_map_audit(folders),
        "interpretation": (
            "Role-map MAE should be zero because both conditions use the same unbiased pre-run; "
            "output MAE measures downstream sensitivity, not semantic improvement."
        ),
    }
    destination.with_name("role_action_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results_colab")
    parser.add_argument("--prompt", default="2b_transfer_roles")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output")
    args = parser.parse_args()
    path = render_role_action_comparison(
        args.results_root, Path(args.prompt).stem, args.seed, args.output
    )
    print(path)


if __name__ == "__main__":
    main()
