"""Compare seed-matched prompt-only, attention-only, and full CharaConsist runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from characonsist.story import parse_prompt_scenes
from characonsist.experiments.conditions import CONSISTENCY_MODES, parse_consistency_modes
from characonsist.visualization.lambda_comparison import _frame_image_path, _render_grid


def find_component_result(results_root, mode, seed, prompt_stem, prompt_index=0):
    candidate = (
        Path(results_root) / "component_ablation" / mode / f"seed_{seed}"
        / "bg_fg" / prompt_stem / f"prompt_{prompt_index}"
    )
    return candidate if candidate.is_dir() else None


def create_component_comparison(
    results_root,
    prompt_file,
    output_path,
    seed=2025,
    modes=CONSISTENCY_MODES,
    prompt_index=0,
    panel_size=256,
):
    """Render mode rows and save an auditable prompt-only-referenced diff summary."""
    prompt_file = Path(prompt_file)
    scenes = parse_prompt_scenes(prompt_file)
    if not scenes or not 0 <= prompt_index < len(scenes):
        raise ValueError(f"No prompt scene {prompt_index} found in {prompt_file}")
    parsed_modes = parse_consistency_modes(modes)
    rows = [
        (
            mode,
            find_component_result(results_root, mode, seed, prompt_file.stem, prompt_index),
        )
        for mode in parsed_modes
    ]

    def provider(_mode, result, column):
        if result is None:
            return None, ""
        path = _frame_image_path(result, column)
        return (Image.open(path).convert("RGB"), "") if path.is_file() else (None, "")

    output_path = Path(output_path)
    _render_grid(
        prompt_file,
        scenes[prompt_index],
        rows,
        f"component ablation - matched seed {seed}",
        output_path,
        provider,
        panel_size,
        row_label=lambda mode: mode.replace("_", " "),
    )

    available = [(mode, result) for mode, result in rows if result is not None]
    reference_mode, reference = next(
        ((mode, result) for mode, result in available if mode == "prompt_only"),
        available[0] if available else (None, None),
    )
    audit = {
        "schema_version": 1,
        "prompt": prompt_file.stem,
        "seed": int(seed),
        "difference_reference": reference_mode,
        "conditions": {},
    }
    for mode, result in rows:
        condition = {"result": "available" if result is not None else "missing", "frames": []}
        trace_path = result / "component_trace.json" if result is not None else None
        if trace_path is not None and trace_path.is_file():
            condition["component_trace"] = json.loads(trace_path.read_text(encoding="utf-8"))
        for column in range(1, len(scenes[prompt_index])):
            metrics = {"frame": column - 1, "status": "UNAVAILABLE"}
            if result is not None and reference is not None:
                current_path = _frame_image_path(result, column)
                reference_path = _frame_image_path(reference, column)
                if current_path.is_file() and reference_path.is_file():
                    current = np.asarray(Image.open(current_path).convert("RGB"), dtype=np.int16)
                    baseline = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.int16)
                    difference = np.abs(current - baseline)
                    metrics = {
                        "frame": column - 1,
                        "status": "CHANGED" if np.any(difference) else "UNCHANGED",
                        "mae": float(difference.mean()),
                        "changed_pixels_percent": float(np.any(difference, axis=2).mean() * 100),
                    }
            condition["frames"].append(metrics)
        audit["conditions"][mode] = condition

    audit_path = output_path.with_name("comparison_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return output_path, audit_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results_colab"))
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--modes", default=",".join(CONSISTENCY_MODES))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seed < 0 or args.panel_size < 64:
        parser.error("seed must be non-negative and panel size must be at least 64")
    if not args.prompt_file.is_file() or not args.results_root.is_dir():
        parser.error("prompt file and results root must exist")
    output = args.output or (
        args.results_root / "component_comparisons" / args.prompt_file.stem
        / f"seed_{args.seed}" / "outputs.jpg"
    )
    try:
        image_path, audit_path = create_component_comparison(
            args.results_root,
            args.prompt_file,
            output,
            args.seed,
            args.modes,
            args.prompt_index,
            args.panel_size,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved component comparison: {image_path}")
    print(f"Saved comparison audit: {audit_path}")


if __name__ == "__main__":
    main()
