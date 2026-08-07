"""Compare seed-matched CharaConsist component-ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from characonsist.diagnostics.action_attention import colorize_action_scores
from characonsist.experiments.conditions import CONSISTENCY_MODES, parse_consistency_modes
from characonsist.story import parse_prompt_scenes
from characonsist.visualization.lambda_comparison import (
    _crop_diagnostic_half,
    _frame_image_path,
    _load_npz,
    _map_panel,
    _placeholder,
    _render_grid,
)


DEFAULT_COMPONENT_VIEWS = (
    "outputs",
    "incremental_differences",
    "masks",
    "action_attention",
    "points",
    "merge_effective_mean",
    "component_audit",
)

INCREMENTAL_REFERENCE = {
    "prompt_only": None,
    "attention_only": "prompt_only",
    "full": "attention_only",
}


def find_component_result(results_root, mode, seed, prompt_stem, prompt_index=0):
    candidate = (
        Path(results_root) / "component_ablation" / mode / f"seed_{seed}"
        / "bg_fg" / prompt_stem / f"prompt_{prompt_index}"
    )
    return candidate if candidate.is_dir() else None


def _component_rows(results_root, modes, seed, prompt_stem, prompt_index):
    return [
        (
            mode,
            find_component_result(results_root, mode, seed, prompt_stem, prompt_index),
        )
        for mode in parse_consistency_modes(modes)
    ]


def _diagnostic_path(output_path, view, suffix=".jpg"):
    output_path = Path(output_path)
    if output_path.stem == "outputs":
        return output_path.parent / f"{view}{suffix}"
    return Path(f"{output_path.with_suffix('')}_{view}{suffix}")


def _load_trace(result):
    path = result / "component_trace.json" if result is not None else None
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _image_metrics(result, reference, column):
    metrics = {"frame": column - 1, "status": "UNAVAILABLE"}
    if result is None or reference is None:
        return metrics
    current_path = _frame_image_path(result, column)
    reference_path = _frame_image_path(reference, column)
    if not current_path.is_file() or not reference_path.is_file():
        return metrics
    current = np.asarray(Image.open(current_path).convert("RGB"), dtype=np.int16)
    baseline = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.int16)
    difference = np.abs(current - baseline)
    return {
        "frame": column - 1,
        "status": "CHANGED" if np.any(difference) else "UNCHANGED",
        "mae": float(difference.mean()),
        "changed_pixels_percent": float(np.any(difference, axis=2).mean() * 100),
    }


def _artifact_availability(result, frame_count):
    if result is None:
        return {
            "outputs": 0,
            "masks": 0,
            "action_attention": 0,
            "points": 0,
            "merge_diagnostics": 0,
        }
    stems = ["id", *(str(index) for index in range(frame_count - 1))]
    return {
        "outputs": sum(_frame_image_path(result, index).is_file() for index in range(frame_count)),
        "masks": sum((result / "mask" / f"{stem}_mask.jpg").is_file() for stem in stems),
        "action_attention": sum(
            (result / "action_attention" / f"{stem}_overlay.jpg").is_file()
            for stem in stems
        ),
        "points": sum(
            (result / "points" / f"{index}_dense.jpg").is_file()
            for index in range(frame_count - 1)
        ),
        "merge_diagnostics": sum(
            (result / "merge_diagnostics" / f"{index}_weights.npz").is_file()
            for index in range(frame_count - 1)
        ),
    }


def _write_component_audit(rows, prompt_file, seed, frame_count, output_path):
    row_lookup = dict(rows)
    audit = {
        "schema_version": 2,
        "prompt": Path(prompt_file).stem,
        "seed": int(seed),
        "comparison_strategy": "incremental_predecessor",
        "comparison_pairs": {
            mode: reference for mode, reference in INCREMENTAL_REFERENCE.items()
            if mode in row_lookup
        },
        "conditions": {},
    }
    for mode, result in rows:
        reference_mode = INCREMENTAL_REFERENCE.get(mode)
        reference = row_lookup.get(reference_mode) if reference_mode else result
        condition = {
            "result": "available" if result is not None else "missing",
            "reference_mode": reference_mode,
            "artifact_counts": _artifact_availability(result, frame_count),
            "frames": [],
        }
        trace = _load_trace(result)
        if trace is not None:
            condition["component_trace"] = trace
        for column in range(1, frame_count):
            condition["frames"].append(_image_metrics(result, reference, column))
        audit["conditions"][mode] = condition

    audit_path = _diagnostic_path(output_path, "comparison_audit", ".json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit_path


def create_component_comparison(
    results_root,
    prompt_file,
    output_path,
    seed=2025,
    modes=CONSISTENCY_MODES,
    prompt_index=0,
    panel_size=256,
):
    """Render generated outputs and write the incremental component audit."""
    prompt_file = Path(prompt_file)
    scenes = parse_prompt_scenes(prompt_file)
    if not scenes or not 0 <= prompt_index < len(scenes):
        raise ValueError(f"No prompt scene {prompt_index} found in {prompt_file}")
    rows = _component_rows(results_root, modes, seed, prompt_file.stem, prompt_index)

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
        f"component outputs - matched seed {seed}",
        output_path,
        provider,
        panel_size,
        row_label=lambda mode: mode.replace("_", " "),
    )
    audit_path = _write_component_audit(
        rows, prompt_file, seed, len(scenes[prompt_index]), output_path
    )
    return output_path, audit_path


def create_component_diagnostic_comparisons(
    results_root,
    prompt_file,
    output_path,
    seed=2025,
    modes=CONSISTENCY_MODES,
    prompt_index=0,
    panel_size=256,
    views=DEFAULT_COMPONENT_VIEWS,
    diff_gain=8.0,
):
    """Render output, mechanism, and causal component-attribution views."""
    prompt_file = Path(prompt_file)
    scenes = parse_prompt_scenes(prompt_file)
    if not scenes or not 0 <= prompt_index < len(scenes):
        raise ValueError(f"No prompt scene {prompt_index} found in {prompt_file}")
    prompt_parts = scenes[prompt_index]
    rows = _component_rows(results_root, modes, seed, prompt_file.stem, prompt_index)
    row_lookup = dict(rows)
    destinations = []

    if "outputs" in views:
        output, audit_path = create_component_comparison(
            results_root, prompt_file, output_path, seed, modes, prompt_index, panel_size
        )
        destinations.append(output)
    else:
        audit_path = _write_component_audit(
            rows, prompt_file, seed, len(prompt_parts), output_path
        )

    def render(view, title, provider):
        destination = _diagnostic_path(output_path, view)
        destinations.append(
            _render_grid(
                prompt_file,
                prompt_parts,
                rows,
                f"{title} - matched seed {seed}",
                destination,
                provider,
                panel_size,
                row_label=lambda mode: mode.replace("_", " "),
            )
        )

    if "incremental_differences" in views:
        def difference_provider(mode, result, column):
            reference_mode = INCREMENTAL_REFERENCE.get(mode)
            if column == 0:
                return None, "identity reference"
            if reference_mode is None:
                return _placeholder(panel_size, "reference"), "causal baseline"
            reference = row_lookup.get(reference_mode)
            if result is None or reference is None:
                return None, ""
            current_path = _frame_image_path(result, column)
            reference_path = _frame_image_path(reference, column)
            if not current_path.is_file() or not reference_path.is_file():
                return None, ""
            current = np.asarray(Image.open(current_path).convert("RGB"), dtype=np.int16)
            baseline = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.int16)
            difference = np.abs(current - baseline)
            scalar = np.clip(difference.mean(axis=2) / 255.0 * diff_gain, 0, 1)
            panel = Image.fromarray(colorize_action_scores(scalar), mode="RGB")
            return panel, f"vs {reference_mode}; MAE {difference.mean():.3f}"

        render(
            "incremental_differences",
            f"incremental output difference x{diff_gain:g}",
            difference_provider,
        )

    def artifact_provider(kind):
        def provider(mode, result, column):
            if result is None:
                return None, ""
            stem = "id" if column == 0 else str(column - 1)
            if kind == "masks":
                path = result / "mask" / f"{stem}_mask.jpg"
                return (_crop_diagnostic_half(path, panel_size), "mask") if path.is_file() else (None, "")
            if kind == "action_attention":
                path = result / "action_attention" / f"{stem}_overlay.jpg"
                return (Image.open(path).convert("RGB"), "action overlay") if path.is_file() else (None, "")
            if column == 0:
                return _placeholder(panel_size, "N/A"), "points begin on generated frames"
            if mode == "prompt_only":
                return _placeholder(panel_size, "not enabled"), "no identity point matching"
            path = result / "points" / f"{stem}_dense.jpg"
            metadata = result / "points" / f"{stem}_dense.json"
            if not path.is_file():
                return None, ""
            valid = None
            if metadata.is_file():
                data = json.loads(metadata.read_text(encoding="utf-8"))
                valid = sum(bool(item["valid"]) for item in data["matches"])
            annotation = f"valid matches: {valid}" if valid is not None else "points"
            return _crop_diagnostic_half(path, panel_size), annotation
        return provider

    for view in ("masks", "action_attention", "points"):
        if view in views:
            render(view, view.replace("_", " "), artifact_provider(view))

    if "merge_effective_mean" in views:
        def merge_provider(mode, result, column):
            if column == 0:
                return _placeholder(panel_size, "N/A"), "merge applies to later frames"
            if mode != "full":
                return _placeholder(panel_size, "not enabled"), "adaptive merge disabled"
            data = _load_npz(result, column - 1) if result is not None else None
            if data is None or not len(data["step_indices"]):
                return None, ""
            aggregate = data["effective_weights"].mean(axis=0)
            return _map_panel(aggregate, panel_size), "mean effective merge weight"

        render("merge_effective_mean", "mean adaptive-merge weights", merge_provider)

    if "component_audit" in views:
        def audit_provider(mode, result, column):
            trace = _load_trace(result)
            if trace is None:
                return None, ""
            if column == 0:
                counts = trace
                label = "total invocations"
            else:
                frames = {str(frame["frame"]): frame for frame in trace.get("frames", [])}
                counts = frames.get(str(column - 1))
                label = f"frame {column - 1}"
            if counts is None:
                return None, ""
            panel = _placeholder(panel_size, counts.get("status", trace.get("status", "UNKNOWN")))
            annotation = (
                f"{label}; attn {counts.get('identity_attention_invocations', 0)}; "
                f"tokens {counts.get('identity_token_applications', 0)}; "
                f"merge {counts.get('adaptive_merge_invocations', 0)}"
            )
            return panel, annotation

        render("component_audit", "component mechanism audit", audit_provider)

    return destinations, audit_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results_colab"))
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--modes", default=",".join(CONSISTENCY_MODES))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--diff-gain", type=float, default=8.0)
    parser.add_argument("--views", default=",".join(DEFAULT_COMPONENT_VIEWS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seed < 0 or args.panel_size < 64 or args.diff_gain <= 0:
        parser.error("seed must be non-negative, panel size >= 64, and diff gain > 0")
    if not args.prompt_file.is_file() or not args.results_root.is_dir():
        parser.error("prompt file and results root must exist")
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    invalid = sorted(set(views) - set(DEFAULT_COMPONENT_VIEWS))
    if invalid:
        parser.error(f"unknown views: {', '.join(invalid)}")
    output = args.output or (
        args.results_root / "component_comparisons" / args.prompt_file.stem
        / f"seed_{args.seed}" / "outputs.jpg"
    )
    try:
        destinations, audit_path = create_component_diagnostic_comparisons(
            args.results_root,
            args.prompt_file,
            output,
            args.seed,
            args.modes,
            args.prompt_index,
            args.panel_size,
            views,
            args.diff_gain,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for destination in destinations:
        print(f"Saved component comparison: {destination}")
    print(f"Saved comparison audit: {audit_path}")


if __name__ == "__main__":
    main()
