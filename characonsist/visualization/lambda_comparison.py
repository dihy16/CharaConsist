"""Render separate, auditable comparisons of one prompt across lambda values."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from characonsist.diagnostics.action_attention import colorize_action_scores
from characonsist.story import (
    format_action_shot,
    load_font,
    parse_prompt_scenes,
    resize_square,
    wrap_text,
)
from characonsist.experiments.conditions import lambda_label, parse_action_gate_strengths


LAMBDA_DIRECTORY_PATTERN = re.compile(r"^lambda_(\d+)p(\d+)$")
DEFAULT_VIEWS = (
    "outputs",
    "masks",
    "action_attention",
    "points",
    "merge_effective_mean",
    "merge_suppression_mean",
    "merge_suppression_max",
    "merge_step_curves",
    "difference",
    "audit",
)


def _lambda_from_directory(path: Path) -> float:
    match = LAMBDA_DIRECTORY_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Invalid lambda directory name: {path.name}")
    return float(f"{match.group(1)}.{match.group(2)}")


def find_lambda_directories(results_root: Path, strengths=None):
    """Return requested or discovered lambda directories in numeric order."""
    if strengths is not None:
        values = parse_action_gate_strengths(strengths)
        return [(value, results_root / lambda_label(value)) for value in values]
    directories = []
    for path in results_root.glob("lambda_*"):
        if path.is_dir() and LAMBDA_DIRECTORY_PATTERN.match(path.name):
            directories.append((_lambda_from_directory(path), path))
    return sorted(directories, key=lambda item: item[0])


def find_prompt_result(lambda_directory, seed, prompt_stem, prompt_index):
    """Find the result directory while allowing prompt subdirectories."""
    base = lambda_directory / f"seed_{seed}" / "bg_fg"
    if not base.is_dir():
        return None
    matches = [
        path
        for path in base.rglob(f"prompt_{prompt_index}")
        if path.is_dir() and path.parent.name == prompt_stem
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple results found for {prompt_stem}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def _text_size(draw, text, font):
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def _draw_centered_lines(draw, lines, bounds, font, fill, gap=3):
    left, top, right, _ = bounds
    line_height = _text_size(draw, "Ay", font)[1]
    y = top
    for line in lines:
        width, _ = _text_size(draw, line, font)
        draw.text((left + (right - left - width) / 2, y), line, font=font, fill=fill)
        y += line_height + gap


def _placeholder(size, text):
    image = Image.new("RGB", (size, size), (232, 232, 232))
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    width, height = _text_size(draw, text, font)
    draw.text(((size - width) / 2, (size - height) / 2), text, font=font, fill=(130, 130, 130))
    return image


def _annotate(image, text):
    if not text:
        return image
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(13)
    lines = wrap_text(draw, text, font, image.width - 12)[:2]
    line_height = _text_size(draw, "Ay", font)[1] + 2
    height = len(lines) * line_height + 8
    draw.rectangle((0, image.height - height, image.width, image.height), fill=(0, 0, 0))
    y = image.height - height + 4
    for line in lines:
        draw.text((6, y), line, font=font, fill=(255, 255, 255))
        y += line_height
    return image


def _crop_diagnostic_half(path, panel_size):
    with Image.open(path) as source:
        source = source.convert("RGB")
        if source.width >= source.height * 2:
            source = source.crop((source.width // 2, 0, source.width, source.height))
        return resize_square(source, panel_size)


def _load_npz(result_directory, frame):
    path = result_directory / "merge_diagnostics" / f"{frame}_weights.npz"
    if not path.is_file():
        return None
    return dict(np.load(path, allow_pickle=False))


def _map_panel(values, panel_size, scale=0.8):
    normalized = np.clip(np.asarray(values, dtype=np.float32) / scale, 0.0, 1.0)
    image = Image.fromarray(colorize_action_scores(normalized), mode="RGB")
    return image.resize((panel_size, panel_size), Image.Resampling.NEAREST)


def _curve_panel(data, panel_size):
    image = Image.new("RGB", (panel_size, panel_size), "white")
    draw = ImageDraw.Draw(image)
    margin = 28
    draw.rectangle((margin, margin, panel_size - 10, panel_size - margin), outline=(100, 100, 100))
    steps = data["step_indices"].astype(float)
    valid = data["valid_mask"].astype(bool)
    series = {}
    for name in ("base_weights", "effective_weights", "suppressed_weights"):
        values = data[name].astype(np.float32)
        series[name] = np.asarray(
            [values[i][valid[i]].mean() if valid[i].any() else 0.0 for i in range(len(steps))]
        )
    colors = {
        "base_weights": (80, 80, 80),
        "effective_weights": (30, 100, 210),
        "suppressed_weights": (210, 45, 45),
    }
    if len(steps):
        low, high = steps.min(), steps.max()
        span = max(high - low, 1.0)
        for name, values in series.items():
            points = []
            for step, value in zip(steps, values):
                x = margin + (step - low) / span * (panel_size - margin - 10)
                y = panel_size - margin - np.clip(value / 0.8, 0, 1) * (panel_size - 2 * margin)
                points.append((float(x), float(y)))
            if len(points) > 1:
                draw.line(points, fill=colors[name], width=3)
            elif points:
                x, y = points[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[name])
    font = load_font(11)
    draw.text((4, 4), "gray=base blue=effective red=suppressed", font=font, fill=(30, 30, 30))
    draw.text((4, panel_size - 18), "denoising step", font=font, fill=(30, 30, 30))
    return image


def _render_grid(
    prompt_file, prompt_parts, rows, title_suffix, output_path, provider, panel_size,
    row_label=None,
):
    left_width, title_height, header_height, row_gap, margin = 150, 48, 118, 12, 16
    frame_count = len(prompt_parts)
    width = margin * 2 + left_width + frame_count * panel_size
    height = margin * 2 + title_height + header_height + len(rows) * panel_size + max(0, len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font, heading_font, caption_font = load_font(24), load_font(18), load_font(14)
    row_font, status_font = load_font(22), load_font(14)
    draw.text((margin, margin), f"{prompt_file.stem} - {title_suffix}", font=title_font, fill=(20, 20, 20))
    header_top = margin + title_height
    for column, (_, _, action) in enumerate(prompt_parts):
        left = margin + left_width + column * panel_size
        heading = "ID" if column == 0 else f"Frame {column}"
        heading_width, _ = _text_size(draw, heading, heading_font)
        draw.text((left + (panel_size - heading_width) / 2, header_top), heading, font=heading_font, fill=(20, 20, 20))
        lines = wrap_text(draw, format_action_shot(action), caption_font, panel_size - 16)[:4]
        _draw_centered_lines(draw, lines, (left + 8, header_top + 28, left + panel_size - 8, header_top + header_height), caption_font, (70, 70, 70))

    rows_top = header_top + header_height
    for row_index, row in enumerate(rows):
        strength, result_directory = row
        top = rows_top + row_index * (panel_size + row_gap)
        label = row_label(strength) if row_label is not None else f"lambda = {strength:g}"
        label_width, label_height = _text_size(draw, label, row_font)
        label_y = top + panel_size / 2 - label_height
        draw.text((margin + (left_width - label_width) / 2, label_y), label, font=row_font, fill=(20, 20, 20))
        if result_directory is None:
            status = "missing / failed"
            status_width, _ = _text_size(draw, status, status_font)
            draw.text((margin + (left_width - status_width) / 2, label_y + label_height + 8), status, font=status_font, fill=(170, 45, 45))
        for column in range(frame_count):
            panel, annotation = provider(strength, result_directory, column)
            if panel is None:
                panel = _placeholder(panel_size, "N/A" if column == 0 else "missing")
            else:
                panel = resize_square(panel, panel_size)
            panel = _annotate(panel, annotation)
            canvas.paste(panel, (margin + left_width + column * panel_size, top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)
    return output_path


def _frame_image_path(result_directory, column):
    return result_directory / ("id.jpg" if column == 0 else f"{column - 1}.jpg")


def create_lambda_comparison(results_root, prompt_file, output_path, seed=2025, strengths=None, prompt_index=0, panel_size=256):
    """Backward-compatible generated-output comparison."""
    scenes = parse_prompt_scenes(prompt_file)
    if not scenes or not 0 <= prompt_index < len(scenes):
        raise ValueError(f"No prompt scene {prompt_index} found in {prompt_file}")
    lambda_directories = find_lambda_directories(results_root, strengths)
    if not lambda_directories:
        raise ValueError(f"No lambda result directories found below {results_root}")
    rows = [(value, find_prompt_result(path, seed, prompt_file.stem, prompt_index)) for value, path in lambda_directories]

    def provider(_strength, result, column):
        if result is None:
            return None, ""
        path = _frame_image_path(result, column)
        return (Image.open(path).convert("RGB"), "") if path.is_file() else (None, "")

    return _render_grid(prompt_file, scenes[prompt_index], rows, f"outputs - seed {seed}", output_path, provider, panel_size)


def create_diagnostic_comparisons(results_root, prompt_file, output_path, seed=2025, strengths=None, prompt_index=0, panel_size=256, views=DEFAULT_VIEWS, diff_gain=8.0):
    """Create separate diagnostic figures and a machine-readable comparison audit."""
    scenes = parse_prompt_scenes(prompt_file)
    if not scenes or not 0 <= prompt_index < len(scenes):
        raise ValueError(f"No prompt scene {prompt_index} found in {prompt_file}")
    prompt_parts = scenes[prompt_index]
    lambda_directories = find_lambda_directories(results_root, strengths)
    if not lambda_directories:
        raise ValueError(f"No lambda result directories found below {results_root}")
    rows = [(value, find_prompt_result(path, seed, prompt_file.stem, prompt_index)) for value, path in lambda_directories]
    base_path = output_path.with_suffix("")

    def diagnostic_path(view, suffix=".jpg"):
        """Use concise names for the default per-prompt output directory."""
        if output_path.stem == "outputs":
            return output_path.parent / f"{view}{suffix}"
        return Path(f"{base_path}_{view}{suffix}")
    destinations = []

    if "outputs" in views:
        destinations.append(create_lambda_comparison(results_root, prompt_file, output_path, seed, strengths, prompt_index, panel_size))

    def artifact_provider(kind):
        def provider(_strength, result, column):
            if result is None or column == 0 and kind == "points":
                return None, ""
            stem = "id" if column == 0 else str(column - 1)
            if kind == "masks":
                path = result / "mask" / f"{stem}_mask.jpg"
                return (_crop_diagnostic_half(path, panel_size), "mask") if path.is_file() else (None, "")
            if kind == "action_attention":
                path = result / "action_attention" / f"{stem}_overlay.jpg"
                return (Image.open(path).convert("RGB"), "action overlay") if path.is_file() else (None, "")
            path = result / "points" / f"{stem}_dense.jpg"
            metadata = result / "points" / f"{stem}_dense.json"
            if not path.is_file():
                return None, ""
            valid = None
            if metadata.is_file():
                data = json.loads(metadata.read_text(encoding="utf-8"))
                valid = sum(bool(item["valid"]) for item in data["matches"])
            return _crop_diagnostic_half(path, panel_size), f"valid matches: {valid}" if valid is not None else "points"
        return provider

    for view in ("masks", "action_attention", "points"):
        if view in views:
            destination = diagnostic_path(view)
            destinations.append(_render_grid(prompt_file, prompt_parts, rows, f"{view.replace('_', ' ')} - seed {seed}", destination, artifact_provider(view), panel_size))

    merge_views = {
        "merge_effective_mean": ("effective_weights", "mean"),
        "merge_suppression_mean": ("suppressed_weights", "mean"),
        "merge_suppression_max": ("suppressed_weights", "max"),
    }
    for view, (array_name, reduction) in merge_views.items():
        if view not in views:
            continue
        def provider(_strength, result, column, array_name=array_name, reduction=reduction):
            if result is None or column == 0:
                return None, ""
            data = _load_npz(result, column - 1)
            if data is None or not len(data["step_indices"]):
                return None, ""
            values = data[array_name]
            aggregate = values.mean(axis=0) if reduction == "mean" else values.max(axis=0)
            return _map_panel(aggregate, panel_size), f"{reduction} {array_name.replace('_', ' ')}"
        destination = diagnostic_path(view)
        destinations.append(_render_grid(prompt_file, prompt_parts, rows, f"{view.replace('_', ' ')} - seed {seed}", destination, provider, panel_size))

    if "merge_step_curves" in views:
        def curve_provider(_strength, result, column):
            if result is None or column == 0:
                return None, ""
            data = _load_npz(result, column - 1)
            return (_curve_panel(data, panel_size), f"steps: {len(data['step_indices'])}") if data is not None else (None, "")
        destination = diagnostic_path("merge_step_curves")
        destinations.append(_render_grid(prompt_file, prompt_parts, rows, f"merge step curves - seed {seed}", destination, curve_provider, panel_size))

    successful = [(strength, result) for strength, result in rows if result is not None]
    reference_strength, reference_result = successful[0] if successful else (None, None)
    comparison_audit = {"schema_version": 1, "prompt": prompt_file.stem, "seed": seed, "difference_reference": reference_strength, "conditions": {}}
    for strength, result in rows:
        condition = {"result": "available" if result is not None else "missing", "frames": []}
        audit_path = result / "action_gate_audit.json" if result is not None else None
        if audit_path is not None and audit_path.is_file():
            condition["action_gate_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
        for column in range(1, len(prompt_parts)):
            metrics = {"frame": column - 1, "status": "UNAVAILABLE"}
            if result is not None and reference_result is not None:
                left_path = _frame_image_path(result, column)
                right_path = _frame_image_path(reference_result, column)
                if left_path.is_file() and right_path.is_file():
                    left = np.asarray(Image.open(left_path).convert("RGB"), dtype=np.int16)
                    right = np.asarray(Image.open(right_path).convert("RGB"), dtype=np.int16)
                    difference = np.abs(left - right)
                    metrics = {
                        "frame": column - 1,
                        "status": "CHANGED" if np.any(difference) else "UNCHANGED",
                        "mae": float(difference.mean()),
                        "changed_pixels_percent": float(np.any(difference, axis=2).mean() * 100),
                    }
            condition["frames"].append(metrics)
        comparison_audit["conditions"][lambda_label(strength)] = condition

    audit_path = diagnostic_path("comparison_audit", ".json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(comparison_audit, indent=2) + "\n", encoding="utf-8")

    if "difference" in views:
        def difference_provider(strength, result, column):
            if result is None or reference_result is None or column == 0:
                return None, ""
            left = np.asarray(Image.open(_frame_image_path(result, column)).convert("RGB"), dtype=np.int16)
            right = np.asarray(Image.open(_frame_image_path(reference_result, column)).convert("RGB"), dtype=np.int16)
            difference = np.abs(left - right)
            scalar = np.clip(difference.mean(axis=2) / 255.0 * diff_gain, 0, 1)
            panel = Image.fromarray(colorize_action_scores(scalar), mode="RGB")
            return panel, f"vs lambda {reference_strength:g}; MAE {difference.mean():.3f}"
        destination = diagnostic_path("difference")
        destinations.append(_render_grid(prompt_file, prompt_parts, rows, f"output difference x{diff_gain:g} - seed {seed}", destination, difference_provider, panel_size))

    if "audit" in views:
        def audit_provider(strength, result, column):
            if result is None or column == 0:
                return None, ""
            path = result / "action_gate_audit.json"
            if not path.is_file():
                return None, ""
            audit = json.loads(path.read_text(encoding="utf-8"))
            frames = {int(item["frame"]): item for item in audit.get("frames", [])}
            frame = frames.get(column - 1)
            if frame is None:
                return None, ""
            panel = _placeholder(panel_size, frame["status"])
            text = f"steps {frame['steps_recorded']} modified {frame['modified_token_applications']} max {frame['suppression_max']:.4f}"
            return panel, text
        destination = diagnostic_path("audit")
        destinations.append(_render_grid(prompt_file, prompt_parts, rows, f"action gate audit - seed {seed}", destination, audit_provider, panel_size))

    return destinations, audit_path


def main():
    parser = argparse.ArgumentParser(description="Compare one prompt across lambda values.")
    parser.add_argument("--results-root", type=Path, default=Path("results_colab"))
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lambdas", default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--diff-gain", type=float, default=8.0)
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seed < 0 or args.panel_size < 64 or args.diff_gain <= 0:
        parser.error("seed must be non-negative, panel size >= 64, and diff gain > 0")
    if not args.prompt_file.is_file() or not args.results_root.is_dir():
        parser.error("prompt file and results root must exist")
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    invalid = sorted(set(views) - set(DEFAULT_VIEWS))
    if invalid:
        parser.error(f"unknown views: {', '.join(invalid)}")
    output = args.output or (
        args.results_root / "comparisons" / args.prompt_file.stem
        / f"seed_{args.seed}" / "outputs.jpg"
    )
    try:
        destinations, audit_path = create_diagnostic_comparisons(
            args.results_root, args.prompt_file, output, args.seed,
            args.lambdas, args.prompt_index, args.panel_size, views, args.diff_gain,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for destination in destinations:
        print(f"Saved lambda comparison: {destination}")
    print(f"Saved comparison audit: {audit_path}")


if __name__ == "__main__":
    main()
