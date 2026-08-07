"""Matched-seed comparison and scoring sheet for action-binding results."""

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from characonsist.experiments.conditions import action_binding_label


CONDITIONS = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 1.0))
FIELDS = (
    "condition", "beta", "gamma", "seed", "man_drinking",
    "woman_arms_folded", "identities_correct", "combined_binding_success", "notes",
)


def _load_or_placeholder(path, size=(384, 384)):
    if path.is_file():
        return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    image = Image.new("RGB", size, "#ececec")
    ImageDraw.Draw(image).text((20, 20), f"Missing\n{path.name}", fill="black")
    return image


def render_action_binding_comparison(results_root, output_dir, prompt="2b_final_action_binding"):
    """Render four conditions by five seeds and create a persistent manual CSV."""
    results_root = Path(results_root) / "action_binding_ablation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(2025, 2030)
    cell_size = (384, 384)
    header = 42
    canvas = Image.new("RGB", (cell_size[0] * 5, (cell_size[1] + header) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    rows = []
    for row, (beta, gamma) in enumerate(CONDITIONS):
        label = action_binding_label(beta, gamma)
        for column, seed in enumerate(seeds):
            run_dir = results_root / label / f"seed_{seed}" / "bg_fg" / prompt / "prompt_0"
            image = _load_or_placeholder(run_dir / "0.jpg", cell_size)
            x = column * cell_size[0]
            y = row * (cell_size[1] + header)
            draw.text((x + 8, y + 8), f"beta={beta:g}, gamma={gamma:g}, seed={seed}", fill="black")
            canvas.paste(image, (x, y + header))
            rows.append({
                "condition": label,
                "beta": f"{beta:g}",
                "gamma": f"{gamma:g}",
                "seed": str(seed),
                "man_drinking": "",
                "woman_arms_folded": "",
                "identities_correct": "",
                "combined_binding_success": "",
                "notes": "",
            })
    comparison = output_dir / "action_binding_matrix.png"
    canvas.save(comparison)

    csv_path = output_dir / "manual_scores.csv"
    existing = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing = {
                (row["condition"], row["seed"]): row for row in csv.DictReader(handle)
            }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(existing.get((row["condition"], row["seed"]), row))

    summary = {
        "prompt": prompt,
        "conditions": len(CONDITIONS),
        "seeds": list(seeds),
        "comparison": str(comparison),
        "manual_scores": str(csv_path),
        "note": "Fill booleans as 0/1; combined success requires all three criteria.",
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
