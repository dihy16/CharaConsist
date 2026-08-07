"""Matched 2x2 entity-routing comparison and scoring sheet."""

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from characonsist.experiments.conditions import action_binding_label


CONDITIONS = (
    ("off", 0.0, 0.0),
    ("off", 1.0, 0.5),
    ("hard", 0.0, 0.0),
    ("hard", 1.0, 0.5),
)
FIELDS = (
    "routing_mode", "beta", "gamma", "seed", "man_drinking",
    "woman_arms_folded", "identities_correct", "combined_binding_success", "notes",
)


def _load_or_placeholder(path, size=(384, 384)):
    if path.is_file():
        return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    image = Image.new("RGB", size, "#ececec")
    ImageDraw.Draw(image).text((20, 20), f"Missing\n{path.name}", fill="black")
    return image


def render_entity_routing_comparison(
    results_root, output_dir, prompt="2b_final_action_binding"
):
    """Render routing x action-bias conditions across five matched seeds."""
    results_root = Path(results_root) / "entity_routing_ablation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(2025, 2030)
    cell_size = (384, 384)
    header = 42
    canvas = Image.new("RGB", (cell_size[0] * 5, (cell_size[1] + header) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    rows = []
    for row_index, (mode, beta, gamma) in enumerate(CONDITIONS):
        binding_label = action_binding_label(beta, gamma)
        for column, seed in enumerate(seeds):
            run_dir = (
                results_root / f"routing_{mode}" / binding_label / f"seed_{seed}"
                / "bg_fg" / prompt / "prompt_0"
            )
            image = _load_or_placeholder(run_dir / "0.jpg", cell_size)
            x = column * cell_size[0]
            y = row_index * (cell_size[1] + header)
            draw.text(
                (x + 8, y + 8),
                f"routing={mode}, beta={beta:g}, gamma={gamma:g}, seed={seed}",
                fill="black",
            )
            canvas.paste(image, (x, y + header))
            rows.append({
                "routing_mode": mode,
                "beta": f"{beta:g}",
                "gamma": f"{gamma:g}",
                "seed": str(seed),
                "man_drinking": "",
                "woman_arms_folded": "",
                "identities_correct": "",
                "combined_binding_success": "",
                "notes": "",
            })

    comparison = output_dir / "entity_routing_matrix.png"
    canvas.save(comparison)
    csv_path = output_dir / "manual_scores.csv"
    existing = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing = {
                (item["routing_mode"], item["beta"], item["gamma"], item["seed"]): item
                for item in csv.DictReader(handle)
            }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            key = (row["routing_mode"], row["beta"], row["gamma"], row["seed"])
            writer.writerow(existing.get(key, row))

    manifest = {
        "prompt": prompt,
        "conditions": [list(condition) for condition in CONDITIONS],
        "seeds": list(seeds),
        "comparison": str(comparison),
        "manual_scores": str(csv_path),
        "expected_outcome": (
            "Hard routing should produce zero wrong-bank access and may improve identity "
            "separation; action/prop ownership may remain unchanged because the intervention "
            "does not explicitly route the mug."
        ),
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
