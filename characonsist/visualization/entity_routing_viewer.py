"""Interactive helpers for inspecting matched entity-routing outputs."""

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance

from characonsist.experiments.conditions import action_binding_label


def result_image_path(results_root, mode, beta, gamma, seed, prompt, frame=0):
    """Return the final-frame image path for one routing condition."""
    return (
        Path(results_root) / "entity_routing_ablation" / f"routing_{mode}"
        / action_binding_label(beta, gamma) / f"seed_{seed}" / "bg_fg"
        / prompt / "prompt_0" / f"{frame}.jpg"
    )


def identity_image_path(results_root, mode, beta, gamma, seed, prompt):
    """Return the standalone identity image generated before frame routing begins."""
    return result_image_path(
        results_root, mode, beta, gamma, seed, prompt
    ).with_name("id.jpg")


def discover_routing_seeds(results_root, beta, gamma, prompt, frame=0):
    """Find every seed with an off or hard result for the selected condition."""
    seeds = set()
    for mode in ("off", "hard"):
        run_root = (
            Path(results_root) / "entity_routing_ablation" / f"routing_{mode}"
            / action_binding_label(beta, gamma)
        )
        if not run_root.is_dir():
            continue
        for seed_dir in run_root.glob("seed_*"):
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            path = result_image_path(results_root, mode, beta, gamma, seed, prompt, frame)
            if path.is_file():
                seeds.add(seed)
    return sorted(seeds)


def load_result_image(path, size):
    """Load a result or a labelled placeholder when one condition is missing."""
    path = Path(path)
    if path.is_file():
        return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    image = Image.new("RGB", size, "#eeeeee")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Missing result", fill="black")
    draw.text((20, 48), str(path), fill="black")
    return image


def amplified_difference(off_image, hard_image, factor=4.0):
    """Make otherwise subtle RGB differences visible without changing either input."""
    difference = ImageChops.difference(off_image, hard_image)
    return ImageEnhance.Brightness(difference).enhance(factor)
