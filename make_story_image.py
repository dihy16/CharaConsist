"""Generate a single story image from finished result images and a prompt file.

Usage examples:
  python make_story_image.py --folder path/to/results --prompt-file path/to/prompt.txt
  python make_story_image.py --folder path/to/results

If --prompt-file is omitted the script will try to find a text file in the folder
matching '*prompt*.txt' or the first '.txt' file available.
The output will be saved as `story.jpg` in the results folder by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

PromptPart = Tuple[str, str, str]


def parse_prompt_scenes(file_path: str | Path) -> List[List[PromptPart]]:
    """Parse bg#fg#act prompt files grouped by blank lines."""
    scenes: List[List[PromptPart]] = []
    current: List[PromptPart] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("#", 2)
                if len(parts) == 3:
                    current.append(tuple(part.strip() for part in parts))
            elif current:
                scenes.append(current)
                current = []
    if current:
        scenes.append(current)
    return scenes


def flatten_prompt_scenes(scenes: List[List[PromptPart]]) -> List[PromptPart]:
    return [part for scene in scenes for part in scene]


def format_prompt_caption(bg: str, fg: str, act: str) -> str:
    """Format bg, fg, and act into a single readable caption."""
    bg = bg.rstrip(" ,").strip()
    fg = fg.rstrip(" ,").strip()
    act = act.lstrip("#").strip()
    if fg and fg[0].islower():
        fg = fg[0].upper() + fg[1:]
    return f"{bg}. {fg}. #{act}"


def find_prompt_file(folder: Path) -> Optional[Path]:
    candidates = list(folder.glob("*prompt*.txt"))
    if not candidates:
        candidates = list(folder.glob("*.txt"))
    return candidates[0] if candidates else None


def search_prompt_anywhere(folder: Path) -> Optional[Path]:
    searches = [folder, folder.parent]
    repo_root = Path(__file__).resolve().parent
    searches.extend([repo_root / "prompts", repo_root / "examples"])
    for search_dir in searches:
        if search_dir and search_dir.exists():
            for pattern in ("*prompt*.txt", "*.txt"):
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]
    return None


def _image_sort_key(path: Path):
    stem = path.stem.lower()
    if stem == "id":
        return (0, 0)
    if stem.isdigit():
        return (1, int(stem))
    return (2, stem)


def list_finished_images(folder: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    imgs = [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in exts
        and path.is_file()
        and "_pre" not in path.stem
        and path.stem.lower() != "story"
    ]
    imgs.sort(key=_image_sort_key)
    return imgs


def collect_result_images(folder: Path, num_prompts: int) -> List[Path]:
    paths: List[Path] = []
    id_path = folder / "id.jpg"
    if id_path.exists():
        paths.append(id_path)
        for index in range(num_prompts - 1):
            frame_path = folder / f"{index}.jpg"
            if frame_path.exists():
                paths.append(frame_path)
    else:
        for index in range(num_prompts):
            frame_path = folder / f"{index}.jpg"
            if frame_path.exists():
                paths.append(frame_path)
    return paths


def load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    words = text.split()
    if not words:
        return []

    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if _text_size(draw, candidate, font)[0] <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def resize_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


def make_story_image(
    images: List[Path],
    caption_text: str,
    out_path: Path,
    panel_size: int = 512,
) -> None:
    if not images:
        raise ValueError("No finished images found to compose.")

    thumbs = [resize_square(Image.open(path), panel_size) for path in images]

    top_margin = 12
    bottom_margin = 24
    gap_below_labels = 8
    gap_above_caption = 20
    label_font = load_font(14)
    caption_font = load_font(20)

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_line_height = _text_size(measure, "Ay", label_font)[1]
    label_area_height = label_line_height

    total_width = panel_size * len(thumbs)
    caption_lines = wrap_text(measure, caption_text, caption_font, total_width - 40)
    caption_line_height = _text_size(measure, "Ay", caption_font)[1]
    caption_area_height = (
        len(caption_lines) * caption_line_height
        + max(0, len(caption_lines) - 1) * 4
    )

    canvas_height = (
        top_margin
        + label_area_height
        + gap_below_labels
        + panel_size
        + gap_above_caption
        + caption_area_height
        + bottom_margin
    )

    canvas = Image.new("RGB", (total_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    label_color = (0, 0, 0)
    caption_color = (90, 0, 130)
    image_y = top_margin + label_area_height + gap_below_labels

    for column, thumb in enumerate(thumbs):
        x = column * panel_size
        label = str(column)
        label_width, _ = _text_size(draw, label, label_font)
        draw.text(
            (x + (panel_size - label_width) / 2, top_margin),
            label,
            fill=label_color,
            font=label_font,
        )
        canvas.paste(thumb, (x, image_y))

    caption_y = image_y + panel_size + gap_above_caption
    for line in caption_lines:
        text_width, _ = _text_size(draw, line, caption_font)
        draw.text(
            ((total_width - text_width) / 2, caption_y),
            line,
            fill=caption_color,
            font=caption_font,
        )
        caption_y += caption_line_height + 4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def save_story_visualization(
    out_dir: str | Path,
    prompt_parts: List[PromptPart],
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    if not prompt_parts:
        return None

    folder = Path(out_dir)
    images = collect_result_images(folder, len(prompt_parts))
    if not images:
        images = list_finished_images(folder)
    if not images:
        print(f"No images found for story visualization in {folder}")
        return None

    count = min(len(images), len(prompt_parts))
    images = images[:count]
    prompt_parts = prompt_parts[:count]

    caption = format_prompt_caption(*prompt_parts[0])
    destination = out_path or (folder / "story.jpg")
    make_story_image(images, caption, destination)
    print(f"Saved story visualization: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Compose finished images into a single story image.")
    parser.add_argument("--folder", "-f", type=Path, default=Path.cwd(), help="results folder containing images")
    parser.add_argument("--prompt-file", "-p", type=Path, default=None, help="file containing the prompt text")
    parser.add_argument("--output", "-o", type=Path, default=None, help="output filename (story.jpg by default)")

    args = parser.parse_args()
    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}")
        return

    prompt_file = args.prompt_file or find_prompt_file(folder) or search_prompt_anywhere(folder)
    if not prompt_file or not prompt_file.exists():
        print("No prompt file provided or found in nearby locations.")
        return

    print("Using prompt file:", prompt_file)
    scenes = parse_prompt_scenes(prompt_file)
    prompt_parts = flatten_prompt_scenes(scenes)
    if not prompt_parts:
        print("No valid bg#fg#act prompt lines found.")
        return

    out_path = args.output or (folder / "story.jpg")
    try:
        save_story_visualization(folder, prompt_parts, out_path)
    except Exception as exc:
        print("Failed to create story image:", exc)


if __name__ == "__main__":
    main()
