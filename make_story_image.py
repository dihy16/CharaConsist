"""Generate a single story image from finished result images and a prompt file.

Prompt files use environment#character#action lines where:
  - environment -> [Environment] (shared scene context)
  - character   -> [Character] (shared identity)
  - action      -> [Action / Shot] (frame-specific, shown under each panel)

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

ENVIRONMENT_HEADING = "[Environment]"
CHARACTER_HEADING = "[Character]"


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


def _clean_part(text: str) -> str:
    return text.rstrip(" ,").strip()


def _capitalize_first(text: str) -> str:
    if text and text[0].islower():
        return text[0].upper() + text[1:]
    return text


def format_environment_part(bg: str) -> str:
    return _capitalize_first(_clean_part(bg))


def format_character_part(fg: str) -> str:
    return _capitalize_first(_clean_part(fg))


def format_action_shot(act: str) -> str:
    return _capitalize_first(act.lstrip("#").strip())


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


def _block_height(num_lines: int, line_height: int, line_gap: int) -> int:
    if num_lines <= 0:
        return 0
    return num_lines * line_height + (num_lines - 1) * line_gap


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


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    y: int,
    total_width: int,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    line_height: int,
    line_gap: int,
) -> int:
    for line in lines:
        line_width, _ = _text_size(draw, line, font)
        draw.text(((total_width - line_width) / 2, y), line, fill=color, font=font)
        y += line_height + line_gap
    return y


def make_story_image(
    images: List[Path],
    environment_text: str,
    character_text: str,
    action_shot_texts: List[str],
    out_path: Path,
    panel_size: int = 512,
) -> None:
    if not images:
        raise ValueError("No finished images found to compose.")
    if len(action_shot_texts) != len(images):
        raise ValueError(
            f"Expected {len(images)} action/shot captions, got {len(action_shot_texts)}."
        )

    thumbs = [resize_square(Image.open(path), panel_size) for path in images]

    top_margin = 12
    bottom_margin = 24
    gap_below_context = 16
    gap_below_index = 8
    gap_below_panel = 10
    gap_below_heading = 6
    gap_between_context_parts = 10

    heading_font = load_font(12)
    context_font = load_font(16)
    index_font = load_font(14)
    panel_font = load_font(11)

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    heading_line_height = _text_size(measure, "Ay", heading_font)[1]
    context_line_height = _text_size(measure, "Ay", context_font)[1]
    index_line_height = _text_size(measure, "Ay", index_font)[1]
    panel_line_height = _text_size(measure, "Ay", panel_font)[1]

    total_width = panel_size * len(thumbs)
    context_width = total_width - 48

    env_lines = wrap_text(measure, environment_text, context_font, context_width)
    char_lines = wrap_text(measure, character_text, context_font, context_width)

    panel_text_width = panel_size - 12
    wrapped_action_shots: List[List[str]] = []
    max_panel_lines = 0
    for text in action_shot_texts:
        lines = wrap_text(measure, text, panel_font, panel_text_width)
        wrapped_action_shots.append(lines)
        max_panel_lines = max(max_panel_lines, len(lines))

    context_area_height = (
        heading_line_height
        + gap_below_heading
        + _block_height(len(env_lines), context_line_height, 3)
        + gap_between_context_parts
        + heading_line_height
        + gap_below_heading
        + _block_height(len(char_lines), context_line_height, 3)
    )
    panel_caption_height = _block_height(max_panel_lines, panel_line_height, 2)

    canvas_height = (
        top_margin
        + context_area_height
        + gap_below_context
        + index_line_height
        + gap_below_index
        + panel_size
        + gap_below_panel
        + panel_caption_height
        + bottom_margin
    )

    canvas = Image.new("RGB", (total_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    heading_color = (70, 70, 70)
    environment_color = (30, 90, 150)
    character_color = (90, 0, 130)
    index_color = (0, 0, 0)
    action_shot_color = (50, 50, 50)

    y = top_margin
    env_heading_width, _ = _text_size(draw, ENVIRONMENT_HEADING, heading_font)
    draw.text(
        ((total_width - env_heading_width) / 2, y),
        ENVIRONMENT_HEADING,
        fill=heading_color,
        font=heading_font,
    )
    y += heading_line_height + gap_below_heading
    y = _draw_centered_lines(
        draw, env_lines, y, total_width, context_font, environment_color,
        context_line_height, 3,
    )
    y += gap_between_context_parts

    char_heading_width, _ = _text_size(draw, CHARACTER_HEADING, heading_font)
    draw.text(
        ((total_width - char_heading_width) / 2, y),
        CHARACTER_HEADING,
        fill=heading_color,
        font=heading_font,
    )
    y += heading_line_height + gap_below_heading
    y = _draw_centered_lines(
        draw, char_lines, y, total_width, context_font, character_color,
        context_line_height, 3,
    )

    index_y = y + gap_below_context
    image_y = index_y + index_line_height + gap_below_index
    panel_caption_y = image_y + panel_size + gap_below_panel

    for column, thumb in enumerate(thumbs):
        x = column * panel_size
        label = str(column)
        label_width, _ = _text_size(draw, label, index_font)
        draw.text(
            (x + (panel_size - label_width) / 2, index_y),
            label,
            fill=index_color,
            font=index_font,
        )
        canvas.paste(thumb, (x, image_y))

        text_y = panel_caption_y
        for line in wrapped_action_shots[column]:
            line_width, _ = _text_size(draw, line, panel_font)
            draw.text(
                (x + (panel_size - line_width) / 2, text_y),
                line,
                fill=action_shot_color,
                font=panel_font,
            )
            text_y += panel_line_height + 2

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

    environment = format_environment_part(prompt_parts[0][0])
    character = format_character_part(prompt_parts[0][1])
    action_shots = [format_action_shot(act) for _, _, act in prompt_parts]
    destination = out_path or (folder / "story.jpg")
    make_story_image(images, environment, character, action_shots, destination)
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
