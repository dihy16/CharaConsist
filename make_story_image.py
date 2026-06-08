"""Generate a single story image from finished result images and a prompt file.

Prompt files use bg#fg#act lines where:
  - fg  -> [Character] part (identity / appearance)
  - bg + act -> [Action / Environment] part (scene and action)

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

CHARACTER_HEADING = "[Character]"
ACTION_ENV_HEADING = "[Action / Environment]"


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


def format_character_part(fg: str) -> str:
    fg = _clean_part(fg)
    if fg and fg[0].islower():
        fg = fg[0].upper() + fg[1:]
    return fg


def format_action_env_part(bg: str, act: str) -> str:
    bg = _clean_part(bg)
    act = act.lstrip("#").strip()
    if act:
        return f"{bg}. {act}"
    return bg


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


def make_story_image(
    images: List[Path],
    character_text: str,
    action_env_texts: List[str],
    out_path: Path,
    panel_size: int = 512,
) -> None:
    if not images:
        raise ValueError("No finished images found to compose.")
    if len(action_env_texts) != len(images):
        raise ValueError(
            f"Expected {len(images)} action/environment captions, got {len(action_env_texts)}."
        )

    thumbs = [resize_square(Image.open(path), panel_size) for path in images]

    top_margin = 12
    bottom_margin = 24
    gap_below_index = 8
    gap_below_panel = 10
    gap_above_character = 22
    gap_below_heading = 6

    index_font = load_font(14)
    heading_font = load_font(12)
    panel_font = load_font(11)
    character_heading_font = load_font(13)
    character_font = load_font(18)

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    index_line_height = _text_size(measure, "Ay", index_font)[1]
    heading_line_height = _text_size(measure, "Ay", heading_font)[1]
    panel_line_height = _text_size(measure, "Ay", panel_font)[1]
    character_heading_line_height = _text_size(measure, "Ay", character_heading_font)[1]
    character_line_height = _text_size(measure, "Ay", character_font)[1]

    panel_text_width = panel_size - 12
    wrapped_action_env: List[List[str]] = []
    max_panel_lines = 0
    for text in action_env_texts:
        lines = wrap_text(measure, text, panel_font, panel_text_width)
        wrapped_action_env.append(lines)
        max_panel_lines = max(max_panel_lines, len(lines))

    panel_caption_height = (
        heading_line_height
        + gap_below_heading
        + _block_height(max_panel_lines, panel_line_height, 2)
    )

    total_width = panel_size * len(thumbs)
    character_body_lines = wrap_text(
        measure, character_text, character_font, total_width - 48
    )
    character_area_height = (
        character_heading_line_height
        + gap_below_heading
        + _block_height(len(character_body_lines), character_line_height, 4)
    )

    canvas_height = (
        top_margin
        + index_line_height
        + gap_below_index
        + panel_size
        + gap_below_panel
        + panel_caption_height
        + gap_above_character
        + character_area_height
        + bottom_margin
    )

    canvas = Image.new("RGB", (total_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    index_color = (0, 0, 0)
    heading_color = (70, 70, 70)
    action_env_color = (30, 90, 150)
    character_heading_color = (120, 40, 150)
    character_color = (90, 0, 130)

    image_y = top_margin + index_line_height + gap_below_index
    panel_caption_y = image_y + panel_size + gap_below_panel

    for column, thumb in enumerate(thumbs):
        x = column * panel_size
        label = str(column)
        label_width, _ = _text_size(draw, label, index_font)
        draw.text(
            (x + (panel_size - label_width) / 2, top_margin),
            label,
            fill=index_color,
            font=index_font,
        )
        canvas.paste(thumb, (x, image_y))

        heading_width, _ = _text_size(draw, ACTION_ENV_HEADING, heading_font)
        draw.text(
            (x + (panel_size - heading_width) / 2, panel_caption_y),
            ACTION_ENV_HEADING,
            fill=heading_color,
            font=heading_font,
        )

        text_y = panel_caption_y + heading_line_height + gap_below_heading
        for line in wrapped_action_env[column]:
            line_width, _ = _text_size(draw, line, panel_font)
            draw.text(
                (x + (panel_size - line_width) / 2, text_y),
                line,
                fill=action_env_color,
                font=panel_font,
            )
            text_y += panel_line_height + 2

    character_y = panel_caption_y + panel_caption_height + gap_above_character
    heading_width, _ = _text_size(draw, CHARACTER_HEADING, character_heading_font)
    draw.text(
        ((total_width - heading_width) / 2, character_y),
        CHARACTER_HEADING,
        fill=character_heading_color,
        font=character_heading_font,
    )
    character_y += character_heading_line_height + gap_below_heading
    for line in character_body_lines:
        line_width, _ = _text_size(draw, line, character_font)
        draw.text(
            ((total_width - line_width) / 2, character_y),
            line,
            fill=character_color,
            font=character_font,
        )
        character_y += character_line_height + 4

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

    character = format_character_part(prompt_parts[0][1])
    action_env_texts = [
        format_action_env_part(bg, act) for bg, _, act in prompt_parts
    ]
    destination = out_path or (folder / "story.jpg")
    make_story_image(images, character, action_env_texts, destination)
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
