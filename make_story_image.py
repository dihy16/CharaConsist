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
import os
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
import textwrap


def find_prompt_file(folder: Path) -> Optional[Path]:
    candidates = list(folder.glob("*prompt*.txt"))
    if not candidates:
        candidates = list(folder.glob("*.txt"))
    return candidates[0] if candidates else None


def search_prompt_anywhere(folder: Path) -> Optional[Path]:
    # Try folder, parent, repo prompts folder, examples, then any txt nearby
    searches = [folder, folder.parent]
    repo_root = Path(__file__).resolve().parent
    searches.append(repo_root / "prompts")
    searches.append(repo_root / "examples")
    for s in searches:
        if s and s.exists():
            # prefer files with 'prompt' in name
            for p in s.glob("*prompt*.txt"):
                return p
            for p in s.glob("*.txt"):
                return p
    return None


def list_finished_images(folder: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]
    # exclude preview/_pre images
    imgs = [p for p in imgs if "_pre" not in p.stem and "-pre" not in p.stem]
    imgs.sort()
    return imgs


def make_story_image(images: List[Path], prompt_text: str, out_path: Path, max_height=512):
    if not images:
        raise ValueError("No finished images found to compose.")

    thumbs = []
    labels = []
    for p in images:
        img = Image.open(p).convert("RGB")
        # scale to max_height while keeping aspect
        w, h = img.size
        if h != max_height:
            new_w = int(w * (max_height / h))
            img = img.resize((new_w, max_height), Image.LANCZOS)
        thumbs.append(img)
        labels.append(p.stem)

    padding = 16
    label_height = 24
    bottom_prompt_height = 160

    total_width = sum(im.width for im in thumbs) + padding * (len(thumbs) + 1)
    canvas_height = label_height + max(im.height for im in thumbs) + bottom_prompt_height + padding * 2

    canvas = Image.new("RGB", (total_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        prompt_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        prompt_font = font

    x = padding
    for im, label in zip(thumbs, labels):
        # draw label centered above image
        w, h = im.size
        label_y = padding
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            txt_w = bbox[2] - bbox[0]
            txt_h = bbox[3] - bbox[1]
        except Exception:
            txt_w, txt_h = font.getsize(label)
        draw.text((x + (w - txt_w) / 2, label_y), label, fill=(0, 0, 0), font=font)

        # paste image
        img_y = padding + label_height
        canvas.paste(im, (x, img_y))
        x += w + padding

    # draw prompt text at bottom
    prompt_area_top = padding + label_height + max(im.height for im in thumbs) + 20
    prompt_area_width = total_width - padding * 2
    # prepare multiline prompt text and draw it centered with a faint background
    lines = textwrap.wrap(prompt_text, width=80)
    prompt_block = "\n".join(lines) if lines else ""
    try:
        bbox = draw.multiline_textbbox((0, 0), prompt_block, font=prompt_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        # fallback sizing
        tw = prompt_font.getsize(prompt_block)[0] if prompt_block else 0
        line_h = prompt_font.getsize("A")[1]
        th = line_h * max(1, len(lines)) + 6 * max(0, len(lines) - 1)

    if prompt_block:
        pad = 12
        box_x = int((total_width - tw) / 2) - pad
        box_y = int(prompt_area_top) - pad
        box_w = int(tw) + pad * 2
        box_h = int(th) + pad * 2
        draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(250, 250, 250))
        draw.multiline_text(((total_width - tw) / 2, prompt_area_top), prompt_block, fill=(80, 0, 120), font=prompt_font, align="center")

    canvas.save(out_path, quality=92)


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

    prompt_file = args.prompt_file
    if prompt_file is None:
        found = find_prompt_file(folder)
        prompt_file = found

    prompt_text = ""
    used_prompt_file = None
    if prompt_file and prompt_file.exists():
        used_prompt_file = prompt_file
    else:
        cand = search_prompt_anywhere(folder)
        if cand:
            used_prompt_file = cand

    if used_prompt_file:
        try:
            raw = used_prompt_file.read_text(encoding="utf-8").strip()
            # If file contains multiple lines, try to find an 'Identity Prompt:' line
            if "Identity Prompt" in raw:
                for line in raw.splitlines():
                    if "Identity Prompt" in line:
                        # extract after ':' if present
                        if ":" in line:
                            prompt_text = line.split(":", 1)[1].strip()
                        else:
                            prompt_text = line.strip()
                        break
            else:
                # use first non-empty line
                for line in raw.splitlines():
                    if line.strip():
                        prompt_text = line.strip()
                        break
        except Exception:
            prompt_text = ""
    else:
        print("No prompt file provided or found in nearby locations. Prompt text will be empty.")
    if used_prompt_file:
        print("Using prompt file:", used_prompt_file)

    images = list_finished_images(folder)
    if not images:
        print("No finished images found in", folder)
        return

    out_path = args.output or (folder / "story.jpg")
    try:
        make_story_image(images, prompt_text, out_path)
        print("Saved:", out_path)
    except Exception as e:
        print("Failed to create story image:", e)


if __name__ == "__main__":
    main()
