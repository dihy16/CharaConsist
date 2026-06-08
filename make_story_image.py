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
    lines = textwrap.wrap(prompt_text, width=80)
    # try to fit font size larger if possible
    y = prompt_area_top
    for line in lines:
        try:
            tb = draw.textbbox((0, 0), line, font=prompt_font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
        except Exception:
            tw, th = prompt_font.getsize(line)
        draw.text(((total_width - tw) / 2, y), line, fill=(80, 0, 120), font=prompt_font)
        y += th + 6

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
    if prompt_file and prompt_file.exists():
        try:
            prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        except Exception:
            prompt_text = ""

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
