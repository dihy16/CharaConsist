#!/usr/bin/env python3
"""Browse matched routing-off and routing-hard images with the arrow keys."""

import argparse
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:  # pragma: no cover - platform dependent
    tk = None

from PIL import ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from characonsist.visualization.entity_routing_viewer import (
    amplified_difference,
    discover_routing_seeds,
    identity_image_path,
    load_result_image,
    result_image_path,
)


class EntityRoutingViewer:
    """A small Tk viewer for side-by-side, seed-matched comparisons."""

    def __init__(self, root, args, seeds):
        self.root = root
        self.args = args
        self.seeds = seeds
        self.index = 0
        self.image_size = (args.size, args.size)
        self.image_labels = []

        root.title("Entity routing: off vs hard")
        self.title = tk.Label(root, font=("TkDefaultFont", 13, "bold"))
        self.title.pack(pady=(10, 2))
        tk.Label(
            root,
            text="Left/Right: change seed    Home/End: first/last    Q or Esc: close",
        ).pack(pady=(0, 10))

        panels = tk.Frame(root)
        panels.pack(padx=10, pady=(0, 10))
        for row, image_type in enumerate(("Identity image", "Story frame")):
            for column, condition in enumerate(("routing off", "routing hard", "difference ×4")):
                frame = tk.Frame(panels)
                frame.grid(row=row, column=column, padx=5, pady=4)
                tk.Label(
                    frame,
                    text=f"{image_type}: {condition}",
                    font=("TkDefaultFont", 11, "bold"),
                ).pack()
                image_label = tk.Label(frame)
                image_label.pack()
                self.image_labels.append(image_label)

        root.bind("<Left>", lambda _event: self.move(-1))
        root.bind("<Right>", lambda _event: self.move(1))
        root.bind("<Home>", lambda _event: self.set_index(0))
        root.bind("<End>", lambda _event: self.set_index(len(self.seeds) - 1))
        root.bind("q", lambda _event: root.destroy())
        root.bind("Q", lambda _event: root.destroy())
        root.bind("<Escape>", lambda _event: root.destroy())
        self.render()

    def move(self, delta):
        self.set_index((self.index + delta) % len(self.seeds))

    def set_index(self, index):
        self.index = index
        self.render()

    def render(self):
        seed = self.seeds[self.index]
        off_path = result_image_path(
            self.args.results_root, "off", self.args.beta, self.args.gamma,
            seed, self.args.prompt, self.args.frame,
        )
        hard_path = result_image_path(
            self.args.results_root, "hard", self.args.beta, self.args.gamma,
            seed, self.args.prompt, self.args.frame,
        )
        id_off_path = identity_image_path(
            self.args.results_root, "off", self.args.beta, self.args.gamma,
            seed, self.args.prompt,
        )
        id_hard_path = identity_image_path(
            self.args.results_root, "hard", self.args.beta, self.args.gamma,
            seed, self.args.prompt,
        )
        off_image = load_result_image(off_path, self.image_size)
        hard_image = load_result_image(hard_path, self.image_size)
        id_off_image = load_result_image(id_off_path, self.image_size)
        id_hard_image = load_result_image(id_hard_path, self.image_size)
        images = (
            id_off_image,
            id_hard_image,
            amplified_difference(id_off_image, id_hard_image),
            off_image,
            hard_image,
            amplified_difference(off_image, hard_image),
        )
        for label, image in zip(self.image_labels, images):
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.image = photo
        self.title.configure(
            text=(
                f"Seed {seed} ({self.index + 1}/{len(self.seeds)})   "
                f"β={self.args.beta:g}, γ={self.args.gamma:g}   "
                f"{self.args.prompt}, frame {self.args.frame}"
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results_colab")
    parser.add_argument("--prompt", default="2b_final_action_binding")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--size", type=int, default=420, help="display width/height per image")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = discover_routing_seeds(
        args.results_root, args.beta, args.gamma, args.prompt, args.frame
    )
    if not seeds:
        print(
            "No matching final images found. Check --results-root, --prompt, "
            "--beta, --gamma, and --frame.",
            file=sys.stderr,
        )
        return 2
    if tk is None:
        print("Tk is not available in this Python installation.", file=sys.stderr)
        return 2
    root = tk.Tk()
    EntityRoutingViewer(root, args, seeds)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
