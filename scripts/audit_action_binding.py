#!/usr/bin/env python3
"""Audit zero-strength equivalence and the one-seed mechanical checkpoint."""

import argparse
import hashlib
import json

import numpy as np
from PIL import Image
from pathlib import Path

def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mae(first, second):
    first_pixels = np.asarray(Image.open(first).convert("RGB"), dtype=np.float32)
    second_pixels = np.asarray(Image.open(second).convert("RGB"), dtype=np.float32)
    if first_pixels.shape != second_pixels.shape:
        return None
    return float(np.abs(first_pixels - second_pixels).mean())


def _maps_equal(first, second):
    with np.load(first) as first_maps, np.load(second) as second_maps:
        return set(first_maps.files) == set(second_maps.files) and all(
            np.array_equal(first_maps[key], second_maps[key]) for key in first_maps.files
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compared = ("id.jpg", "0_pre.jpg", "0.jpg")
    hashes_equal = {
        name: _digest(args.reference / name) == _digest(args.zero / name)
        for name in compared
    }
    maes = {name: _mae(args.reference / name, args.zero / name) for name in compared}
    map_name = Path("action_binding/frozen_character_maps.npz")
    maps_equal = _maps_equal(args.zero / map_name, args.weak / map_name)
    trace = json.loads((args.weak / "action_binding/action_binding_trace.json").read_text())
    invocations = sum(frame.get("invocations", 0) for frame in trace["frames"])
    locations = {
        location
        for frame in trace["frames"]
        for location in frame.get("locations", {})
    }
    expected_locations = {f"step_{step}:block_{block}" for step in range(1, 41) for block in range(38)}
    complete_attention_window = locations == expected_locations
    final_mae = _mae(args.zero / "0.jpg", args.weak / "0.jpg")
    equivalence_pass = all(hashes_equal.values()) and all(value == 0.0 for value in maes.values())
    mechanical_pass = (
        maps_equal
        and invocations == 38 * 40
        and complete_attention_window
        and final_mae is not None
        and final_mae > 0.0
    )
    payload = {
        "status": "pass" if equivalence_pass and mechanical_pass else "fail",
        "zero_strength_equivalence": {
            "status": "pass" if equivalence_pass else "fail",
            "sha256_equal": hashes_equal,
            "mae": maes,
        },
        "weak_contrastive_checkpoint": {
            "status": "pass" if mechanical_pass else "fail",
            "frozen_maps_exactly_equal": maps_equal,
            "bias_invocations": invocations,
            "expected_bias_invocations": 38 * 40,
            "complete_steps_1_through_40_all_38_blocks": complete_attention_window,
            "final_image_mae": final_mae,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
