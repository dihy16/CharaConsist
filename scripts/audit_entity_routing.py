#!/usr/bin/env python3
"""Audit compatibility controls and hard-routing mechanics across the 2x2 pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from characonsist.experiments.conditions import action_binding_label
from characonsist.diagnostics.entity_routing import validate_entity_routing_trace


LEGACY_EXPECTED_INVOCATIONS = 38 * 40
LEGACY_FOREGROUND_ROUTING_INVOCATIONS = 19 * 40


def _validate_trace(trace):
    """Validate current traces and the 38-layer legacy trace-count bug."""
    expected = trace.get("expected_invocations_per_frame")
    if expected == LEGACY_EXPECTED_INVOCATIONS:
        expected = LEGACY_FOREGROUND_ROUTING_INVOCATIONS
    return validate_entity_routing_trace(
        trace.get("id_map_report", {}), trace.get("frames", []), expected
    )


def _mae(first, second):
    first_pixels = np.asarray(Image.open(first).convert("RGB"), dtype=np.float32)
    second_pixels = np.asarray(Image.open(second).convert("RGB"), dtype=np.float32)
    if first_pixels.shape != second_pixels.shape:
        return None
    return float(np.abs(first_pixels - second_pixels).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results_colab"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparisons/entity_routing/2b_final_action_binding/mechanical_audit.json"),
    )
    parser.add_argument("--prompt", default="2b_final_action_binding")
    args = parser.parse_args()

    records = []
    overall = True
    for beta, gamma in ((0.0, 0.0), (1.0, 0.5)):
        label = action_binding_label(beta, gamma)
        for seed in range(2025, 2030):
            suffix = Path("bg_fg") / args.prompt / "prompt_0"
            legacy = (
                args.results_root / "action_binding_ablation" / label
                / f"seed_{seed}" / suffix
            )
            off = (
                args.results_root / "entity_routing_ablation" / "routing_off" / label
                / f"seed_{seed}" / suffix
            )
            hard = (
                args.results_root / "entity_routing_ablation" / "routing_hard" / label
                / f"seed_{seed}" / suffix
            )
            record = {"beta": beta, "gamma": gamma, "seed": seed}
            try:
                record["off_vs_legacy_mae"] = _mae(off / "0.jpg", legacy / "0.jpg")
                record["hard_vs_off_mae"] = _mae(hard / "0.jpg", off / "0.jpg")
                trace = json.loads(
                    (hard / "entity_routing" / "entity_routing_trace.json").read_text(
                        encoding="utf-8"
                    )
                )
                trace_validation = _validate_trace(trace)
                record["trace_status"] = trace_validation["status"]
                record["wrong_entity_allowed_pairs"] = trace.get(
                    "wrong_entity_allowed_pairs"
                )
                record["attention_mass_summary"] = trace.get(
                    "attention_mass_summary", {"available": False}
                )
                record["pass"] = all((
                    record["off_vs_legacy_mae"] == 0.0,
                    record["hard_vs_off_mae"] is not None,
                    record["hard_vs_off_mae"] > 0.0,
                    record["trace_status"] == "pass",
                    record["wrong_entity_allowed_pairs"] == 0,
                ))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                record.update({"pass": False, "error": f"{type(exc).__name__}: {exc}"})
            overall = overall and record["pass"]
            records.append(record)

    payload = {"status": "pass" if overall else "fail", "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
