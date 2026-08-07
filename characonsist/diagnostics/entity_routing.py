"""Diagnostics for strict C1/C2 identity routing."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from models.entity_routing import validate_entity_labels


def save_entity_label_preflight(image, labels, foreground_mask, output_dir, stage):
    """Save a categorical C1/C2 overlay and mechanical label report."""
    output_dir = Path(output_dir) / "entity_routing"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = validate_entity_labels(labels, foreground_mask)
    label_array = labels.detach().cpu().numpy()[0]
    np.save(output_dir / f"{stage}_entity_labels.npy", label_array)

    colors = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    colors[label_array == 1] = (230, 70, 70)
    colors[label_array == 2] = (70, 120, 230)
    overlay = Image.fromarray(colors).resize(image.size, Image.Resampling.NEAREST)
    Image.blend(image.convert("RGB"), overlay, 0.45).save(
        output_dir / f"{stage}_entity_overlay.png"
    )
    (output_dir / f"{stage}_map_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def validate_entity_routing_trace(id_report, frames, expected_invocations):
    """Validate routing coverage and attention-mask application counts."""
    wrong_allowed = sum(frame.get("wrong_entity_allowed_pairs", 0) for frame in frames)
    frames_valid = bool(frames)
    for frame in frames:
        pre_report = frame.get("pre_run_match_report") or {}
        reports = frame.get("match_reports", [])
        mass_records = frame.get("attention_mass_records")
        mass_records_valid = mass_records is None or (
            len(mass_records) == expected_invocations
            and all(
                record.get("query_head_count", 0) > 0
                and all(math.isfinite(float(record.get(field, math.nan))) for field in (
                    "wrong_entity_mass_mean",
                    "same_entity_mass_mean",
                    "wrong_fraction_of_identity_mass",
                ))
                for record in mass_records
            )
        )
        frames_valid = frames_valid and all((
            frame.get("current_map_report", {}).get("status") == "pass",
            pre_report.get("status") == "pass",
            frame.get("invocations") == expected_invocations,
            bool(reports),
            all(report.get("status") == "pass" for report in reports),
            mass_records_valid,
        ))
    return {
        "status": (
            "pass"
            if id_report.get("status") == "pass" and wrong_allowed == 0 and frames_valid
            else "fail"
        ),
        "wrong_entity_allowed_pairs": int(wrong_allowed),
    }


def summarize_entity_attention_mass(frames):
    """Aggregate per-layer/timestep counterfactual attention-mass records."""
    records = [
        record
        for frame in frames
        for record in frame.get("attention_mass_records", [])
    ]
    query_head_count = sum(record["query_head_count"] for record in records)
    wrong_sum = sum(record["wrong_entity_mass_sum"] for record in records)
    same_sum = sum(record["same_entity_mass_sum"] for record in records)
    identity_sum = wrong_sum + same_sum
    return {
        "available": bool(records),
        "records": len(records),
        "query_head_count": query_head_count,
        "wrong_entity_mass_mean": (
            wrong_sum / query_head_count if query_head_count else 0.0
        ),
        "same_entity_mass_mean": (
            same_sum / query_head_count if query_head_count else 0.0
        ),
        "wrong_fraction_of_identity_mass": (
            wrong_sum / identity_sum if identity_sum > 0.0 else 0.0
        ),
        "max_wrong_entity_mass_mean": max(
            (record["wrong_entity_mass_mean"] for record in records), default=0.0
        ),
    }


def save_entity_routing_trace(
    output_dir, seed, mode, id_report, frames, expected_invocations
):
    """Persist route coverage, correspondence, and wrong-bank-access counts."""
    output_dir = Path(output_dir) / "entity_routing"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_entity_routing_trace(
        id_report, frames, expected_invocations
    )
    payload = {
        "status": validation["status"],
        "seed": int(seed),
        "mode": mode,
        "id_map_report": id_report,
        "wrong_entity_allowed_pairs": validation["wrong_entity_allowed_pairs"],
        "expected_invocations_per_frame": expected_invocations,
        "attention_mass_summary": summarize_entity_attention_mass(frames),
        "frames": frames,
    }
    (output_dir / "entity_routing_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
