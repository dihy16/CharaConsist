"""Serialize and audit action-gated adaptive-merge diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
MAP_NAMES = (
    "similarities",
    "action_scores",
    "gate_factors",
    "base_weights",
    "effective_weights",
    "suppressed_weights",
    "valid_mask",
)


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _step_summary(frame, step, alpha, invocation_count, maps, eps=1e-8):
    valid = _as_numpy(maps["valid_mask"]).astype(bool)
    base = _as_numpy(maps["base_weights"]).astype(np.float32)
    effective = _as_numpy(maps["effective_weights"]).astype(np.float32)
    suppressed = _as_numpy(maps["suppressed_weights"]).astype(np.float32)
    action = _as_numpy(maps["action_scores"]).astype(np.float32)
    if not all(np.isfinite(array).all() for array in (base, effective, suppressed, action)):
        raise RuntimeError("Merge diagnostic arrays must contain only finite values.")
    tolerance = 1e-6
    if np.any(effective < -tolerance) or np.any(effective > base + tolerance):
        raise RuntimeError("Effective merge weights must remain within [0, baseline].")

    valid_base = base[valid]
    valid_effective = effective[valid]
    valid_suppressed = suppressed[valid]
    valid_action = action[valid]
    modified = valid_suppressed > eps
    return {
        "schema_version": SCHEMA_VERSION,
        "frame": str(frame),
        "step": int(step),
        "alpha": float(alpha),
        "attention_layer_invocations": int(invocation_count),
        "matched_tokens": int(valid.sum()),
        "modified_tokens": int(modified.sum()),
        "action_mean": float(valid_action.mean()) if valid_action.size else 0.0,
        "base_mean": float(valid_base.mean()) if valid_base.size else 0.0,
        "effective_mean": float(valid_effective.mean()) if valid_effective.size else 0.0,
        "suppression_mean": float(valid_suppressed.mean()) if valid_suppressed.size else 0.0,
        "suppression_max": float(valid_suppressed.max()) if valid_suppressed.size else 0.0,
        "suppression_sum": float(valid_suppressed.sum()),
    }


def save_frame_merge_diagnostics(state, output_dir, frame, gate_strength, eps=1e-8):
    """Save exact step maps and return a compact frame-level audit."""
    output_dir = Path(output_dir)
    diagnostics_dir = output_dir / "merge_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    records = state.get("records", {}) if state else {}
    invocations = state.get("invocations", {}) if state else {}
    steps = sorted(records)
    summaries = []

    arrays = {name: [] for name in MAP_NAMES}
    alphas = []
    for step in steps:
        record = records[step]
        maps = record["maps"]
        summaries.append(
            _step_summary(frame, step, record["alpha"], invocations.get(step, 0), maps, eps)
        )
        alphas.append(float(record["alpha"]))
        for name in MAP_NAMES:
            arrays[name].append(_as_numpy(maps[name]))

    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "gate_strength": np.asarray(gate_strength, dtype=np.float32),
        "step_indices": np.asarray(steps, dtype=np.int32),
        "alphas": np.asarray(alphas, dtype=np.float32),
    }
    for name, values in arrays.items():
        dtype = np.bool_ if name == "valid_mask" else np.float32
        payload[name] = np.asarray(values, dtype=dtype)
    np.savez_compressed(diagnostics_dir / f"{frame}_weights.npz", **payload)

    trace_path = output_dir / "action_gate_trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as stream:
        for summary in summaries:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")

    matched = sum(item["matched_tokens"] for item in summaries)
    modified = sum(item["modified_tokens"] for item in summaries)
    suppression_sum = sum(item["suppression_sum"] for item in summaries)
    suppression_max = max((item["suppression_max"] for item in summaries), default=0.0)
    invocation_count = sum(item["attention_layer_invocations"] for item in summaries)
    if gate_strength == 0:
        if suppression_sum > eps or modified:
            raise RuntimeError("Lambda zero changed merge weights; baseline control is invalid.")
        status = "CONTROL"
    elif modified:
        status = "PASS"
    else:
        status = "WARN"
    audit = {
        "schema_version": SCHEMA_VERSION,
        "frame": str(frame),
        "gate_strength": float(gate_strength),
        "status": status,
        "steps_recorded": len(steps),
        "attention_layer_invocations": invocation_count,
        "matched_token_applications": matched,
        "modified_token_applications": modified,
        "suppression_sum": float(suppression_sum),
        "suppression_max": float(suppression_max),
    }
    print(
        "[action-gate] "
        f"frame={frame} lambda={gate_strength:g} steps={len(steps)} "
        f"matched={matched} modified={modified} "
        f"suppression_max={suppression_max:.6f} status={status}",
        flush=True,
    )
    return audit


def save_prompt_gate_audit(output_dir, frame_audits, gate_strength, seed):
    """Save and print the aggregate prompt-level audit."""
    output_dir = Path(output_dir)
    statuses = [item["status"] for item in frame_audits]
    if "WARN" in statuses:
        status = "WARN"
    elif gate_strength == 0:
        status = "CONTROL"
    else:
        status = "PASS"
    audit = {
        "schema_version": SCHEMA_VERSION,
        "gate_strength": float(gate_strength),
        "seed": int(seed),
        "status": status,
        "frames": frame_audits,
        "steps_recorded": sum(item["steps_recorded"] for item in frame_audits),
        "attention_layer_invocations": sum(
            item["attention_layer_invocations"] for item in frame_audits
        ),
        "matched_token_applications": sum(
            item["matched_token_applications"] for item in frame_audits
        ),
        "modified_token_applications": sum(
            item["modified_token_applications"] for item in frame_audits
        ),
        "suppression_sum": sum(item["suppression_sum"] for item in frame_audits),
        "suppression_max": max(
            (item["suppression_max"] for item in frame_audits), default=0.0
        ),
    }
    (output_dir / "action_gate_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[action-gate-summary] "
        f"lambda={gate_strength:g} frames={len(frame_audits)} "
        f"modified={audit['modified_token_applications']} status={status}",
        flush=True,
    )
    return audit
