"""Audit which CharaConsist consistency components ran for an ablation."""

from __future__ import annotations

import json
from pathlib import Path


CONSISTENCY_MODES = ("prompt_only", "attention_only", "full")


def normalize_consistency_mode(mode, use_interpolate=False):
    """Resolve an explicit mode or preserve the legacy interpolation switch."""
    if mode is None:
        return "full" if use_interpolate else "attention_only"
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in CONSISTENCY_MODES:
        raise ValueError(
            f"Unknown consistency mode {mode!r}; choose from {', '.join(CONSISTENCY_MODES)}."
        )
    return normalized


def new_component_state():
    return {
        "identity_attention_invocations": 0,
        "identity_token_applications": 0,
        "adaptive_merge_invocations": 0,
    }


def finalize_component_state(state, mode, frame):
    """Validate a frame trace against the selected component isolation mode."""
    state = dict(state or new_component_state())
    identity_calls = int(state.get("identity_attention_invocations", 0))
    merge_calls = int(state.get("adaptive_merge_invocations", 0))
    if mode == "prompt_only" and (identity_calls or merge_calls):
        raise RuntimeError("prompt_only unexpectedly invoked a consistency component.")
    if mode == "attention_only" and (identity_calls == 0 or merge_calls):
        raise RuntimeError(
            "attention_only must invoke identity attention and must not invoke adaptive merge."
        )
    if mode == "full" and (identity_calls == 0 or merge_calls == 0):
        raise RuntimeError("full mode must invoke both identity attention and adaptive merge.")
    return {
        "frame": str(frame),
        "identity_attention_invocations": identity_calls,
        "identity_token_applications": int(state.get("identity_token_applications", 0)),
        "adaptive_merge_invocations": merge_calls,
        "status": "PASS",
    }


def save_component_trace(output_dir, mode, seed, frame_traces):
    """Save a prompt-level, machine-readable component isolation trace."""
    trace = {
        "schema_version": 1,
        "consistency_mode": mode,
        "seed": int(seed),
        "status": "PASS",
        "frames": list(frame_traces),
        "identity_attention_invocations": sum(
            item["identity_attention_invocations"] for item in frame_traces
        ),
        "identity_token_applications": sum(
            item["identity_token_applications"] for item in frame_traces
        ),
        "adaptive_merge_invocations": sum(
            item["adaptive_merge_invocations"] for item in frame_traces
        ),
    }
    path = Path(output_dir) / "component_trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(
        "[component-trace] "
        f"mode={mode} seed={seed} frames={len(frame_traces)} "
        f"identity_calls={trace['identity_attention_invocations']} "
        f"merge_calls={trace['adaptive_merge_invocations']} status=PASS",
        flush=True,
    )
    return trace
