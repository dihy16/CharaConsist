"""Pure tensor helpers for the role-action attention-bias ablation."""

import torch


ROLE_TO_MAP = {
    "subject": "subject",
    "predicate": "interaction",
    "object": "object",
    "recipient": "recipient",
}


def _normalize_in_mask(values, foreground_mask, eps=1e-6):
    values = values.float()
    mask = foreground_mask.to(device=values.device, dtype=values.dtype)
    flat = values.flatten(1)
    flat_mask = mask.flatten(1)
    masked_min = torch.where(flat_mask.bool(), flat, torch.inf).amin(dim=1, keepdim=True)
    masked_min = torch.where(torch.isfinite(masked_min), masked_min, torch.zeros_like(masked_min))
    shifted = (flat - masked_min).clamp_min(0.0) * flat_mask
    scale = shifted.amax(dim=1, keepdim=True).clamp_min(eps)
    return (shifted / scale).view_as(values) * mask


def build_soft_role_maps(role_attention, foreground_mask, eps=1e-6):
    """Turn raw role attention into foreground-localized soft routing maps."""
    if "subject" not in role_attention or "object" not in role_attention:
        return {}
    subject_score = _normalize_in_mask(role_attention["subject"], foreground_mask, eps)
    object_map = _normalize_in_mask(role_attention["object"], foreground_mask, eps)
    predicate_map = _normalize_in_mask(
        role_attention.get("predicate", role_attention["subject"]), foreground_mask, eps
    )
    maps = {"object": object_map, "predicate": predicate_map}

    if "recipient" in role_attention:
        recipient_score = _normalize_in_mask(role_attention["recipient"], foreground_mask, eps)
        total = (subject_score + recipient_score).clamp_min(eps)
        mask = foreground_mask.to(subject_score.dtype)
        maps["subject"] = subject_score / total * mask
        maps["recipient"] = recipient_score / total * mask
    else:
        maps["subject"] = subject_score

    maps["interaction"] = (
        1.0 - (1.0 - maps["subject"]) * (1.0 - object_map)
    ) * foreground_mask.to(subject_score.dtype)
    return maps


def apply_role_attention_bias(
    attention_mask,
    text_seq_len,
    visual_seq_len,
    role_spans,
    role_maps,
    strength,
    trace_state=None,
):
    """Add role-localized finite bias from visual queries to role text keys."""
    if strength == 0.0 or not role_spans or not role_maps:
        return attention_mask
    if strength < 0.0:
        raise ValueError("role-action bias strength must be non-negative.")

    biased = attention_mask.clone()
    query_slice = slice(text_seq_len, text_seq_len + visual_seq_len)
    applied = {}
    max_bias = 0.0
    for role, span in role_spans.items():
        map_name = ROLE_TO_MAP.get(role)
        if map_name is None or map_name not in role_maps:
            continue
        start, end = span
        start = max(0, min(int(start), text_seq_len))
        end = max(start, min(int(end), text_seq_len))
        if start == end:
            continue
        spatial = role_maps[map_name].to(device=biased.device, dtype=biased.dtype)
        spatial = spatial.reshape(spatial.shape[0], -1)
        if spatial.shape[-1] != visual_seq_len:
            raise ValueError(
                f"Role map has {spatial.shape[-1]} positions; expected {visual_seq_len}."
            )
        addition = spatial[:, None, :, None] * float(strength)
        biased[:, :, query_slice, start:end] = (
            biased[:, :, query_slice, start:end] + addition
        )
        applied[role] = int(visual_seq_len * (end - start))
        max_bias = max(max_bias, float(addition.float().amax().item()))

    if trace_state is not None and applied:
        trace_state["invocations"] = trace_state.get("invocations", 0) + 1
        pair_counts = trace_state.setdefault("biased_pairs", {})
        for role, count in applied.items():
            pair_counts[role] = pair_counts.get(role, 0) + count
        trace_state["max_bias"] = max(trace_state.get("max_bias", 0.0), max_bias)
    return biased
