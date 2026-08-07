"""Pure tensor operations for character-conditioned action binding."""

import math

import torch


def normalize_character_map(values, foreground_mask, eps=1e-6):
    """Independently min-max normalize one character map inside foreground."""
    values = values.float()
    mask = foreground_mask.to(device=values.device, dtype=values.dtype)
    flat = values.flatten(1)
    flat_mask = mask.flatten(1).bool()
    masked_min = torch.where(flat_mask, flat, torch.inf).amin(dim=1, keepdim=True)
    masked_max = torch.where(flat_mask, flat, -torch.inf).amax(dim=1, keepdim=True)
    valid = torch.isfinite(masked_min) & torch.isfinite(masked_max)
    scale = (masked_max - masked_min).clamp_min(eps)
    normalized = torch.where(valid, (flat - masked_min) / scale, torch.zeros_like(flat))
    return (normalized.clamp(0.0, 1.0) * flat_mask).view_as(values)


def build_character_maps(character_attention, foreground_mask):
    """Freeze normalized C1/C2 spatial maps from unbiased pre-run attention."""
    return {
        entity_id: normalize_character_map(values, foreground_mask).detach().clone()
        for entity_id, values in character_attention.items()
    }


def validate_character_maps(character_maps, foreground_mask, eps=1e-6):
    """Return a serializable mechanical preflight report for C1/C2 maps."""
    required = ("1", "2")
    report = {"status": "pass", "maps": {}}
    mask = foreground_mask.bool()
    for entity_id in required:
        values = character_maps.get(entity_id)
        if values is None:
            report["status"] = "fail"
            report["maps"][entity_id] = {"error": "missing"}
            continue
        finite = bool(torch.isfinite(values).all().item())
        support = int(((values > eps) & mask).sum().item())
        dynamic_range = float((values[mask].max() - values[mask].min()).item()) if mask.any() else 0.0
        report["maps"][entity_id] = {
            "finite": finite,
            "foreground_support": support,
            "dynamic_range": dynamic_range,
        }
        if not finite or support == 0 or dynamic_range <= eps:
            report["status"] = "fail"
    if all(entity_id in character_maps for entity_id in required):
        identical = bool(torch.allclose(character_maps["1"], character_maps["2"], atol=eps, rtol=0))
        report["maps_identical"] = identical
        if identical:
            report["status"] = "fail"
    return report


def apply_character_action_bias(
    attention_mask,
    text_seq_len,
    visual_seq_len,
    action_spans,
    character_maps,
    beta,
    gamma,
    query,
    key,
    trace_state=None,
    timestep_ind=None,
    block_index=None,
):
    """Bias A1 toward C1 and A2 toward C2 for visual-to-text attention only."""
    beta = float(beta)
    gamma = float(gamma)
    if beta == 0.0 and gamma == 0.0:
        return attention_mask
    if not math.isfinite(beta) or not math.isfinite(gamma) or beta < 0.0 or gamma < 0.0:
        raise ValueError("Action-binding beta and gamma must be finite and non-negative.")
    if not action_spans or not character_maps:
        return attention_mask

    batch_size = query.shape[0]
    query_len = query.shape[-2]
    key_len = key.shape[-2]
    if attention_mask is None:
        biased = torch.zeros(
            (batch_size, 1, query_len, key_len), device=query.device, dtype=query.dtype
        )
    else:
        biased = attention_mask.clone()
    query_slice = slice(text_seq_len, text_seq_len + visual_seq_len)
    applied = {}
    extrema = []
    for entity_id in ("1", "2"):
        if entity_id not in action_spans or entity_id not in character_maps:
            continue
        other_id = "2" if entity_id == "1" else "1"
        start, end = action_spans[entity_id]
        start = max(0, min(int(start), text_seq_len))
        end = max(start, min(int(end), text_seq_len))
        if start == end:
            continue
        correct = character_maps[entity_id].to(device=query.device, dtype=query.dtype).flatten(1)
        if correct.shape[-1] != visual_seq_len:
            raise ValueError(
                f"Character map C{entity_id} has {correct.shape[-1]} positions; "
                f"expected {visual_seq_len}."
            )
        wrong = character_maps.get(other_id)
        if wrong is None:
            wrong = torch.zeros_like(correct)
        else:
            wrong = wrong.to(device=query.device, dtype=query.dtype).flatten(1)
        addition = (beta * correct - gamma * wrong)[:, None, :, None]
        biased[:, :, query_slice, start:end] = biased[:, :, query_slice, start:end] + addition
        extrema.append(addition.float())
        applied[entity_id] = visual_seq_len * (end - start)

    if trace_state is not None and applied:
        trace_state["invocations"] = trace_state.get("invocations", 0) + 1
        counts = trace_state.setdefault("biased_pairs", {})
        for entity_id, count in applied.items():
            counts[entity_id] = counts.get(entity_id, 0) + count
        values = torch.cat([item.flatten() for item in extrema])
        trace_state["min_bias"] = min(trace_state.get("min_bias", math.inf), float(values.min().item()))
        trace_state["max_bias"] = max(trace_state.get("max_bias", -math.inf), float(values.max().item()))
        location = f"step_{timestep_ind}:block_{block_index}"
        locations = trace_state.setdefault("locations", {})
        locations[location] = locations.get(location, 0) + 1
    return biased
