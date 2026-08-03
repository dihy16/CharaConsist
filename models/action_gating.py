"""Pure tensor helpers for action-attention gating."""

import torch


def normalize_action_attention(action_attention, foreground_mask, eps=1e-6):
    """Min-max normalize action attention over foreground tokens per image."""
    if action_attention.shape != foreground_mask.shape:
        raise ValueError(
            "action_attention and foreground_mask must have the same shape, "
            f"got {action_attention.shape} and {foreground_mask.shape}."
        )

    normalized = torch.zeros_like(action_attention)
    for batch_index in range(action_attention.shape[0]):
        mask = foreground_mask[batch_index].bool()
        if not torch.any(mask):
            continue

        foreground_values = action_attention[batch_index][mask]
        value_min = foreground_values.min()
        value_range = foreground_values.max() - value_min
        if value_range <= eps:
            continue

        values = (action_attention[batch_index] - value_min) / value_range
        normalized[batch_index][mask] = values[mask].clamp(0, 1)

    return normalized


def action_gated_merge_weights(
    alpha,
    similarities,
    action_scores=None,
    gate_strength=1.0,
):
    """Return per-token merge weights with optional action suppression."""
    if not 0.0 <= gate_strength <= 1.0:
        raise ValueError(
            f"action gate strength must be between 0 and 1, got {gate_strength}."
        )

    weights = similarities * alpha
    if action_scores is not None:
        action_gate = 1.0 - gate_strength * action_scores.clamp(0, 1)
        weights = weights * action_gate
    return weights.clamp(0, 1)


def build_merge_diagnostic_maps(
    alpha,
    similarities,
    action_scores,
    effective_weights,
    current_indices,
    spatial_shape,
    gate_strength,
    eps=1e-6,
):
    """Build dense, auditable maps from the weights actually used by AdaToMe."""
    if not 0.0 <= gate_strength <= 1.0:
        raise ValueError(
            f"action gate strength must be between 0 and 1, got {gate_strength}."
        )
    similarities = similarities.reshape(-1)
    action_scores = action_scores.reshape(-1).clamp(0, 1)
    effective_weights = effective_weights.reshape(-1)
    current_indices = current_indices.reshape(-1).long()
    lengths = {
        similarities.numel(),
        action_scores.numel(),
        effective_weights.numel(),
        current_indices.numel(),
    }
    if len(lengths) != 1:
        raise ValueError("Merge diagnostic inputs must contain the same number of tokens.")

    height, width = map(int, spatial_shape)
    token_count = height * width
    if current_indices.numel() and (
        current_indices.min().item() < 0 or current_indices.max().item() >= token_count
    ):
        raise ValueError("A merge diagnostic index is outside the spatial feature grid.")

    base_weights = action_gated_merge_weights(alpha, similarities, gate_strength=0.0)
    expected_effective = action_gated_merge_weights(
        alpha,
        similarities,
        action_scores,
        gate_strength=gate_strength,
    )
    if not torch.isfinite(effective_weights).all():
        raise ValueError("Effective merge weights must be finite.")
    if not torch.allclose(effective_weights, expected_effective, atol=eps, rtol=eps):
        raise RuntimeError("Applied merge weights do not match the action-gating formula.")
    if torch.any(effective_weights < -eps) or torch.any(effective_weights > base_weights + eps):
        raise RuntimeError("Effective merge weights must remain within [0, baseline].")

    suppressed_weights = (base_weights - effective_weights).clamp_min(0)
    gate_factors = 1.0 - gate_strength * action_scores
    maps = {}
    for name, values in (
        ("similarities", similarities),
        ("action_scores", action_scores),
        ("gate_factors", gate_factors),
        ("base_weights", base_weights),
        ("effective_weights", effective_weights),
        ("suppressed_weights", suppressed_weights),
    ):
        dense = torch.zeros(token_count, device=values.device, dtype=torch.float32)
        if current_indices.numel():
            dense[current_indices] = values.float()
        maps[name] = dense.reshape(height, width).cpu()

    valid_mask = torch.zeros(token_count, device=current_indices.device, dtype=torch.bool)
    if current_indices.numel():
        valid_mask[current_indices] = True
    maps["valid_mask"] = valid_mask.reshape(height, width).cpu()
    return maps
