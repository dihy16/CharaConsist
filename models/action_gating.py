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
