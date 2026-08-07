"""Pure helpers for strict two-entity identity routing."""

import torch

from .action_binding import validate_character_maps


ENTITY_IDS = ("1", "2")


def should_build_entity_labels(entity_routing_mode, timestep_ind, routing_step):
    """Return whether routing labels should be frozen at this denoising step."""
    return entity_routing_mode == "hard" and timestep_ind == routing_step


def should_apply_entity_matching(
    entity_routing_mode, is_pre_run, timestep_ind, routing_step
):
    """Return whether labels are ready for entity-restricted matching."""
    return entity_routing_mode == "hard" and (
        not is_pre_run or timestep_ind == routing_step
    )


def build_entity_labels(character_maps, foreground_mask):
    """Assign every foreground token to the stronger normalized C1/C2 map."""
    report = validate_character_maps(character_maps, foreground_mask)
    if report["status"] != "pass":
        raise ValueError(f"Invalid C1/C2 maps for entity routing: {report}")
    stacked = torch.stack(
        [character_maps[entity_id].float() for entity_id in ENTITY_IDS], dim=1
    )
    labels = stacked.argmax(dim=1).to(torch.long) + 1
    return labels * foreground_mask.to(device=labels.device, dtype=torch.long)


def validate_entity_labels(labels, foreground_mask):
    """Return entity-token counts and reject missing or out-of-mask labels."""
    labels = labels.to(torch.long)
    foreground = foreground_mask.to(device=labels.device, dtype=torch.bool)
    counts = {
        entity_id: int(((labels == int(entity_id)) & foreground).sum().item())
        for entity_id in ENTITY_IDS
    }
    outside = int(((labels != 0) & ~foreground).sum().item())
    status = "pass" if outside == 0 and all(counts.values()) else "fail"
    return {
        "status": status,
        "entity_token_counts": counts,
        "labeled_background_tokens": outside,
    }


def get_entity_restricted_matches(cross_sim, id_labels, curr_labels, sim_threshold):
    """Match current tokens only against identity tokens with the same owner."""
    if cross_sim.ndim != 3 or cross_sim.shape[0] != 1:
        raise ValueError("Entity routing currently requires a batch size of one.")
    id_labels = id_labels.reshape(-1).to(device=cross_sim.device, dtype=torch.long)
    curr_labels = curr_labels.reshape(-1).to(device=cross_sim.device, dtype=torch.long)
    if cross_sim.shape[1:] != (curr_labels.numel(), id_labels.numel()):
        raise ValueError(
            "Cross-similarity shape must match current and identity entity-label maps."
        )

    full_max = torch.zeros(
        (1, curr_labels.numel()), device=cross_sim.device, dtype=cross_sim.dtype
    )
    full_argmax = torch.zeros(
        (1, curr_labels.numel()), device=cross_sim.device, dtype=torch.long
    )
    matched_id = []
    matched_curr = []
    report = {"status": "pass", "entities": {}}
    for entity_id in (1, 2):
        curr_indices = torch.nonzero(curr_labels == entity_id, as_tuple=True)[0]
        id_indices = torch.nonzero(id_labels == entity_id, as_tuple=True)[0]
        if curr_indices.numel() == 0 or id_indices.numel() == 0:
            raise ValueError(f"Entity C{entity_id} has no routable visual tokens.")
        similarities = cross_sim[0].index_select(0, curr_indices).index_select(1, id_indices)
        values, local_indices = similarities.max(dim=1)
        valid = values > float(sim_threshold)
        valid_curr = curr_indices[valid]
        valid_id = id_indices[local_indices[valid]]
        if valid_curr.numel() == 0:
            raise ValueError(
                f"Entity C{entity_id} has no matches above similarity threshold {sim_threshold}."
            )
        full_max[0, valid_curr] = values[valid]
        full_argmax[0, valid_curr] = valid_id
        matched_curr.append(valid_curr)
        matched_id.append(valid_id)
        report["entities"][str(entity_id)] = {
            "current_tokens": int(curr_indices.numel()),
            "identity_tokens": int(id_indices.numel()),
            "valid_matches": int(valid_curr.numel()),
            "min_similarity": float(values[valid].min().item()),
            "max_similarity": float(values[valid].max().item()),
        }

    id_indices = torch.cat(matched_id)
    curr_indices = torch.cat(matched_curr)
    report["total_valid_matches"] = int(curr_indices.numel())
    return id_indices, curr_indices, full_max, full_argmax, report


def build_entity_identity_masks(base_mask, appended_entity_labels, curr_entity_labels):
    """Return newly blocked wrong-owner pairs and allowed same-owner pairs."""
    appended = appended_entity_labels.reshape(-1).to(base_mask.device, dtype=torch.long)
    current = curr_entity_labels.reshape(-1).to(base_mask.device, dtype=torch.long)
    if base_mask.shape[-1] != appended.numel() or base_mask.shape[-2] != current.numel():
        raise ValueError("Entity labels do not align with the expanded attention-mask block.")
    foreground_keys = appended.view(1, 1, 1, -1) > 0
    current_owner = current.view(1, 1, -1, 1)
    key_owner = appended.view(1, 1, 1, -1)
    if base_mask.dtype == torch.bool:
        base_allowed = ~base_mask
    else:
        base_allowed = torch.isfinite(base_mask)
    wrong_owner = foreground_keys & (current_owner != key_owner)
    same_owner = foreground_keys & (current_owner == key_owner)
    return wrong_owner & base_allowed, same_owner & base_allowed


def measure_entity_attention_mass(
    query,
    key,
    attention_mask,
    wrong_entity_mask,
    same_entity_mask,
    query_chunk_size=64,
):
    """Measure counterfactual pre-routing attention mass in bounded chunks."""
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("Entity attention-mass diagnostics require batch size one.")
    if attention_mask is None or not attention_mask.dtype.is_floating_point:
        raise ValueError("Entity attention-mass diagnostics require an additive mask.")
    if wrong_entity_mask.shape != same_entity_mask.shape:
        raise ValueError("Wrong- and same-entity masks must have matching shapes.")
    if wrong_entity_mask.shape[-2:] != (query.shape[-2], key.shape[-2]):
        raise ValueError("Entity attention masks must align with query and key lengths.")

    eligible_queries = (wrong_entity_mask | same_entity_mask).any(dim=-1)[0, 0]
    query_indices = torch.nonzero(eligible_queries, as_tuple=True)[0]
    if query_indices.numel() == 0:
        return {
            "wrong_entity_mass_sum": 0.0,
            "same_entity_mass_sum": 0.0,
            "query_head_count": 0,
            "wrong_entity_mass_mean": 0.0,
            "same_entity_mass_mean": 0.0,
            "wrong_fraction_of_identity_mass": 0.0,
        }

    wrong_sum = torch.zeros((), device=query.device, dtype=torch.float32)
    same_sum = torch.zeros((), device=query.device, dtype=torch.float32)
    scale = query.shape[-1] ** -0.5
    key_transposed = key.transpose(-2, -1)
    for start in range(0, query_indices.numel(), query_chunk_size):
        indices = query_indices[start:start + query_chunk_size]
        query_chunk = query.index_select(-2, indices)
        wrong_chunk = wrong_entity_mask.index_select(-2, indices)
        same_chunk = same_entity_mask.index_select(-2, indices)
        counterfactual_mask = attention_mask.index_select(-2, indices).float()
        counterfactual_mask = counterfactual_mask.masked_fill(wrong_chunk, 0.0)
        logits = torch.matmul(query_chunk, key_transposed).float() * scale
        probabilities = torch.softmax(logits + counterfactual_mask, dim=-1)
        wrong_sum = wrong_sum + (probabilities * wrong_chunk).sum()
        same_sum = same_sum + (probabilities * same_chunk).sum()

    query_head_count = int(query_indices.numel() * query.shape[1])
    wrong_sum = float(wrong_sum.item())
    same_sum = float(same_sum.item())
    identity_sum = wrong_sum + same_sum
    return {
        "wrong_entity_mass_sum": wrong_sum,
        "same_entity_mass_sum": same_sum,
        "query_head_count": query_head_count,
        "wrong_entity_mass_mean": wrong_sum / query_head_count,
        "same_entity_mass_mean": same_sum / query_head_count,
        "wrong_fraction_of_identity_mass": (
            wrong_sum / identity_sum if identity_sum > 0.0 else 0.0
        ),
    }


def apply_entity_identity_mask(
    base_mask,
    appended_entity_labels,
    curr_entity_labels,
    trace=None,
    entity_masks=None,
):
    """Block every cross-entity query-to-appended-identity pair."""
    wrong_entity, same_entity = entity_masks or build_entity_identity_masks(
        base_mask, appended_entity_labels, curr_entity_labels
    )
    if base_mask.dtype == torch.bool:
        routed = base_mask | wrong_entity
        finite_or_allowed = ~routed
    else:
        routed = base_mask.masked_fill(wrong_entity, float("-inf"))
        finite_or_allowed = torch.isfinite(routed)

    if trace is not None:
        allowed = same_entity & finite_or_allowed
        wrong_allowed = wrong_entity & finite_or_allowed
        trace["invocations"] = trace.get("invocations", 0) + 1
        trace["allowed_same_entity_pairs"] = trace.get(
            "allowed_same_entity_pairs", 0
        ) + int(allowed.sum().item())
        trace["blocked_cross_entity_pairs"] = trace.get(
            "blocked_cross_entity_pairs", 0
        ) + int(wrong_entity.sum().item())
        trace["wrong_entity_allowed_pairs"] = trace.get(
            "wrong_entity_allowed_pairs", 0
        ) + int(wrong_allowed.sum().item())
    return routed
