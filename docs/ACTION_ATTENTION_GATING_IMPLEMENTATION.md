# Action-Attention Gating Report

## Outcome

CharaConsist already accepts action text through its three-part prompt format:

```text
background # character # action
```

The implementation therefore does not add another prompt field or introduce
learned action tokens. It tracks the existing action field as a cumulative T5
token span and uses its image-attention map as a continuous gate on adaptive
token merge.

This is a training-free change. It does not load OneHOI weights, alter FLUX
RoPE, or replace CharaConsist's point-tracking attention mask.

## Previous behavior

The runner previously retained only `bg_len` and `real_len`. The custom
attention processor consequently divided text into:

```text
background | character + action
```

The action words contributed to foreground-mask extraction but could not be
measured independently. Adaptive token merge used only the scheduled merge
strength and current-to-identity similarity:

```text
merge_weight_j = alpha_schedule * similarity_j
```

Identity features could therefore be injected strongly into hands, pose,
contact regions, and manipulated objects even when those regions needed to
change to satisfy the frame action.

## Implemented data flow

1. The runner builds the same concatenated prompt and measures three cumulative
   boundaries: `background_end`, `action_start`, and `real_end`.
2. Each participating attention processor retains the existing background and
   foreground maps and additionally averages attention over
   `[action_start:real_end)`.
3. The pipeline averages action attention across the same layers used for mask
   extraction.
4. Action attention is min-max normalized over current foreground tokens.
   Background locations remain zero. Empty, truncated, missing, or constant
   action maps also become zero so the original merge is preserved.
5. For every matched current foreground token, adaptive merge now uses:

```text
action_gate_j = 1 - action_gate_strength * action_score_j
merge_weight_j = alpha_schedule * similarity_j * action_gate_j
```

At the default strength of `1`, a normalized score of `1` suppresses explicit
adaptive merging at that token. A score of `0` retains the original merge.
`--action_gate_strength 0` provides a baseline-compatible ablation.

## Effect on point tracking

Point correspondence, identity K/V selection, foreground/background routing,
and the additive `0/-inf` attention mask are unchanged. The action signal acts
after scaled-dot-product attention, specifically on the explicit adaptive
token merge.

This preserves the lower-risk first step requested for the experiment:

- identity point matches remain available;
- identity consistency remains strong in low-action regions;
- action-sensitive regions receive less direct identity hidden-state blending;
- no new attention-mask topology or backend behavior is introduced.

Because identity K/V can still influence the preceding point-tracking
attention, this gate does not remove every identity contribution from
high-action regions. If evaluation shows that merge-only gating is
insufficient, earlier attention gating should be treated as a separate
experiment using a finite per-query logit bias. The existing `0/-inf` mask
must not be multiplied by the action gate because doing so is ineffective for
allowed entries and can create undefined `0 * -inf` values.

## Why OneHOI's HOI attention mask is unnecessary

OneHOI introduces separately encoded subject, object, and action grounding
tokens. Its structured mask uses role and instance metadata plus boxes or
arbitrary masks to control which HOI and image tokens may communicate. Those
tokens and that topology are part of a trained OneHOI architecture.

CharaConsist's action text is already present in the normal FLUX prompt, and
the requested behavior only needs a continuous spatial signal for merge
plasticity. Porting OneHOI's structured mask would require new role-aware
inputs, grounding representations, spatial layouts, and compatible trained
parameters without being necessary for the chosen gate.

## Controls and compatibility

- Prompt files remain unchanged.
- `--use_interpolate` is still required for adaptive token merge and therefore
  for action gating to have an effect.
- `--action_gate_strength` accepts values from `0` through `1` and defaults to
  `1`.
- Direct or legacy calls that use `set_text_len()` receive an empty action span
  and retain baseline merge behavior.
- The standalone `point_and_mask/` demonstration and notebooks are not part of
  the production inference path and were not changed.

## Validation

CPU unit coverage verifies cumulative token boundaries, truncation fallback,
foreground-only action normalization, neutral degenerate maps, baseline
equivalence at strength `0`, full suppression at maximum action attention,
and invalid strength rejection.

End-to-end visual validation still requires local FLUX.1-dev weights and CUDA.
Use the same prompt, seed, dimensions, and interpolation settings for:

```text
--action_gate_strength 0
--action_gate_strength 0.5
--action_gate_strength 1
```

Compare character identity outside the action region, pose/contact compliance,
hand and manipulated-object geometry, foreground masks, and temporal
consistency across frames.
