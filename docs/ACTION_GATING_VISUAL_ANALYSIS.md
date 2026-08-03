# Action-Gating Visual Analysis

**Date:** 2026-08-03  
**Runs reviewed:** seed `2025`; `1a_anchor_verb`, `2b_transfer`, and
`3_large_pose_change`.

## Executive summary

The action-gating variant affects final generations. Increasing lambda changes
nearly all output pixels, and the change magnitude rises from `lambda=0.5` to
`lambda=1` on the two prompts with a baseline. This is strong evidence that
the lambda-controlled path influences the final denoising pass.

This is **not yet evidence that the variant improves action following**.
Identity and broad scene composition remain stable, but important action
failures remain at every lambda: throwing and tearing are weak, mug ownership
and actor roles are wrong in the transfer story, and difficult poses remain
inaccurate. The variant is currently a controllable identity-plasticity
ablation, not a validated solution for action fidelity.

The exact merge diagnostics were added after these experiments and are absent
from the downloaded results. This report can establish output-level effects and
inspect saved attention, mask, and point controls, but cannot show the per-step
merge weights actually suppressed during these legacy runs.

## Evidence reviewed

| Prompt | Comparison directory | Conditions |
| --- | --- | --- |
| `1a_anchor_verb` | `results_colab/comparisons/1a_anchor_verb/seed_2025/` | lambda 0, 0.5, 1 |
| `2b_transfer` | `results_colab/comparisons/2b_transfer/seed_2025/` | lambda 0, 0.5, 1 |
| `3_large_pose_change` | `results_colab/comparisons/3_large_pose_change/seed_2025/` | lambda 0.5, 1; lambda 0 missing |

`3_large_pose_change` has no lambda-zero output because that run failed. Its
difference map correctly uses lambda 0.5 as its reference rather than silently
treating the missing baseline as an image.

### Output-level response to lambda

The table aggregates four story frames from each `comparison_audit.json`. MAE
is mean absolute RGB difference on the 0--255 scale; changed pixels have at
least one changed RGB channel.

| Prompt | Comparison | Mean MAE | Mean changed pixels |
| --- | --- | ---: | ---: |
| `1a_anchor_verb` | lambda 0.5 vs 0 | 3.080 | 85.4% |
| `1a_anchor_verb` | lambda 1 vs 0 | 5.003 | 93.3% |
| `2b_transfer` | lambda 0.5 vs 0 | 4.345 | 87.1% |
| `2b_transfer` | lambda 1 vs 0 | 5.409 | 90.9% |
| `3_large_pose_change` | lambda 1 vs 0.5 | 2.650 | 77.0% |

The response is monotonic in the two complete sweeps: lambda 1 differs more
from lambda 0 than lambda 0.5 does. Difference maps are spatially broad, not
limited to hands or props. This is expected in iterative diffusion: a local
identity-merge change can alter later denoising globally. Changed-pixel
percentage is evidence of sensitivity to lambda, not a gate-localization
metric.

### Controls: masks, action maps, and point matches

Foreground masks and action scores remain highly similar across lambda. Mask
IoU is computed from the saved mask-only panel; action-map correlation is from
the normalized score arrays.

| Prompt | Comparison | Mean mask IoU | Minimum IoU | Mean action correlation | Mean action MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| `1a_anchor_verb` | lambda 0.5 vs 0 | 0.9967 | 0.9936 | 0.9970 | 0.0056 |
| `1a_anchor_verb` | lambda 1 vs 0 | 0.9937 | 0.9863 | 0.9924 | 0.0094 |
| `2b_transfer` | lambda 0.5 vs 0 | 0.9693 | 0.9435 | 0.9885 | 0.0067 |
| `2b_transfer` | lambda 1 vs 0 | 0.9553 | 0.8842 | 0.9805 | 0.0100 |
| `3_large_pose_change` | lambda 1 vs 0.5 | 0.9740 | 0.9385 | 0.9886 | 0.0060 |

This agrees with the design: lambda changes downstream adaptive merging, not
foreground segmentation or action-attention extraction. Action overlays focus
on books/hands for `1a`, the mug/handoff region for `2b`, and torso/limbs for
`3_large_pose_change`, but they also cover broad portions of each subject. The
action text repeats the subject noun (for example, "the woman ..."), so the
current map is not a pure verb/contact-region signal.

Point-correspondence snapshots are byte-identical across lambda for every
available condition, as expected: they are captured in the unmodified pre-run
before the final action-gated pass. Valid-match counts are controls, not
measures of lambda's effect:

| Prompt | Frame 1 | Frame 2 | Frame 3 | Frame 4 |
| --- | ---: | ---: | ---: | ---: |
| `1a_anchor_verb` | 1,073 | 795 | 983 | 1,115 |
| `2b_transfer` | 1,030 | 1,194 | 1,300 | 1,014 |
| `3_large_pose_change` | 40 | 61 | 3 | 131 |

The large-pose story has a severe correspondence collapse, especially in the
lying-down frame with three matches. That is a more immediate limitation than
fine lambda selection for that prompt.

## Qualitative findings

### `1a_anchor_verb`: identity is stable, difficult verbs are not solved

All lambdas preserve the woman, cardigan, library, and book reasonably well.
The reading frame is plausible. The throwing frame still looks like the woman
is holding a book rather than throwing it; the balancing frame introduces a
stack of books rather than reliably one thick book; and the tearing frame is
primarily an open-book pose. Lambda changes facial, clothing, book, and
background details, but no setting provides a clear action-compliance win.

### `2b_transfer`: interaction is plausible, but ownership and roles fail

The scene maintains both people and the kitchen at all lambdas, and the handoff
frames contain a cup between them. However, later frames fail the requested
state transition: the "man alone" frame can retain a mug with each person, and
the final frame can show the woman drinking while the man stands with folded
arms. These are role-assignment and object-ownership failures. A single shared
identity bank with merge-only action gating is not expected to resolve them.

### `3_large_pose_change`: broad identity remains, but pose tracking is weak

Lambda 0.5 and 1 retain a recognizable muscular man and gym context while
changing body appearance. The squat and fetal-position frames are broadly
plausible. The forward-bend frame does not consistently keep straight legs,
and the requested supine, arms-crossed pose is not achieved. Since lambda zero
is missing and valid matches are extremely sparse, this is currently a stress
test of correspondence robustness rather than evidence of a gating benefit.

## What can and cannot be concluded

### Supported by the current runs

- Changing lambda changes final images, and stronger gating yields larger
  output differences in the complete sweeps.
- The variant does not destroy broad identity or scene consistency here.
- The pre-run correspondence path is stable across lambda, compatible with the
  gate being isolated to the intended later merge stage.

### Not supported yet

- A claim that lambda 1 improves action compliance, interaction correctness,
  or temporal consistency relative to lambda 0.
- A claim that the lambda-zero large-pose failure was fixed by the variant.
  Point selection is lambda-independent and similarly sparse at lambda 0.5
  and 1; the missing baseline may instead be an empty-match edge case, stale
  worker state, or another run-specific condition.
- A direct claim about merge suppression. No `action_gate_trace.jsonl`,
  `action_gate_audit.json`, or `merge_diagnostics/*_weights.npz` artifact
  exists in the reviewed results.

## Recommended next steps

### 1. Establish the internal causal trace

Start a fresh Colab worker with the current code, then rerun the same prompts
at lambda 0, 0.5, and 1 with seed 2025. Before interpreting quality, require:

- lambda 0 reports `CONTROL` with zero suppression;
- positive lambda reports `PASS` with non-zero suppression where eligible
  tokens exist;
- every frame exports an NPZ merge map, JSONL step trace, and prompt audit.

Inspect `merge_effective_mean.jpg`, `merge_suppression_mean.jpg`,
`merge_suppression_max.jpg`, and `merge_step_curves.jpg`. These prove applied
merge weights rather than only output consequences.

### 2. Make correspondence failure observable and safe

Reproduce `3_large_pose_change` at lambda zero in a fresh worker with exact
diagnostics. Instrument zero- and one-match cases, keep correspondence indices
one-dimensional, and skip adaptive merge safely when no eligible tokens exist.
Only then attribute the prior failure to the original baseline or a code-path
difference.

### 3. Evaluate quality across seeds

After the trace run succeeds, use the three priority prompts at lambda 0, 0.5,
and 1 across at least three seeds. Score every frame blind to lambda on:

1. character identity outside the action region;
2. requested pose or verb compliance;
3. object count and ownership;
4. actor-role correctness for interactions;
5. anatomy and temporal story consistency.

Use this ordinal human rubric as the primary semantic metric. Add an identity
embedding or masked DINO-style similarity score as a secondary measure;
full-image similarity and pixel MAE are not quality metrics for this question.

### 4. Refine the single-character gate signal

Test a predicate/contact-focused score that emphasizes verb and manipulated-
object tokens while excluding repeated subject nouns. The goal is to preserve
face and stable clothing while releasing hands, limbs, contact surfaces, and
the manipulated object. Track this as a new ablation against the current
whole-action-span gate.

If exact traces show meaningful merge suppression but no semantic gain, test a
separate, higher-risk ablation that attenuates earlier identity attention using a
finite query-dependent bias. Do not multiply the existing `0/-inf` attention
mask by the gate: that leaves allowed entries unchanged and risks invalid
`0 * -inf` arithmetic.

### 5. Address transfer failures with entity-aware routing

For `2b_transfer`, more lambda tuning is unlikely to determine who owns the
mug or performs the action. The next structural direction is the previously
assessed multi-entity variant: separate identity banks, per-entity masks and
correspondences, and entity-specific adaptive merge. Keep this independent
from single-character gate experiments so its extra variables do not obscure
the gate's effect.

## Reproduction

Regenerate the current comparisons without GPU inference with:

`python visualize_lambda_results.py --results-root results_colab --prompt-file prompts/stress_test/1a_anchor_verb.txt --lambdas 0,0.5,1 --seed 2025`

Repeat the command for `2b_transfer.txt` and `3_large_pose_change.txt`. Exact
merge panels in the reviewed directories are placeholders by design; they
become informative only after the fresh instrumented rerun.
