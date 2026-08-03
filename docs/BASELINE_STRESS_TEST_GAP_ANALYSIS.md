# CharaConsist Baseline Stress-Test Gap Analysis

Date: 2026-08-03

## Purpose

This report identifies concrete limitations of CharaConsist as the baseline to
improve. It compares the newly copied outputs under
`results/bg_fg/<prompt>/prompt_0/` with the corresponding prompt files under
`prompts/stress_test/`.

This is a qualitative audit of the available images, not a controlled comparison
of the action-gating variant. The run metadata, seed, model revision, and exact
inference settings are not present beside these results, so the observations
below should be treated as candidate baseline gaps to reproduce under controlled
settings. Visual differences or attractive images are not counted as task
success unless the requested action, pose, relation, and object state are shown.

## Coverage

| Prompt | Result coverage | Overall finding |
| --- | --- | --- |
| `1a_anchor_verb.txt` | Complete | Identity is stable, but throwing and tearing are not depicted. |
| `1b_rare_verb.txt` | Complete | Static chair handling works better than dragging, standing on, or throwing it. |
| `2_close_object.txt` | Complete | Most close interactions work; the requested non-contact final relation fails. |
| `2b_transfer.txt` | Complete | The handoff sequence mostly works, but exclusive ownership is not established in frame 3. |
| `3_large_pose_change.txt` | Complete | Large pose instructions collapse into easier, materially different poses. |
| `3b_prop_consistency.txt` | Complete | Character/prop appearance persists, but open-to-closed umbrella state changes fail. |
| `4_bg_to_fg.txt` | Complete | The mug is carried through the story, but its initial location and precise release/sip actions are inaccurate. |
| `4b_2char_bg2fg.txt` | Complete | Reading and pointing work; sliding, closing, and handing back are not clearly depicted. |
| `5_relational.txt` | Incomplete | Only `id.jpg` and `0_pre.jpg` exist; no final story frame or `story.jpg` is available. |
| `5b_directional_relational.txt` | Complete | A simple position swap works, but the coordinated leading pose and clothing specification fail. |
| `6_composite.txt` | Incomplete | Only frame 0 is finalized; later requested state transitions cannot be evaluated. |

Result folders such as `contact`, `girl_fox`, `hard`, `occlusion`, `posevary`,
`rescue`, and `three` were not included because no same-named prompt exists in
`prompts/stress_test/`. The broad `prompt_0` and `prompt_1` folders were also
excluded because they cannot be paired unambiguously with one stress-test file.

## Per-story discrepancies

### 1a: anchored identity with action verbs

The character, clothing, library, and book remain visually consistent. The
action semantics do not.

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Reading an open textbook on the desk | She reads or writes in an open book at the desk. | Pass, with minor ambiguity between reading and writing. |
| 2 | Throwing the textbook toward a shelf across the room | She holds the book and reaches toward the shelf. There is no throwing pose, release, or visible trajectory. | Clear failure. |
| 3 | Balancing one thick textbook flat on her head | She balances a stack of about three books. | Action passes, object count/state does not. |
| 4 | Tearing a page out of the textbook | She holds an open book and appears to point at or turn/write on a page. No tear is visible. | Clear failure. |

This story directly supports the original action-gating motivation: maintaining
identity does not guarantee that action-sensitive body and object configurations
remain free enough to express the requested predicate. It does **not** establish
that action gating fixes the problem; that requires a matched baseline/variant
comparison.

#### Interpolation comparison for the throwing frame

An additional pair of outputs was identified for this prompt:

- `results/bg_fg/1a_anchor_verb/prompt_0/story.jpg` was generated with
  `use_interpolate=False`.
- `results_colab/lambda_0p00/seed_2025/bg_fg/1a_anchor_verb/prompt_0/story.jpg`
  was generated with `use_interpolate=True`.

Neither frame 2 depicts throwing. With interpolation disabled, the woman keeps
the book close to her body while extending her free hand toward the shelf; the
result reads as shelving or selecting a book. With interpolation enabled, she
holds the book with both hands in an even more static presentation pose. It has
no extended throwing arm, release point, displaced or airborne book, or implied
trajectory toward the shelf.

Qualitatively, interpolation does not recover the requested action in this pair
and may strengthen the conservative/static solution: the character and book are
clear, but the predicate is less dynamic. This is consistent with the hypothesis
that consistency-oriented feature reuse can compete with action-specific spatial
change. It is not yet causal evidence because the non-Colab result does not have
seed and complete run metadata stored beside it. A valid interpolation ablation
must rerun both values with the same seed, initialization, model revision,
resolution, and inference schedule.

### 1b: rare chair verbs

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Lifting a chair above his head with both arms extended | The chair is above his head and both arms are raised. | Pass, although the grip/contact is visually weak. |
| 2 | Dragging the chair by one leg | He walks beside an upright chair and holds its upper portion. The specified one-leg grip and dragging geometry are absent. | Failure. |
| 3 | Standing on top of the chair with both feet | He stands behind the chair with both feet on the sidewalk. | Clear failure. |
| 4 | Throwing the chair into a dumpster | The chair is already partly in the dumpster while he leans over it. The image reads as placing or pushing, not throwing. | Partial/failure. |

The man also changes noticeably in face shape and apparent age across frames.
This prompt exposes both predicate fidelity and some identity drift.

### 2: close two-person/object interaction

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Envelope handoff with fingers touching | Both hold the envelope at the transfer point. | Pass. |
| 2 | Arm-wrestling with elbows on a coffee table | The pair are posed in an arm-wrestling configuration across a table. | Pass. |
| 3 | Man lifting woman fully off the floor in an embrace | The embrace and lifting pose are present; at least one foot is clearly raised and the other is obscured. | Likely pass, with occlusion preventing strict verification of both feet. |
| 4 | Back-to-back, arms crossed, not touching | Their backs visibly touch. | Relation failure. |

This is one of the stronger stories. Its remaining failure is a precise negative
spatial constraint (`not touching`), rather than general inability to render two
people interacting.

### 2b: mug ownership transfer

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Woman extending the mug toward the man | She presents the mug toward him. | Pass. |
| 2 | Both gripping the mug during handoff | Both visibly contact the same mug. | Pass. |
| 3 | Man holding the mug alone; woman's hands empty at her sides | The woman's hand remains on the handle while the man's hands support the mug. Her hands are not empty or at her sides. | Clear ownership-state failure. |
| 4 | Man drinking; woman watching with arms folded | The man drinks and the woman has folded arms. | Pass. |

The sequence demonstrates that CharaConsist can generate a coarse transfer arc,
but the intermediate ownership state is not reliably bound to the intended
person. This is better targeted by entity/object-state routing than by merely
increasing action-gate strength.

### 3: large pose change

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Forward fold, hands on floor, legs straight | He is in a deep lunge/all-fours pose with a bent knee. | Clear failure. |
| 2 | Full squat, thighs parallel, arms extended forward | He squats, but his arms are lowered beside his knees rather than extended forward. | Partial. |
| 3 | Lying flat on his back, arms crossed, eyes open | He is upright on his elbows/forearms with his torso facing the camera. | Clear failure. |
| 4 | Curled tightly in a fetal position on the floor | He is seated and curled around one raised knee rather than lying on his side. | Partial/failure. |

This is the most systematic pose-fidelity failure in the available complete
results. The outputs preserve a muscular shirtless man and gym aesthetic but
replace difficult body configurations with nearby easier ones. Because the
previous lambda-zero run also produced a tensor-shape error for this prompt,
this case should additionally be tracked as a baseline robustness/correspondence
case, separate from semantic pose scoring.

### 3b: prop consistency and state transitions

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Side bend while gripping the open umbrella | The side bend and open umbrella are depicted. | Pass. |
| 2 | Crouching with umbrella tilted on her shoulder | She crouches, but the umbrella remains broadly open behind/above her and the shoulder-resting relation is unclear. | Partial. |
| 3 | Closed umbrella laid across her knees | No closed umbrella lies across her knees; a small rolled yellow object appears beside her. | Clear failure. |
| 4 | Lying down, holding the closed umbrella vertically above her | She lies down and raises an umbrella, but it is open and horizontal. | Clear state/orientation failure. |

This is strong evidence that appearance persistence (yellow umbrella) and
state persistence/transitions (open versus closed, across versus vertical) are
different problems.

### 4: background-to-foreground mug

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| ID | Mug untouched on the shelf behind her | The mug is on the desk in the foreground. | Initial object-location failure. |
| 1 | Turn and reach toward the mug on the shelf | She turns and reaches to a shelf-level mug. | Pass. |
| 2 | Hold near face and take a sip | She holds it near her face, but the rim does not clearly reach her mouth. | Partial. |
| 3 | Place it on the desk with hands releasing | The mug is on the desk, but she still grips it. | Contact-state failure. |
| 4 | Type on laptop, mug untouched to her right | Laptop and stationary desk mug are present; active typing is only weakly shown. | Mostly pass. |

The mug survives the foreground transition, but exact location and hand-contact
states are weaker than object presence.

### 4b: two-character notebook transition

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Woman picks up and opens notebook; man watches | She holds it open while he watches. | Pass. |
| 2 | Woman slides the open notebook across the table | The notebook rests open near the woman; no sliding gesture or clear transfer toward the man is shown. | Failure. |
| 3 | Man reads while woman points to a page | Both conditions are depicted. | Pass. |
| 4 | Man closes it and hands it back, both hands on cover | The notebook remains visibly open while both hold it. | Clear open/closed-state failure. |

As with the umbrella, object identity is easier than the requested state change.

### 5: cross-scene left/right relation

The directory contains an identity image and only `0_pre.jpg`. The preliminary
coffee-shop image places the woman on the image left of the man and roughly
matches the first story frame, but there is no finalized `0.jpg`, no remaining
frames, and no story sheet. It is therefore not evidence for or against
cross-scene relational consistency. The missing outputs are an incomplete-run
or collection gap that should be resolved before evaluating this prompt.

### 5b: directional relation and coordinated pose

| Frame | Requested | Observed | Assessment |
| --- | --- | --- | --- |
| 1 | Side by side at barre with specified left/right hands | They are side by side at the barre, but the exact hand assignments are not clearly satisfied. | Partial. |
| 2 | Man in front, woman directly behind; positions swapped | The man is foreground/central and the woman appears behind him. | Pass. |
| 3 | Man leads with right hand; woman behind with both hands on his shoulders | The man extends both arms and the woman does not have both hands on his shoulders. | Clear coordinated-pose failure. |
| 4 | Facing each other with hands clasped | They face one another and clasp hands. | Pass. |

The man is shirtless in every image despite the character specification only
saying he wears gray sweatpants. The lower garment is consistent, but upper-body
appearance is under-specified and drifts to a stereotypical dance/fitness image.
The prompt file also contains a mojibake sequence (`â€”`) in frame 2; it should
be repaired before using this prompt in a controlled text-encoding experiment.

### 6: composite key/object/pose sequence

Only story frame 0 is finalized. It successfully shows the woman placing a key
into the man's hand with close contact, although the image contains two visible
keys/key-like elements and therefore makes object count ambiguous. `1_pre.jpg`
suggests a crouching cabinet interaction, but preliminary images are not the
baseline's final output. Frames 1 through 3 and `story.jpg` are missing, so the
cabinet, overhead-key, and high-shelf transitions cannot be scored.

## Cross-story diagnosis

The audit reveals several separable baseline gaps:

1. **Predicate/action fidelity:** throwing, tearing, dragging, standing on, and
   coordinated leading are replaced by static holding, reaching, or easier
   contact poses. This is the gap that motivated action gating.
2. **Large-pose fidelity:** the character remains recognizable while difficult
   whole-body geometry is replaced by a lunge, supported sit, or seated curl.
   This may involve correspondence failure upstream of adaptive merge.
3. **Object-state transitions:** the umbrella and notebook remain recognizable
   but fail to become closed; placement/release details are also unreliable.
4. **Ownership and contact state:** a transfer may be broadly depicted while
   the wrong person still touches the object, or a requested non-contact
   relation becomes contact.
5. **Entity-specific relations:** simple two-person arrangements can work, but
   precise hand assignments and multi-body coordination remain fragile.
6. **Run robustness/coverage:** two prompt runs are incomplete, and the large
   pose prompt has separately triggered a lambda-zero tensor-shape failure.

These categories should not be collapsed into one “consistency” score. A method
can improve character similarity while worsening action, pose, object state, or
role binding.

## Implications for the action-gating hypothesis

The hypothesis remains reasonable but narrower than the full set of failures:
uniform identity feature merging may overconstrain regions that must change to
express a new action. `1a_anchor_verb` and parts of `1b_rare_verb` are appropriate
tests of that claim because identity is preserved while the requested predicate
is lost.

The current images do not prove that the implemented gate improves those cases.
The prior lambda sweep showed output sensitivity without a visible semantic
gain. Other failures need different mechanisms:

- ownership/role binding in `2b_transfer` suggests entity-specific routing;
- large-pose collapse in `3_large_pose_change` requires correspondence and pose
  diagnostics before downstream merge changes;
- open/closed prop failures in `3b_prop_consistency` and `4b_2char_bg2fg` suggest
  explicit object/state tracking or state-aware conditioning;
- negative spatial constraints and precise hand contact need relation/contact
  evaluation, not only identity preservation.

## Recommended next experiments

1. Reproduce the unaltered baseline and action-gated variant with identical
   model state, prompt, seed, resolution, and schedule for a small diagnostic
   subset: `1a_anchor_verb`, `1b_rare_verb`, and `3_large_pose_change`.
2. Score each frame using separate labels for identity, predicate, body pose,
   object state, object owner, contact, and spatial relation. Do not use pixel
   change percentage as a semantic metric.
3. Use `1a` throwing and tearing as the primary action-gating test. Verify the
   gate trace and localized suppression first, then check whether the requested
   verb improves without unacceptable identity loss.
4. Treat `2b` frame 3 as an entity/object ownership-routing test rather than an
   action-gate-strength test.
5. Diagnose point correspondence on `3_large_pose_change` independently. A
   downstream gate cannot recover pose evidence that was already lost during
   matching.
6. Add explicit object-state checks using `3b` frames 3–4 and `4b` frame 4.
7. Rerun or recover `5_relational` and `6_composite` before drawing conclusions
   from them, and save run metadata plus failure logs beside every result set.

## Bottom line

The experiment direction is not fundamentally invalid. The baseline evidence
supports the motivating observation that character consistency can coexist with
poor action fidelity. The mistake would be to interpret every discovered gap as
an action-gating problem. The next useful result is a controlled demonstration
that gating improves a specifically action-sensitive frame such as throwing or
tearing; other gap classes should be evaluated with mechanisms designed for
pose correspondence, entity routing, or object-state tracking.
