# Manual Colab Console Workflow

This workflow uses local shell scripts instead of copying multi-line commands
into the terminal. Run every command below from the `CharaConsist` directory
under Linux or WSL. Each shell script calls `colab exec -f` for the remote
Python work, so the persistent Colab kernel remains the only place FLUX is
loaded.

## Configure once per session

Create an ignored local configuration file and set your Colab session name,
paths, and accelerator:

```bash
cp scripts/colab_console/manual_config.example.json \
  scripts/colab_console/manual_config.json
nano scripts/colab_console/manual_config.json
```

The default configuration stages the model from Google Drive to
`/content/models/FLUX.1-dev`. Keep the staged path in `model_path`; the Drive
model remains unchanged and `/content` is cleared when the session ends.

All scripts accept the configuration at
`scripts/colab_console/manual_config.json`. To use a differently located
configuration, set `CHARACONSIST_MANUAL_CONFIG=/path/to/config.json` before
calling a script. Results default to `results_colab`; override that destination
with `CHARACONSIST_RESULTS_DIR=/path/to/results`.

## Script map

| Script | Purpose |
| --- | --- |
| `00_restart_kernel.sh` | Restart the kernel before switching worker type or loading new source. |
| `01_prepare_session.sh` | Create the configured A100/H100 session if needed, mount Drive, and upload configuration. |
| `02_stage_model.sh` | Copy the checkpoint to VM-local storage and verify the copy. |
| `03_upload_source_and_prompts.sh [prompt-dir]` | Upload current source plus prompt files and extract them remotely. |
| `04_start_worker.sh <lambda\|component\|role\|binding\|routing>` | Load FLUX once for the selected experiment type. |
| `05_run_task.sh <mode> <prompt.txt> <value> [seed]` | Run one task and automatically download/extract its archive. |
| `05_run_entity_routing_matrix.sh` | Run the matched K=2 routing × action-bias diagnostic. |
| `06_compare_lambda.sh`, `06_compare_component.sh`, `06_compare_role_action.sh` | Render local comparisons and audits. |
| `07_monitor.sh [lines]` | Show Colab status and recent worker logs. |
| `08_finish.sh [--keep]` | Finalize, download `batch-summary.json`, and stop the session unless retained. |

Invoke them with `bash`; executable permissions are not required.

## Lambda-gating diagnostic run

Use a fresh session when you need exact gate diagnostics from the current
source. The two conditions must share the prompt and seed. Start with
`1a_anchor_verb.txt`: it isolates the original action-fidelity hypothesis
without the multi-character routing confound.

```bash
bash scripts/colab_console/01_prepare_session.sh
bash scripts/colab_console/02_stage_model.sh
bash scripts/colab_console/03_upload_source_and_prompts.sh
bash scripts/colab_console/04_start_worker.sh lambda

bash scripts/colab_console/05_run_task.sh lambda 1a_anchor_verb.txt 0 2025
bash scripts/colab_console/05_run_task.sh lambda 1a_anchor_verb.txt 1 2025
bash scripts/colab_console/06_compare_lambda.sh \
  prompts/stress_test/1a_anchor_verb.txt 0,1 2025
```

`lambda=0` is the unaltered CharaConsist control. `lambda=1` is the maximum
valid action-merge suppression: the formula is
`base_merge * (1 - lambda * action_score)`, so values above one would merely
saturate clamped weights instead of making a controlled stronger variant.

The latest run writes `action_gate_trace.jsonl`, `action_gate_audit.json`, and
`merge_diagnostics/*.npz` below the resulting `prompt_0/` directory. Expect
`CONTROL` for lambda zero. A positive lambda should report `PASS` with modified
tokens and nonzero suppression; `WARN` means it had no eligible merge tokens.
Those files establish mechanism execution, not action-fidelity improvement.

## Component-attribution experiment

This compares `prompt_only`, `attention_only`, and unaltered `full`
CharaConsist, all at the same seed with the action gate fixed at zero. Use a
fresh kernel if a lambda worker is already loaded; staging and source files
remain on the VM after a kernel restart.

```bash
bash scripts/colab_console/00_restart_kernel.sh
bash scripts/colab_console/04_start_worker.sh component

bash scripts/colab_console/05_run_task.sh component 1a_anchor_verb.txt prompt_only 2025
bash scripts/colab_console/05_run_task.sh component 1a_anchor_verb.txt attention_only 2025
bash scripts/colab_console/05_run_task.sh component 1a_anchor_verb.txt full 2025
bash scripts/colab_console/06_compare_component.sh \
  prompts/stress_test/1a_anchor_verb.txt 2025
```

Prioritize `1a_anchor_verb.txt`, `1b_rare_verb.txt`, and
`3b_prop_consistency.txt`. This experiment attributes what CharaConsist's
existing components contribute; it is not a test of the role-action variant.

## Role-action routing experiment

`2b_transfer_roles.txt` contains optional `[S]` subject, `[A]` predicate,
`[O]` object, and `[R]` recipient annotations. Tags are stripped before FLUX
encoding, and the cleaned action text is tested to match `2b_transfer.txt`.
The two conditions retain full CharaConsist, fix action gating at zero, and use
the same seed:

```bash
bash scripts/colab_console/00_restart_kernel.sh
bash scripts/colab_console/04_start_worker.sh role

bash scripts/colab_console/05_run_task.sh role 2b_transfer_roles.txt 0 2025
bash scripts/colab_console/05_run_task.sh role 2b_transfer_roles.txt 1 2025
bash scripts/colab_console/06_compare_role_action.sh \
  prompts/stress_test/2b_transfer_roles.txt 2025
```

The output directory is
`results_colab/role_action_ablation/role_bias_<strength>/seed_2025/bg_fg/`.
Each annotated run saves `role_action_trace.json` and raw/overlay role maps.
`role_action_audit.json` reports output pixel MAE and role-map MAE. The role
maps should match exactly between strength zero and one because both are
computed in the unbiased pre-run. Assess frames 3 and 4 for mug ownership and
actor/observer binding; changed pixels alone are not an improvement.

## Character-conditioned action-binding experiment

This staged ablation keeps point tracking, foreground/background masks, adaptive
token merge, and identity K/V injection unchanged. It changes only visual-query
attention to indexed action spans. `C1/A1` denote the man and `C2/A2` the woman.
Every run uses full CharaConsist with action gating and role routing disabled.

Start a fresh binding worker. It is configured for the four conditions
`(beta,gamma) = (0,0), (1,0), (1,0.5), (2,1)` and seeds 2025--2029:

```bash
scripts/colab_console/04_start_worker.sh binding
```

Run the tagged zero-strength preflight and inspect both overlays. C1 must locate
the man, C2 must locate the woman, and the automated report must pass:

```bash
scripts/colab_console/05_run_binding_preflight.sh
scripts/colab_console/05_approve_binding_maps.sh
```

The approval command records a local marker only after the human map check. It
does not change model output. Next, compare the untagged baseline against the
cleaned tagged zero-strength prompt, then run the weak contrastive condition at
the same seed:

```bash
scripts/colab_console/05_run_binding_checkpoints.sh
```

`mechanical_audit_seed_2025.json` must report exact ID/pre/final equivalence for
the zero control, identical frozen maps between zero and weak runs, nonzero bias
effect, all 1,520 expected invocations (38 blocks x steps 1--40), and a nonzero
final-image MAE. The local audit needs NumPy and Pillow in the Python environment
that runs these WSL scripts. Set that up once if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pillow
```

Keep `.venv` activated while running the audit and matrix scripts. Only then run
the 20 matched samples:

```bash
scripts/colab_console/05_run_binding_matrix.sh
```

The matrix script then runs `compare_action_binding_results.py`, producing a
four-by-five image grid and `manual_scores.csv` under
`comparisons/action_binding/2b_final_action_binding/`.

Score each output manually for `man_drinking`, `woman_arms_folded`,
`identities_correct`, and their combined success. Restore the complete transfer
sequence with `2b_transfer_action_binding.txt` only if a binding condition beats
the baseline across the five matched seeds. Frame 2 is intentionally untagged
because both people jointly grip the mug; frames 0, 1, 3, and 4 are bound.

After filling every combined-success cell with `0` or `1`, run the selected
condition against its matched zero control on the complete sequence:

```bash
scripts/colab_console/05_run_binding_full_sequence.sh 1:0.5
```

`check_binding_scores.py` blocks this step unless the selected condition's
five-seed success rate is strictly higher than the baseline rate.

To run an individual condition, encode beta and gamma as `beta:gamma`:

```bash
scripts/colab_console/05_run_task.sh binding 2b_final_action_binding.txt 1:0.5 2025
```

## K=2 entity-specific identity routing

This diagnostic partitions the saved foreground identity tokens into C1 and C2,
restricts point matching to the same entity, blocks cross-entity access to the
appended identity K/V columns, and feeds only same-entity matches to adaptive
merge. Ordinary within-frame attention is unchanged. Routing `off` is the exact
compatibility control.

Start a fresh worker and run the four conditions `{off,hard} x {0:0,1:0.5}`
across seeds 2025--2029:

```bash
bash scripts/colab_console/00_restart_kernel.sh
bash scripts/colab_console/04_start_worker.sh routing
bash scripts/colab_console/05_run_entity_routing_matrix.sh
```

An individual routing value is encoded as `mode:beta:gamma`:

```bash
bash scripts/colab_console/05_run_task.sh \
  routing 2b_final_action_binding.txt hard:1:0.5 2025
```

Each hard run writes `entity_routing/entity_routing_trace.json` plus identity
and current-frame categorical overlays. Mechanism success requires valid C1/C2
coverage, same-entity matches for both characters, and zero
`wrong_entity_allowed_pairs`. New traces also include one counterfactual
`attention_mass_records` entry per routed layer and timestep. The aggregate
`attention_mass_summary` reports how much pre-mask probability went to wrong-
versus same-entity identity keys; a low wrong-entity mass explains a mechanically
active intervention with little visible output change. The comparison is written under
`comparisons/entity_routing/2b_final_action_binding/`. The matrix script also
writes `mechanical_audit.json`; it requires exact routing-off equivalence to the
prior action-binding results and a nonzero hard-routing output change.

Expected outcome: hard routing should eliminate wrong-identity-bank access and
may improve identity separation. It may not change who owns the mug because the
anchor associates the mug with C2 and this diagnostic does not explicitly route
the prop. If the trace passes but combined binding does not beat both off
controls, record entity routing as mechanically valid but semantically
insufficient; do not increase beta/gamma as the next step.

After filling all manual score cells, enforce the semantic gate and only then
run the full transfer sequence against its matched control:

```bash
python3 scripts/check_entity_routing_scores.py
bash scripts/colab_console/05_run_task.sh \
  routing 2b_transfer_action_binding.txt off:0:0 2025
bash scripts/colab_console/05_run_task.sh \
  routing 2b_transfer_action_binding.txt hard:1:0.5 2025
```

An untagged action frame is a deliberate neutral action-bias frame; hard entity
routing still applies because its C1/C2 identity tags remain present.

## Direct Drive mode

To use the slower previous workflow, set `model_path` equal to `model_drive`
in `manual_config.json`. Run `01_prepare_session.sh`, skip
`02_stage_model.sh`, then upload source/prompts and start the desired worker.
Keep Drive mounted for the worker's lifetime.

## Monitor and finish

Run this from a second terminal while a task is active:

```bash
bash scripts/colab_console/07_monitor.sh
```

After downloading all desired results:

```bash
bash scripts/colab_console/08_finish.sh
```

Pass `--keep` to retain the VM. Otherwise it stops the session after saving
`results_colab/batch-summary.json`. Never upload model weights, tokens, or
generated results to Git.
