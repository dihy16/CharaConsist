# CharaConsist Agent Guide

## Scope

These instructions apply only to `CharaConsist/`. Treat the sibling `OneHOI/`
tree as research context only: do not copy its trained modules, weights, or
backbone into CharaConsist unless explicitly asked.

## Research Objective

CharaConsist is the baseline this repository is trying to improve upon. Work
should first identify and reproduce a concrete baseline limitation, then test
the smallest controlled change that targets it. Preserve a runnable unaltered
baseline and use matched prompts, seeds, image sizes, model state, and schedules
for comparisons.

Do not treat a changed image as evidence of an improvement. Distinguish among:

- implementation/mechanism validation (the new path executed as designed);
- output sensitivity (the variant changed the generated pixels);
- task improvement (identity, action, pose, interaction, or consistency became
  measurably better than the CharaConsist baseline).

Document negative and inconclusive findings, upstream failures, and prompts
that are outside a variant's intended scope. Use these gaps to select the next
hypothesis instead of escalating intervention strength without a causal
control.

## Architecture

- `inference.py` is a compatibility CLI; implementation lives in
  `characonsist/inference.py`, which owns model initialization,
  identity/frame sequencing, and visual outputs.
- `characonsist/prompts.py` builds the combined prompt and cumulative T5 span
  boundaries. Prompt records use `background#foreground#action` and may be
  truncated at 512 tokens.
- `models/pipeline_characonsist.py` coordinates denoising and identity-bank
  use. `models/attention_processor_characonsist.py` performs span attention,
  masks, point matching, and adaptive merging. Keep their shared kwargs and
  tensor shapes consistent.
- `models/action_gating.py` contains pure, unit-testable action-gating helpers.
- `characonsist/diagnostics/`, `characonsist/experiments/`,
  `characonsist/runners/`, and `characonsist/visualization/` separate reusable
  artifact logic, condition definitions, orchestration, and result rendering.
- `run_colab.sh`, `run_colab_bootstrap.py`, and `run_colab_remote.py` are the
  supported Colab batch path. Do not extend the older `run_colab.py` unless a
  task explicitly targets it.
- `point_and_mask/` is a standalone experimental utility, not the primary
  inference path.

## Implementation Rules

- Preserve the three prompt spans and calculate boundaries from cumulative
  prefixes; do not tokenize fragments independently when changing span logic.
- Keep `--action_gate_strength=0` behavior equivalent to the ungated merge.
  Validate new tensor operations for batch, token, and spatial dimensions.
- Keep model weights, generated outputs, `.env`, OAuth credentials, and
  Hugging Face tokens out of commits and command output.
- Respect existing uncommitted user changes. Avoid broad rewrites of attention
  code when a targeted helper or interface change is sufficient.

## Validation

Run the narrowest relevant checks before broader ones:

```bash
python -m unittest discover -s tests -v
python -m py_compile inference.py characonsist/inference.py run_colab_bootstrap.py run_colab_remote.py
bash -n run_colab.sh
```

For prompt changes, check every nonblank record contains exactly three
nonempty `#`-separated fields. GPU inference is optional and requires explicit
user authorization.

## Colab Operations

- Require an explicit user-provided model path and accelerator choice before
  creating a session. Never print the value of `HF_TOKEN`.
- Preserve successful prompt-file results even if later files fail; download
  `batch-summary.json` with available outputs.
- Stop the session after completion or failure unless the user explicitly
  passes `--keep`. Confirm cleanup with `colab sessions` when the task asks
  for it.
