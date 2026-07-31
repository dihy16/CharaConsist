---
name: develop-characonsist
description: Maintain and operate the CharaConsist FLUX.1 consistency pipeline. Use for changes or debugging in CharaConsist inference, prompt-span parsing, identity attention and action gating, tests, or Google Colab batch execution and result recovery.
---

# Develop CharaConsist

Work only in `CharaConsist/`. Read `AGENTS.md` and the smallest relevant
source files before editing. Treat `OneHOI/` as inspiration only; do not port
its trained components or weights without explicit direction.

## Choose the Work Path

- **Inference, prompts, or action gating:** inspect `inference.py`,
  `prompt_utils.py`, `models/pipeline_characonsist.py`,
  `models/attention_processor_characonsist.py`, and relevant tests. Preserve
  cumulative token boundaries for `background#foreground#action`; the action
  span must remain distinct after tokenizer truncation.
- **Colab batches:** use `run_colab.sh` with `run_colab_bootstrap.py` and
  `run_colab_remote.py`. Do not use `run_colab.py` for new workflows. Require
  an explicit remote model path and accelerator before creating a session.
- **Standalone masks or points:** change `point_and_mask/` only when the task
  explicitly concerns that utility; it is not the main inference path.

## Implement Safely

- Keep prompt/span bookkeeping and attention-processor interfaces synchronized.
  Validate tensor shape, device, dtype, and batch behavior at each boundary.
- Preserve baseline behavior: `--action_gate_strength 0` must disable action
  suppression, and unchanged prompt files must retain their existing format.
- Prefer pure helper functions for attention or merge math and add focused
  tests under `tests/`.
- Do not expose or commit `.env`, Hugging Face tokens, OAuth credentials,
  model weights, or generated results.

## Validate

Run the narrowest applicable checks, then expand if the change crosses layers:

```bash
python -m unittest discover -s tests -v
python -m py_compile inference.py prompt_utils.py run_colab_bootstrap.py run_colab_remote.py
bash -n run_colab.sh
```

For prompt changes, validate all nonblank records as three nonempty
`#`-separated fields. Run actual GPU inference only when the user authorizes
the compute cost.

## Run Colab Batches

- Use a unique session name and monitor it with `colab status -s <name>` or
  `colab log -s <name> -n 100`.
- Let the remote runner finish every prompt file. Successful files are moved
  into final results; failures appear in `batch-summary.json`.
- Download available results before returning an aggregate failure. Stop the
  session unless `--keep` was explicitly requested, then verify with
  `colab sessions` if cleanup matters to the task.
