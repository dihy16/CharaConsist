# Manual Colab Console Sweep

Use this guide when you want to run the CharaConsist lambda sweep yourself, one
step at a time, instead of invoking `run_colab.sh`. Commands below run from an
Ubuntu or WSL terminal in the `CharaConsist` directory.

## Recommended stress-test priority

Run each prompt at both `lambda=0` and `lambda=1` with the same seed so the
effect of action gating can be compared directly. Prioritize the prompts in
this order:

1. `1a_anchor_verb.txt` — single-character action baseline.
2. `3_large_pose_change.txt` — identity consistency through large pose changes.
3. `2b_transfer.txt` — two-character interaction and object handoff.
4. `3b_prop_consistency.txt` — prop persistence through pose and state changes.
5. `5b_directional_relational.txt` — directional relations and position swaps.
6. `4_bg_to_fg.txt` — background-to-foreground object transition.
7. `6_composite.txt` — combined multi-character and object stress case.
8. `4b_2char_bg2fg.txt` — two-character background-to-foreground transition.
9. `2_close_object.txt` — close-contact interaction and separation.
10. `1b_rare_verb.txt` — unusual action verbs.
11. `5_relational.txt` — broader relational-scene coverage.

If compute is limited, start with `1a_anchor_verb.txt`,
`3_large_pose_change.txt`, and `2b_transfer.txt` before expanding the sweep.

## 1. Create and prepare a session

```bash
cd "/mnt/d/10 Personal/10.11 Projects/1HOIConsist/CharaConsist"

SESSION="characonsist-manual-a100"
MODEL_DRIVE="/content/drive/MyDrive/Colab/models/FLUX.1-dev"
MODEL_PATH="/content/models/FLUX.1-dev"
REMOTE_ROOT="/content/CharaConsist"

colab new -s "$SESSION" --gpu A100
colab drivemount -s "$SESSION"
```

## 2. Stage the checkpoint on local Colab storage (recommended)

Reading model shards through the Google Drive mount is slow. Copy the model to
the VM's local disk once per session, then keep using `MODEL_PATH` below. This
does not upload or duplicate the model on Drive; `/content` is temporary and
is cleared when the session stops.

```bash
cat <<PY | colab exec -s "$SESSION" --timeout 7200
from pathlib import Path
import subprocess

Path("$MODEL_PATH").mkdir(parents=True, exist_ok=True)
subprocess.run([
    "rsync", "-a", "--info=progress2",
    "$MODEL_DRIVE/", "$MODEL_PATH/",
], check=True)
PY
```

`rsync` safely resumes an interrupted copy and skips files already copied in
the same session. Check that `MODEL_PATH` is passed to `worker.start()`; do not
continue using the Drive path after staging.

### Alternative: load directly from Google Drive (previous workflow)

Before local staging was added, the worker loaded the checkpoint directly from
the Drive mount. This avoids the initial copy, but model startup and shard
reads are usually much slower. Skip the staging command above and point
`MODEL_PATH` at the mounted directory instead:

```bash
MODEL_PATH="$MODEL_DRIVE"
```

Then follow the remaining steps unchanged. In particular, pass this
`MODEL_PATH` value to `worker.start()` in step 4:

```python
print(json.dumps(worker.start(
    "$REMOTE_ROOT",
    "$MODEL_PATH",  # /content/drive/MyDrive/... when loading from Drive
    "black-forest-labs/FLUX.1-dev",
    0,
    "0,0.5,1",
    "2025",
), indent=2), flush=True)
```

Keep the Drive mounted for the entire worker lifetime. Do not use this mode
after a local staging copy unless you deliberately want to switch back to the
slower mounted path.

## 3. Upload source and prompts

Create and upload the same source and prompt archives used by the batch
runner. The model is not included in these archives.

```bash
WORK_DIR="$(mktemp -d)"
tar -C . -czf "$WORK_DIR/source.tar.gz" \
  inference.py prompt_utils.py point_visualization.py action_visualization.py \
  merge_diagnostics.py \
  sweep_utils.py make_story_image.py run_colab_remote.py run_batch_inference.py \
  run_colab_worker.py requirements-colab.txt models
tar -C prompts/stress_test -czf "$WORK_DIR/prompts.tar.gz" .

printf '%s\n' "from pathlib import Path; Path('$REMOTE_ROOT').mkdir(parents=True, exist_ok=True)" \
  | colab exec -s "$SESSION"
colab upload -s "$SESSION" "$WORK_DIR/source.tar.gz" "$REMOTE_ROOT/source.tar.gz"
colab upload -s "$SESSION" "$WORK_DIR/prompts.tar.gz" "$REMOTE_ROOT/prompts.tar.gz"
printf '%s\n' "import pathlib, subprocess; root=pathlib.Path('$REMOTE_ROOT'); subprocess.run(['tar', '-xzf', str(root/'source.tar.gz'), '-C', str(root)], check=True)" \
  | colab exec -s "$SESSION"
```

If a worker is already initialized, uploading new source does not replace its
imported pipeline classes. Start a fresh Colab session and repeat model staging
and `worker.start()` before collecting exact merge diagnostics.

For the first diagnostic run after updating this code, restart explicitly:

```bash
colab stop -s "$SESSION"
colab new -s "$SESSION" --gpu A100
colab drivemount -s "$SESSION"
```

The restart clears `/content`, so repeat step 2 (unless loading directly from
Drive), then repeat step 3 and start the worker in step 4. Do not merely upload
the new archive into a session whose worker has already imported the old
attention processor.

## 4. Start the worker once

This installs the Colab-safe dependencies and loads FLUX once. It can take
several minutes; wait for the `status: ready` result before running a task.

```bash
cat <<PY | colab exec -s "$SESSION" --timeout 7200
import json
import sys

sys.path.insert(0, "$REMOTE_ROOT")
import run_colab_worker as worker

print(json.dumps(worker.start(
    "$REMOTE_ROOT",
    "$MODEL_PATH",
    "black-forest-labs/FLUX.1-dev",
    0,
    "0,0.5,1",
    "2025",
), indent=2), flush=True)
PY
```

## 5. Run and download one task

Run one prompt at one lambda value. The command streams that task's inference
output directly to your terminal.

```bash
PROMPT="3_large_pose_change.txt"
LAMBDA="0.5"
LAMBDA_DIR="lambda_0p50"

cat <<PY | colab exec -s "$SESSION" --timeout 7200
import base64
import json
import sys

sys.path.insert(0, "$REMOTE_ROOT")
import run_colab_worker as worker

print(json.dumps(worker.run_one("$PROMPT", $LAMBDA, 2025)), flush=True)
PY

mkdir -p "results_colab/$LAMBDA_DIR/seed_2025/bg_fg"
colab download -s "$SESSION" \
  "$REMOTE_ROOT/result_archives/$LAMBDA_DIR/seed_2025/bg_fg/${PROMPT%.txt}.tar.gz" \
  "$WORK_DIR/${PROMPT%.txt}.tar.gz"
tar -xzf "$WORK_DIR/${PROMPT%.txt}.tar.gz" -C results_colab
```

Repeat this section for each desired prompt and lambda value. The expected output
path is:

```text
results_colab/lambda_0p50/seed_2025/bg_fg/1a_anchor_verb/prompt_0/
```

The worker recognizes completed remote tasks using `_SUCCESS.json`; rerunning
the same `worker.run_one(...)` call returns `skipped` instead of generating it
again. Download each successful archive before stopping the session, because
the remote runtime storage is temporary.

### Run the original ungated CharaConsist baseline

Use `lambda=0` to reproduce the original adaptive identity-merge behavior
without action-attention gating. At zero strength, the gate multiplier is one,
so action scores do not change the merge weights. The current runner may still
calculate and save action maps for diagnostics, but those calculations do not
affect image generation.

The worker must have been started with `0` included in its strength list, as in
step 4. Run and download the baseline result with:

```bash
PROMPT="3_large_pose_change.txt"
LAMBDA="0"
LAMBDA_DIR="lambda_0p00"

cat <<PY | colab exec -s "$SESSION" --timeout 7200
import json
import sys

sys.path.insert(0, "$REMOTE_ROOT")
import run_colab_worker as worker

print(json.dumps(worker.run_one("$PROMPT", $LAMBDA, 2025)), flush=True)
PY

mkdir -p "results_colab/$LAMBDA_DIR/seed_2025/bg_fg"
colab download -s "$SESSION" \
  "$REMOTE_ROOT/result_archives/$LAMBDA_DIR/seed_2025/bg_fg/${PROMPT%.txt}.tar.gz" \
  "$WORK_DIR/${PROMPT%.txt}.tar.gz"
tar -xzf "$WORK_DIR/${PROMPT%.txt}.tar.gz" -C results_colab
```

Only run the download and extraction commands when `worker.run_one()` returns
`status: success` or `status: skipped`. A failed task does not produce a result
archive.

## 6. Compare a prompt across lambda values

After downloading the results, create separate labeled comparison figures with
a row for each lambda and a column for each story frame:

```bash
python visualize_lambda_results.py \
  --results-root results_colab \
  --prompt-file prompts/stress_test/3_large_pose_change.txt \
  --lambdas 0,0.5,1 \
  --seed 2025
```

Each prompt and seed now have their own comparison directory. The generated
output comparison is:

```text
results_colab/comparisons/3_large_pose_change/seed_2025/outputs.jpg
```

The same directory contains `masks.jpg`, `action_attention.jpg`, `points.jpg`,
exact mean effective merge weights, mean and maximum suppression,
denoising-step curves, amplified output differences, and an audit dashboard.
`comparison_audit.json` records pixel MAE and propagation status.

Each successful remote prompt result also contains:

```text
action_gate_trace.jsonl
action_gate_audit.json
merge_diagnostics/<frame>_weights.npz
```

The terminal prints `[action-gate]` summaries after each frame. Lambda zero
must report `CONTROL`; a positive lambda normally reports `PASS`. `WARN` means
no eligible token was changed, while a mathematical invariant violation fails
the task. If a requested lambda failed or was not downloaded, its row remains
visible as `missing / failed`. Omit `--lambdas` to discover available lambda
directories automatically. Existing results generated before exact merge
capture show missing merge/audit panels and must be rerun for those diagnostics.

## 7. Monitor and finish

In another local terminal:

```bash
colab status -s "$SESSION"
colab log -s "$SESSION" -n 100
```

After all tasks are complete, download the summary and stop the VM:

```bash
printf '%s\n' "import json, sys; sys.path.insert(0, '$REMOTE_ROOT'); import run_colab_worker as worker; print(json.dumps(worker.finish(), indent=2))" \
  | colab exec -s "$SESSION"
colab download -s "$SESSION" "$REMOTE_ROOT/results/batch-summary.json" results_colab/batch-summary.json
colab stop -s "$SESSION"
rm -rf "$WORK_DIR"
```

Use `colab console -s "$SESSION"` only when you need a remote shell for
diagnosis. The `colab exec` commands above are the manual Python execution
path and preserve the one-model worker across tasks.
