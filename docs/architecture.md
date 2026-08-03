# CharaConsist Architecture

## Overview

CharaConsist is an inference-focused Python project for generating a consistent
character across a sequence of FLUX.1-dev images. It extends the Diffusers
`FluxPipeline` with custom attention processors that:

- derive a foreground mask from text-to-image attention;
- retain identity-image attention features across diffusion steps;
- selectively share foreground and, optionally, background features with later
  frames; and
- use cross-image similarity to match foreground tokens and interpolate their
  hidden states.

There is no training pipeline or web service in this repository. The primary
interface is a local batch-inference command, with optional Colab CLI wrappers.

## Component Map

```text
prompt .txt files
       |
       v
inference.py -> characonsist.inference -> results/<run>/
  prompt parsing                              |- generated images
  run sequencing                              |- mask visualizations
  output serialization                         `- story.jpg (mix mode)
       |
       v
models.pipeline_characonsist.CharaConsistPipeline
  FLUX denoising loop + consistency policy
       |
       v
models.attention_processor_characonsist
  custom Flux attention processors
       |
       v
Diffusers FluxPipeline + FLUX.1-dev weights + CUDA/PyTorch

characonsist.story
  prompt metadata + completed frames -> labeled story montage

characonsist/
  diagnostics/       saved masks, points, action maps, and trace audits
  experiments/       reproducible sweep and ablation conditions
  runners/           local batch and persistent Colab orchestration
  visualization/     lambda and component comparison renderers

point_and_mask/
  standalone mask-extraction and point-matching implementation
```

## Entrypoints and Operations

| Entry point | Responsibility |
| --- | --- |
| `inference.py` | Stable compatibility CLI for `characonsist.inference`. |
| `characonsist/` | Packaged runtime, diagnostics, experiment conditions, runners, and visualization implementation. |
| `run.sh` | Thin local shell loop over all `.txt` files in a prompt directory. |
| `run_colab.py` / `run_colab.sh` | Experimental Colab CLI orchestration: create a session, upload prompts, invoke inference, and download results. |
| `make_story_image.py` | Stable CLI for the packaged story-montage generator. |
| `point_and_mask/` | Independent version of the mask extraction and point matching logic for experimentation/visualization. |

The runtime dependencies in `requirements.txt` center on PyTorch/CUDA,
Diffusers, Accelerate, Transformers/Hugging Face packages, NumPy, OpenCV, and
Pillow. Model weights are external and supplied through `--model_path`.

## Input Contract

`inference.py` consumes a prompt file containing lines in this form:

```text
background # character # action
```

Blank lines delimit scenes. For each non-blank line, the runner computes:

- `bg_len`: the number of background tokens;
- `action_start`: the first token in the existing action field;
- `real_len`: the number of tokens in the complete prompt; and
- the concatenated FLUX prompt.

The cumulative boundaries let the attention processors separate background,
foreground, and action attention without assuming that independently tokenized
fragment lengths are additive. Foreground remains the combined character and
action range for backward-compatible mask extraction. In normal mode each
blank-line-delimited scene becomes a separate `prompt_<n>` output directory.
In `--mix_mode`, all scenes are flattened into one sequence and the first frame
of a later scene is marked `update_bg=True`.

## Generation Flow

1. **Model initialization** — `inference.py` loads
   `CharaConsistPipeline.from_pretrained(model_path)`, then installs custom
   processors using `reset_attn_processor`. Initialization modes choose direct
   GPU placement, CPU offload, balanced multi-GPU placement, or sequential
   offload.
2. **Identity generation** — the first prompt is invoked with `is_id=True`.
   At selected denoising timesteps the pipeline records attention weights,
   image key/value tensors, and attention outputs in the processors' per-step
   `id_attn_bank`.
3. **Foreground-mask extraction** — recorded text-to-image attention is
   averaged across custom transformer layers. Foreground attention greater
   than or equal to background attention becomes a binary spatial mask;
   OpenCV erosion/dilation removes small artifacts.
4. **Pre-run for each later frame** — the pipeline invokes the new prompt with
   `is_pre_run=True`. It obtains cross-image similarity between current and
   identity attention outputs, identifies corresponding foreground tokens, and
   aggregates a continuous action-attention map over the current foreground.
5. **Frame generation** — the normal invocation reuses the identity bank.
   Custom attention expands the current key/value tensors with selected
   identity foreground and/or background tokens. Its attention mask prevents
   invalid foreground/background connections. Matched foreground hidden states
   can also be blended using an adaptive token-merge weight. That merge weight
   is reduced per token by normalized action attention.
6. **Output writing** — the identity image, intermediate `_pre` image, and
   final frames are saved. `--save_mask` writes visual foreground-mask
   overlays; `--save_all_steps` additionally writes per-step mask and matching
   visualizations. Mix mode calls `save_story_visualization` after generation.
7. **Reset** — `reset_id_bank` clears retained attention tensors before the
   next independent scene.

## Core Modules

### `models/pipeline_characonsist.py`

`CharaConsistPipeline` subclasses Diffusers' `FluxPipeline` and retains the
standard FLUX prompt encoding, latent preparation, scheduler loop, transformer
call, and VAE decoding. Its extension is a timestep-dependent consistency
policy passed through `joint_attention_kwargs` and `spatial_kwargs`.

The three modes are:

- **Identity (`is_id`)**: record masks and attention-bank data.
- **Pre-run (`is_pre_run`)**: calculate cross-image similarity and foreground
  point correspondences.
- **Generation**: share identity foreground; share identity background when
  `share_bg` is enabled; optionally update the background bank at a scene
  transition.

The default schedule records mask data at step 10, shares attention features
from steps 1–40, and interpolates foreground features from steps 1–30 with a
cosine-decaying weight after step 11.

### `models/attention_processor_characonsist.py`

This module contains both a standard `FluxAttnProcessor2_0` replacement and
the custom `CharaConsistAttnProcessor2_0` used for FLUX single-transformer
blocks. The custom processor owns the state that connects frames:

| State | Purpose |
| --- | --- |
| `attn_weights` | Background, foreground, and action text-attention maps used for mask extraction and merge gating. |
| `id_attn_bank` | Identity key/value tensors and attention outputs indexed by diffusion timestep. |
| `cross_sims` | Current-to-identity visual-token similarity used to find point matches. |
| `bg_len`, `action_start`, `real_len` | Cumulative prompt token boundaries supplied by the runner. |
| `action_scores` | Normalized continuous action-attention map used to gate adaptive token merge. |

It builds a boolean expanded-attention mask, converts blocked connections to
`-inf`, and supplies it to PyTorch scaled-dot-product attention. The helper
functions at the end of the module install processors, set token lengths,
aggregate masks/similarities, reset identity state, and update spatial size.

### `characonsist/inference.py`

The inference runner is the application layer. In addition to argument parsing
and model setup, it owns prompt-file parsing, deterministic per-call seeds,
the identity/pre-run/final-frame invocation sequence, and serialization of
images and diagnostics. The `--save_mask` option saves rendered foreground
mask overlays; it does not serialize raw attention-mask tensors.

### `characonsist/story.py`

This utility is isolated from diffusion inference. It parses the same prompt
format, selects finished images (excluding `_pre` and existing story images),
and uses Pillow to lay out a montage with shared environment/character text and
per-frame action captions.

### `point_and_mask/`

This directory is a smaller, independent implementation of the project’s
attention-derived mask and cross-image point-matching ideas. Its
`MaskPointPipeline` subclasses `FluxPipeline` and its processor retains only
the state required to create masks and visual-token correspondence, without
the full CharaConsist feature-sharing policy.

## Outputs

For a run rooted at `--out_dir`, normal mode writes:

```text
<out_dir>/
  prompt_0/
    id.jpg
    0_pre.jpg
    0.jpg
    mask/                       # only with --save_mask
      id_mask.jpg
      0_mask.jpg
      id_all_steps/             # with --save_all_steps
      0_all_steps/              # with --save_all_steps
```

Mix mode writes `id.jpg`, numbered pre/final frames, optional masks, and
`story.jpg` directly under `--out_dir`.

## External Boundaries and Constraints

- **Model boundary:** FLUX.1-dev weights must already be available locally at
  `--model_path`; they are not included in the repository.
- **Framework boundary:** the project depends on Diffusers' FLUX pipeline and
  PyTorch 2 scaled-dot-product attention behavior.
- **Hardware boundary:** direct single-GPU initialization (`--init_mode 0`)
  is documented as requiring roughly 37 GB VRAM. CPU-offload modes reduce GPU
  memory requirements at the cost of performance.
- **Remote execution boundary:** Colab wrapper scripts are orchestration only;
  they do not alter inference semantics. A fresh remote environment must have
  the source tree, dependencies, and model weights available before
  `inference.py` can run.
