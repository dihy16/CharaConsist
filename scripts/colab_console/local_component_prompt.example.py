"""Run all component-attribution modes for one prompt at one matched seed.

Copy this file to ``local_component_prompt.py`` and edit PROMPT and SEED.
Each mode receives precisely the same seed.
"""

import json
import sys
from pathlib import Path

PROMPT = "3b_prop_consistency.txt"
SEED = 2025
CONSISTENCY_MODES = ("prompt_only", "attention_only", "full")

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
sys.path.insert(0, config["remote_root"])

import run_colab_worker as worker

for mode in CONSISTENCY_MODES:
    print(f"Running {PROMPT} | mode={mode} | seed={SEED}", flush=True)
    print(json.dumps(worker.run_component(PROMPT, mode, SEED), indent=2), flush=True)
