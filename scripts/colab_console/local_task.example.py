"""Run one prompt and lambda condition using the initialized remote worker.

Copy this file to ``local_task.py`` and edit the three values below for each
condition. It remains local; the Colab CLI sends its contents to the kernel.
"""

import json
import sys
from pathlib import Path

PROMPT = "3_large_pose_change.txt"
LAMBDA = 0.5
SEED = 2025

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
sys.path.insert(0, config["remote_root"])

import run_colab_worker as worker

print(json.dumps(worker.run_one(PROMPT, LAMBDA, SEED), indent=2), flush=True)
