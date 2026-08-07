"""Run the matched strength-zero and strength-one role-action conditions."""

import json
import sys
from pathlib import Path

PROMPT = "2b_transfer_roles.txt"
SEED = 2025
STRENGTHS = (0.0, 1.0)

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
sys.path.insert(0, config["remote_root"])

import run_colab_worker as worker

for strength in STRENGTHS:
    print(f"Running {PROMPT} | role bias={strength} | seed={SEED}", flush=True)
    result = worker.run_role_action(PROMPT, strength, SEED)
    print(json.dumps(result, indent=2), flush=True)
