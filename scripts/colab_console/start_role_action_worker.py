"""Load one worker for the matched role-action attention-bias ablation."""

import json
import sys
from pathlib import Path

ROLE_ACTION_BIAS_STRENGTHS = "0,1"

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
sys.path.insert(0, config["remote_root"])

import run_colab_worker as worker

print(
    json.dumps(
        worker.start(
            config["remote_root"],
            config["model_path"],
            config["model_id"],
            0,
            config["action_gate_strengths"],
            config["seeds"],
            role_action_bias_strengths=ROLE_ACTION_BIAS_STRENGTHS,
        ),
        indent=2,
    ),
    flush=True,
)
