"""Load one worker for the character-conditioned action-binding ablation."""

import json
import sys
from pathlib import Path

ACTION_BINDING_CONDITIONS = "0:0,1:0,1:0.5,2:1"

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
            "2025,2026,2027,2028,2029",
            action_binding_conditions=ACTION_BINDING_CONDITIONS,
        ),
        indent=2,
    ),
    flush=True,
)
