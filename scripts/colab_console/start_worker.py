"""Install runtime dependencies and load the CharaConsist worker once."""

import json
import sys
from pathlib import Path

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
        ),
        indent=2,
    ),
    flush=True,
)
