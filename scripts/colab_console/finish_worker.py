"""Finalize the remote worker and write its batch summary."""

import json
import sys
from pathlib import Path

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
sys.path.insert(0, config["remote_root"])

import run_colab_worker as worker

print(json.dumps(worker.finish(), indent=2), flush=True)
