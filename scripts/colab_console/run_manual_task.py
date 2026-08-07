"""Dispatch the temporary task specification through the initialized worker."""

import json
import sys
from pathlib import Path


config = json.loads(Path("/content/characonsist_console_config.json").read_text())
task = json.loads(Path("/content/characonsist_console_task.json").read_text())
sys.path.insert(0, config["remote_root"])

import run_colab_worker as worker


method = getattr(worker, task["method"])
if task["method"] == "run_action_binding":
    result = method(task["prompt"], task["value"][0], task["value"][1], task["seed"])
elif task["method"] == "run_entity_routing":
    result = method(
        task["prompt"], task["value"][0], task["value"][1], task["value"][2], task["seed"]
    )
else:
    result = method(task["prompt"], task["value"], task["seed"])
print("CHARACONSIST_TASK_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
