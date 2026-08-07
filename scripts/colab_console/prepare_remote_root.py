"""Create the remote source directory for a manual run."""

import json
from pathlib import Path

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
remote_root = Path(config["remote_root"])
remote_root.mkdir(parents=True, exist_ok=True)
print(f"Remote root ready: {remote_root}", flush=True)
