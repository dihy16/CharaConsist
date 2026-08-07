"""Extract the source archive uploaded by the manual-run instructions."""

import json
import subprocess
from pathlib import Path

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
remote_root = Path(config["remote_root"])
archive = remote_root / "source.tar.gz"

subprocess.run(["tar", "-xzf", str(archive), "-C", str(remote_root)], check=True)
print(f"Extracted {archive}", flush=True)
