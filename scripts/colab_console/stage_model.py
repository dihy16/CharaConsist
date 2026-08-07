"""Copy the mounted Google Drive checkpoint to temporary local VM storage."""

import json
import subprocess
import time
from pathlib import Path

config = json.loads(Path("/content/characonsist_console_config.json").read_text())
model_drive = config["model_drive"]
model_path = config["model_path"]

Path(model_path).mkdir(parents=True, exist_ok=True)
print(f"Starting rsync: {model_drive} -> {model_path}", flush=True)
started_at = time.monotonic()
process = subprocess.Popen(
    [
        "rsync",
        "-a",
        "--info=progress2",
        "--outbuf=L",
        f"{model_drive}/",
        f"{model_path}/",
    ]
)

while process.poll() is None:
    copied_bytes = sum(item.stat().st_size for item in Path(model_path).rglob("*") if item.is_file())
    elapsed_seconds = int(time.monotonic() - started_at)
    print(
        f"[model-stage] copied={copied_bytes / 1024 ** 3:.2f} GiB "
        f"elapsed={elapsed_seconds}s",
        flush=True,
    )
    time.sleep(30)

if process.returncode:
    raise subprocess.CalledProcessError(process.returncode, process.args)

print("Model staging complete.", flush=True)
