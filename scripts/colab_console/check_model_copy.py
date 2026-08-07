"""Report source and staged checkpoint sizes for a manual Colab session."""

import json
from pathlib import Path

config = json.loads(Path("/content/characonsist_console_config.json").read_text())


def summarize(label: str, raw_path: str) -> None:
    path = Path(raw_path)
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    size_bytes = sum(item.stat().st_size for item in files)
    print(
        f"{label}: exists={path.exists()} files={len(files)} "
        f"size_gb={size_bytes / 1024 ** 3:.2f}",
        flush=True,
    )


summarize("Drive", config["model_drive"])
summarize("Local", config["model_path"])
