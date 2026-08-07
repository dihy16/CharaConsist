"""Read one value from a temporary manual Colab task specification."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: read_manual_task.py TASK.json KEY")
    task = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    value = task.get(sys.argv[2])
    if value is None or value == "":
        raise SystemExit(f"Missing required task key: {sys.argv[2]}")
    print(value)


if __name__ == "__main__":
    main()
