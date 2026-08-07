"""Read one value from the local manual Colab JSON configuration."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: read_manual_config.py CONFIG.json KEY")
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    value = config.get(sys.argv[2])
    if value is None or value == "":
        raise SystemExit(f"Missing required configuration key: {sys.argv[2]}")
    print(value)


if __name__ == "__main__":
    main()
