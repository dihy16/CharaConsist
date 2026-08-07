"""Configure one manual CharaConsist Colab session.

Copy this file to ``local_config.py``, set the paths, then execute it with
``colab exec -s "$SESSION" -f scripts/colab_console/local_config.py``.
The configuration is stored only on the temporary Colab VM.
"""

import json
from pathlib import Path

CONFIG_PATH = Path("/content/characonsist_console_config.json")

CONFIG = {
    "model_drive": "/content/drive/MyDrive/Colab/models/FLUX.1-dev",
    "model_path": "/content/drive/MyDrive/Colab/models/FLUX.1-dev",
    "remote_root": "/content/CharaConsist",
    "model_id": "black-forest-labs/FLUX.1-dev",
    "action_gate_strengths": "0,0.5,1",
    "seeds": "2025",
}

CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2) + "\n")
print(f"Saved manual-run configuration to {CONFIG_PATH}", flush=True)
print(json.dumps(CONFIG, indent=2), flush=True)
