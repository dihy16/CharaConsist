#!/usr/bin/env python3
"""Bootstrap and stream the remote CharaConsist runner inside a Colab cell."""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--init-mode", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    prompts = root / "prompts_batch"
    prompts.mkdir(exist_ok=True)
    with tarfile.open(root / "prompts.tar.gz", "r:gz") as archive:
        archive.extractall(prompts)

    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "run_colab_remote.py"),
            "--root",
            str(root),
            "--prompts-dir",
            str(prompts),
            "--model-path",
            args.model_path,
            "--model-repo",
            args.model_repo,
            "--init-mode",
            args.init_mode,
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)

    exit_code = process.wait()
    (root / "run-exit-code.txt").write_text(str(exit_code), encoding="utf-8")
    return 0


if __name__ == "__main__":
    main()
