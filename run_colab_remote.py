#!/usr/bin/env python3
"""Remote half of run_colab.sh, executed inside the Colab VM."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def model_snapshot_complete(model_path: Path) -> bool:
    """Check that every shard named by a Hugging Face weight index exists."""
    if not (model_path / "model_index.json").is_file():
        return False

    for index_path in model_path.rglob("*.index.json"):
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        weight_map = index_data.get("weight_map", {})
        for shard_name in set(weight_map.values()):
            shard_path = index_path.parent / shard_name
            if not shard_path.is_file():
                print(f"Missing model shard: {shard_path}", flush=True)
                return False
    return True


def download_model_snapshot(model_path: Path, model_repo: str, hf_token: str) -> None:
    """Download a missing snapshot without passing a token through a CLI argv."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub>=1.0,<2.0"],
            check=True,
        )
        from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=model_repo,
            local_dir=str(model_path),
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face denied or failed the download of {model_repo}. "
            "Confirm that the token has read access and that its owner has "
            "accepted the model license on Hugging Face."
        ) from exc


def ensure_model_snapshot(model_path: Path, model_repo: str, hf_token: str | None) -> bool:
    """Ensure a usable model snapshot exists; return whether it was downloaded."""
    if model_snapshot_complete(model_path):
        return False

    if not hf_token:
        raise RuntimeError(
            f"Model snapshot is missing or incomplete: {model_path}. Set "
            f"HF_TOKEN before running the wrapper so it can resume {model_repo}."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_repo} to {model_path}...", flush=True)
    download_model_snapshot(model_path, model_repo, hf_token)
    if not model_snapshot_complete(model_path):
        raise RuntimeError(
            f"The download finished but the model snapshot is still incomplete: {model_path}"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--init-mode", required=True, choices=["0", "1", "2", "3"])
    args = parser.parse_args()

    root = Path(args.root)
    prompts_dir = Path(args.prompts_dir)
    model_path = Path(args.model_path)
    token_path = root / ".hf_token"

    # Read and immediately remove the uploaded credential. It is needed only
    # when the gated model is absent from the persistent Drive folder.
    hf_token = None
    if token_path.is_file():
        hf_token = token_path.read_text(encoding="utf-8").strip()
        token_path.unlink()

    ensure_model_snapshot(model_path, args.model_repo, hf_token)

    # Install the complete Hugging Face inference stack as a unit. Installing
    # only huggingface_hub can leave Colab's newer transformers incompatible
    # with this project's Diffusers 0.32.x code.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(root / "requirements-colab.txt"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch, diffusers, transformers, huggingface_hub; "
                "print('Runtime versions:', torch.__version__, diffusers.__version__, "
                "transformers.__version__, huggingface_hub.__version__, flush=True)"
            ),
        ],
        check=True,
    )

    prompt_files = sorted(prompts_dir.rglob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt files found in {prompts_dir}")

    return subprocess.run(
        [
            sys.executable,
            "run_batch_inference.py",
            "--root", str(root),
            "--prompts-dir", str(prompts_dir),
            "--model-path", str(model_path),
            "--init-mode", args.init_mode,
            "--results-dir", str(root / "results" / "bg_fg"),
            "--summary", str(root / "results" / "batch-summary.json"),
            "--save-mask",
        ],
        cwd=root,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
