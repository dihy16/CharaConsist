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

    model_is_ready = model_snapshot_complete(model_path)
    # With a token available, always ask `hf download` to reconcile the local
    # snapshot. It skips completed files and resumes missing/partial shards.
    if hf_token or not model_is_ready:
        if not hf_token:
            raise RuntimeError(
                f"Model snapshot is missing or incomplete: {model_path}. Set "
                f"HF_TOKEN before running the wrapper so it can resume {args.model_repo}."
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.model_repo} to {model_path}...", flush=True)
        hf_cli = shutil.which("hf")
        if not hf_cli:
            # Install only the modern download CLI temporarily. The compatible
            # inference package set is installed together after the download.
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub>=1.0,<2.0"],
                check=True,
            )
            hf_cli = shutil.which("hf")
        if not hf_cli:
            raise RuntimeError("The Hugging Face 'hf' CLI was not installed correctly.")

        try:
            subprocess.run(
                [hf_cli, "auth", "login", "--token", hf_token],
                check=True,
            )
            subprocess.run(
                [hf_cli, "download", args.model_repo, "--local-dir", str(model_path)],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Hugging Face denied or failed the download of {args.model_repo}. "
                "Confirm that the token has read access and that its owner has "
                "accepted the model license on Hugging Face."
            ) from exc

    if not model_snapshot_complete(model_path):
        raise RuntimeError(
            f"The download finished but the model snapshot is still incomplete: {model_path}"
        )

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

    failures = []
    for index, prompt_file in enumerate(prompt_files, start=1):
        relative_output = prompt_file.relative_to(prompts_dir).with_suffix("")
        output_dir = root / "results" / "bg_fg" / relative_output
        staging_dir = root / "results_in_progress" / relative_output
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"[{index}/{len(prompt_files)}] Processing {prompt_file}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "inference.py",
                "--init_mode",
                args.init_mode,
                "--prompts_file",
                str(prompt_file),
                "--model_path",
                str(model_path),
                "--out_dir",
                str(staging_dir),
                "--save_mask",
            ],
            cwd=root,
            check=False,
        )
        if completed.returncode == 0:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            staging_dir.replace(output_dir)
            print(f"[{index}/{len(prompt_files)}] Saved results for {prompt_file}", flush=True)
        else:
            shutil.rmtree(staging_dir, ignore_errors=True)
            failures.append(
                {
                    "prompt_file": str(prompt_file.relative_to(prompts_dir)),
                    "exit_code": completed.returncode,
                }
            )
            print(
                f"[{index}/{len(prompt_files)}] FAILED {prompt_file} "
                f"(exit code {completed.returncode}); continuing",
                file=sys.stderr,
                flush=True,
            )

    summary = {
        "total": len(prompt_files),
        "succeeded": len(prompt_files) - len(failures),
        "failed": failures,
    }
    summary_path = root / "results" / "batch-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Batch complete: {summary['succeeded']}/{summary['total']} prompt files succeeded",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
