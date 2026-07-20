#!/usr/bin/env python3
"""Remote half of run_colab.sh, executed inside the Colab VM."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


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

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt")],
        check=True,
    )

    # Read and immediately remove the uploaded credential. It is needed only
    # when the gated model is absent from the persistent Drive folder.
    hf_token = None
    if token_path.is_file():
        hf_token = token_path.read_text(encoding="utf-8").strip()
        token_path.unlink()

    model_is_ready = model_path.is_dir() and (model_path / "model_index.json").is_file()
    if not model_is_ready:
        if not hf_token:
            raise RuntimeError(
                f"Model directory is missing: {model_path}. Set HF_TOKEN before "
                f"running the wrapper so it can download {args.model_repo}."
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.model_repo} to {model_path}...", flush=True)
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

    prompt_files = sorted(prompts_dir.rglob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt files found in {prompts_dir}")

    for index, prompt_file in enumerate(prompt_files, start=1):
        relative_output = prompt_file.relative_to(prompts_dir).with_suffix("")
        output_dir = root / "results" / "bg_fg" / relative_output
        print(f"[{index}/{len(prompt_files)}] Processing {prompt_file}", flush=True)
        subprocess.run(
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
                str(output_dir),
                "--save_mask",
            ],
            cwd=root,
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
