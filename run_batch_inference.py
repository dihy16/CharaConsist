#!/usr/bin/env python3
"""Run independent prompt files through one loaded CharaConsist pipeline."""

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path


MARKER_NAME = "_SUCCESS.json"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-mode", required=True, type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2025)
    return parser


def inference_settings(args):
    return {
        "model_path": args.model_path,
        "init_mode": args.init_mode,
        "gpu_ids": args.gpu_ids,
        "save_mask": args.save_mask,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
    }


def success_record(prompt_file: Path, settings):
    digest = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    return {"prompt_sha256": digest, "settings": settings}


def marker_matches(output_dir: Path, record) -> bool:
    try:
        return json.loads((output_dir / MARKER_NAME).read_text(encoding="utf-8")) == record
    except (OSError, json.JSONDecodeError):
        return False


def write_summary(path: Path, total, generated, skipped, failures):
    summary = {
        "total": total,
        "succeeded": generated + skipped,
        "generated": generated,
        "skipped": skipped,
        "failed": failures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    prompts_dir = Path(args.prompts_dir)
    results_dir = Path(args.results_dir)
    summary_path = Path(args.summary)
    prompt_files = sorted(prompts_dir.rglob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt files found in {prompts_dir}")

    # Import after parsing so this wrapper remains unit-testable without the
    # heavyweight inference dependencies installed.
    from inference import initialize_pipeline, reset_runtime_state, run_prompt_file

    settings = inference_settings(args)
    staging_root = root / "results_in_progress"
    failures = []
    generated = 0
    skipped = 0
    print(f"Loading the pipeline once for {len(prompt_files)} prompt file(s).", flush=True)
    pipe = initialize_pipeline(args)
    try:
        for index, prompt_file in enumerate(prompt_files, start=1):
            relative = prompt_file.relative_to(prompts_dir).with_suffix("")
            output_dir = results_dir / relative
            record = success_record(prompt_file, settings)
            if marker_matches(output_dir, record):
                skipped += 1
                print(f"[{index}/{len(prompt_files)}] Skipping completed {prompt_file}", flush=True)
                write_summary(summary_path, len(prompt_files), generated, skipped, failures)
                continue

            staging_dir = staging_root / relative
            shutil.rmtree(staging_dir, ignore_errors=True)
            print(f"[{index}/{len(prompt_files)}] Processing {prompt_file}", flush=True)
            try:
                run_args = argparse.Namespace(**vars(args))
                run_args.prompts_file = str(prompt_file)
                run_args.out_dir = str(staging_dir)
                run_args.use_interpolate = False
                run_args.action_gate_strength = 1.0
                run_args.share_bg = False
                run_args.save_all_steps = False
                run_args.mix_mode = False
                run_prompt_file(pipe, run_args)
                (staging_dir / MARKER_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(output_dir, ignore_errors=True)
                staging_dir.replace(output_dir)
                generated += 1
                print(f"[{index}/{len(prompt_files)}] Saved results for {prompt_file}", flush=True)
            except Exception as exc:
                shutil.rmtree(staging_dir, ignore_errors=True)
                failures.append({
                    "prompt_file": str(prompt_file.relative_to(prompts_dir)),
                    "exit_code": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"[{index}/{len(prompt_files)}] FAILED {prompt_file}: {exc}; continuing", flush=True)
                traceback.print_exc()
            finally:
                reset_runtime_state(pipe, args)
                write_summary(summary_path, len(prompt_files), generated, skipped, failures)
    finally:
        reset_runtime_state(pipe, args)

    print(f"Batch complete: {generated + skipped}/{len(prompt_files)} prompt files succeeded", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
