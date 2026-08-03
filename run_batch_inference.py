#!/usr/bin/env python3
"""Run independent prompt files through one loaded CharaConsist pipeline."""

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path

from sweep_utils import (
    DEFAULT_ACTION_GATE_STRENGTHS,
    DEFAULT_SEEDS,
    build_sweep_conditions,
    parse_action_gate_strengths,
    parse_seeds,
)


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
    parser.add_argument("--save-points", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--action-gate-strengths",
        type=parse_action_gate_strengths,
        default=parse_action_gate_strengths(DEFAULT_ACTION_GATE_STRENGTHS),
        help="Comma-separated lambda values in [0,1].",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds(DEFAULT_SEEDS),
        help="Comma-separated non-negative seeds.",
    )
    return parser


def inference_settings(args):
    return {
        "model_path": args.model_path,
        "init_mode": args.init_mode,
        "gpu_ids": args.gpu_ids,
        "save_mask": args.save_mask,
        "save_points": getattr(args, "save_points", False),
        "save_action_maps": getattr(args, "save_action_maps", False),
        "save_merge_maps": getattr(args, "save_merge_maps", False),
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "use_interpolate": getattr(args, "use_interpolate", False),
        "action_gate_strength": getattr(args, "action_gate_strength", None),
        "experiment_schema": 2,
    }


def success_record(prompt_file: Path, settings):
    digest = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    return {"prompt_sha256": digest, "settings": settings}


def marker_matches(output_dir: Path, record) -> bool:
    try:
        return json.loads((output_dir / MARKER_NAME).read_text(encoding="utf-8")) == record
    except (OSError, json.JSONDecodeError):
        return False


def write_summary(path: Path, total, generated, skipped, failures, conditions=None):
    summary = {
        "total": total,
        "succeeded": generated + skipped,
        "generated": generated,
        "skipped": skipped,
        "failed": failures,
    }
    if conditions is not None:
        summary["conditions"] = conditions
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def merge_delivery_failures(summary, delivery_failures):
    """Overlay local archive-delivery failures onto a remote batch summary."""
    merged = dict(summary)
    failures = list(summary.get("failed", []))
    existing = {
        (item.get("prompt_file"), item.get("source"), item.get("error"))
        for item in failures
        if isinstance(item, dict)
    }
    failed_prompts = set()
    failed_conditions = {}
    for failure in delivery_failures:
        key = (failure.get("prompt_file"), failure.get("source"), failure.get("error"))
        if key not in existing:
            failures.append(failure)
            existing.add(key)
        if failure.get("prompt_file"):
            failed_prompts.add(failure["prompt_file"])
            failed_conditions[failure["prompt_file"]] = failure.get("condition")

    failed_delivery_count = len(failed_prompts)
    merged["failed"] = failures
    merged["succeeded"] = max(0, int(summary.get("succeeded", 0)) - failed_delivery_count)
    merged["generated"] = max(0, int(summary.get("generated", 0)) - failed_delivery_count)
    condition_stats = merged.get("conditions")
    if isinstance(condition_stats, dict):
        condition_stats = {key: dict(value) for key, value in condition_stats.items()}
        for condition_key in failed_conditions.values():
            if condition_key in condition_stats:
                stats = condition_stats[condition_key]
                if int(stats.get("generated", 0)) > 0:
                    stats["generated"] = int(stats["generated"]) - 1
                else:
                    stats["skipped"] = max(0, int(stats.get("skipped", 0)) - 1)
                stats["failed"] = int(stats.get("failed", 0)) + 1
        merged["conditions"] = condition_stats
    return merged


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    prompts_dir = Path(args.prompts_dir)
    results_dir = Path(args.results_dir)
    summary_path = Path(args.summary)
    prompt_files = sorted(prompts_dir.rglob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt files found in {prompts_dir}")
    conditions = build_sweep_conditions(args.action_gate_strengths, args.seeds)
    total_tasks = len(prompt_files) * len(conditions)

    # Import after parsing so this wrapper remains unit-testable without the
    # heavyweight inference dependencies installed.
    from inference import initialize_pipeline, reset_runtime_state, run_prompt_file

    staging_root = root / "results_in_progress"
    failures = []
    generated = 0
    skipped = 0
    condition_stats = {
        condition.key: {
            "action_gate_strength": condition.action_gate_strength,
            "seed": condition.seed,
            "total": len(prompt_files),
            "generated": 0,
            "skipped": 0,
            "failed": 0,
        }
        for condition in conditions
    }
    print(
        f"Loading the pipeline once for {total_tasks} sweep task(s) "
        f"across {len(prompt_files)} prompt file(s).",
        flush=True,
    )
    pipe = initialize_pipeline(args)
    try:
        task_index = 0
        for condition in conditions:
            stats = condition_stats[condition.key]
            for prompt_file in prompt_files:
                task_index += 1
                relative = prompt_file.relative_to(prompts_dir).with_suffix("")
                condition_relative = condition.output_prefix / relative
                output_dir = results_dir / condition_relative
                run_args = argparse.Namespace(**vars(args))
                run_args.prompts_file = str(prompt_file)
                run_args.out_dir = str(staging_root / condition_relative)
                run_args.seed = condition.seed
                run_args.use_interpolate = True
                run_args.action_gate_strength = condition.action_gate_strength
                run_args.save_action_maps = True
                run_args.save_merge_maps = True
                run_args.share_bg = False
                run_args.save_all_steps = False
                run_args.mix_mode = False
                record = success_record(prompt_file, inference_settings(run_args))
                if marker_matches(output_dir, record):
                    skipped += 1
                    stats["skipped"] += 1
                    print(f"[{task_index}/{total_tasks}] Skipping completed {condition.key}/{relative}", flush=True)
                    write_summary(summary_path, total_tasks, generated, skipped, failures, condition_stats)
                    continue

                staging_dir = Path(run_args.out_dir)
                shutil.rmtree(staging_dir, ignore_errors=True)
                print(f"[{task_index}/{total_tasks}] Processing {condition.key}/{relative}", flush=True)
                try:
                    run_prompt_file(pipe, run_args)
                    (staging_dir / MARKER_NAME).write_text(
                        json.dumps(record, indent=2) + "\n", encoding="utf-8"
                    )
                    output_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.rmtree(output_dir, ignore_errors=True)
                    staging_dir.replace(output_dir)
                    generated += 1
                    stats["generated"] += 1
                    print(f"[{task_index}/{total_tasks}] Saved {condition.key}/{relative}", flush=True)
                except Exception as exc:
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    stats["failed"] += 1
                    failures.append({
                        "prompt_file": str(prompt_file.relative_to(prompts_dir)),
                        "condition": condition.key,
                        "action_gate_strength": condition.action_gate_strength,
                        "seed": condition.seed,
                        "exit_code": 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(
                        f"[{task_index}/{total_tasks}] FAILED {condition.key}/{relative}: {exc}; continuing",
                        flush=True,
                    )
                    traceback.print_exc()
                finally:
                    reset_runtime_state(pipe, run_args)
                    write_summary(summary_path, total_tasks, generated, skipped, failures, condition_stats)
    finally:
        reset_runtime_state(pipe, args)

    print(f"Batch complete: {generated + skipped}/{total_tasks} sweep tasks succeeded", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
