"""Persistent, one-model worker kept alive in the Colab kernel."""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

from characonsist.runners.batch import (
    MARKER_NAME,
    inference_settings,
    marker_matches,
    success_record,
    write_summary,
)
from run_colab_remote import ensure_model_snapshot
from characonsist.experiments.conditions import (
    build_component_conditions,
    build_sweep_conditions,
    find_component_condition,
    find_condition,
)


STATE = {}


def _summary_path():
    return STATE["root"] / "results" / "batch-summary.json"


def _write_summary():
    write_summary(
        _summary_path(),
        STATE["total_tasks"],
        STATE["generated"],
        STATE["skipped"],
        STATE["failures"],
        STATE["condition_stats"],
    )


def _archive_result(output_dir: Path, relative_output: Path) -> Path:
    """Make a file transfer unit because colab-cli cannot download folders."""
    archive_relative = relative_output.parent / f"{relative_output.name}.tar.gz"
    archive_path = STATE["root"] / "result_archives" / archive_relative
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=str(relative_output))
    return archive_path


def start(
    root: str,
    model_path: str,
    model_repo: str,
    init_mode: int,
    action_gate_strengths,
    seeds,
    consistency_modes=None,
):
    """Install dependencies and initialize the pipeline once in this kernel."""
    if STATE:
        return {"status": "ready", "detail": "worker already initialized"}

    root_path = Path(root)
    prompts_dir = root_path / "prompts_batch"
    prompts_dir.mkdir(exist_ok=True)
    with tarfile.open(root_path / "prompts.tar.gz", "r:gz") as archive:
        archive.extractall(prompts_dir)

    token_path = root_path / ".hf_token"
    token = token_path.read_text(encoding="utf-8").strip() if token_path.is_file() else None
    token_path.unlink(missing_ok=True)
    ensure_model_snapshot(Path(model_path), model_repo, token)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(root_path / "requirements-colab.txt")], check=True)

    from characonsist.inference import initialize_pipeline

    config = argparse.Namespace(
        model_path=model_path,
        init_mode=init_mode,
        gpu_ids=[0],
        save_mask=True,
        save_points=True,
        height=768,
        width=768,
        seed=2025,
        use_interpolate=True,
        consistency_mode="full",
        action_gate_strength=0.0,
        save_action_maps=True,
        save_merge_maps=True,
        share_bg=False,
        save_all_steps=False,
        mix_mode=False,
    )
    prompt_files = sorted(prompts_dir.rglob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt files found in {prompts_dir}")
    experiment_kind = "component_ablation" if consistency_modes else "lambda_sweep"
    conditions = (
        build_component_conditions(consistency_modes, seeds)
        if consistency_modes
        else build_sweep_conditions(action_gate_strengths, seeds)
    )
    condition_stats = {
        condition.key: {
            "action_gate_strength": getattr(condition, "action_gate_strength", 0.0),
            "consistency_mode": getattr(condition, "consistency_mode", "full"),
            "seed": condition.seed,
            "total": len(prompt_files),
            "generated": 0,
            "skipped": 0,
            "failed": 0,
        }
        for condition in conditions
    }
    STATE.update(
        root=root_path,
        prompts_dir=prompts_dir,
        config=config,
        pipe=initialize_pipeline(config),
        prompt_files=prompt_files,
        conditions=conditions,
        experiment_kind=experiment_kind,
        condition_stats=condition_stats,
        total_tasks=len(prompt_files) * len(conditions),
        generated=0,
        skipped=0,
        failures=[],
    )
    _write_summary()
    return {
        "status": "ready",
        "total": STATE["total_tasks"],
        "prompt_files": len(prompt_files),
        "conditions": len(conditions),
        "experiment_kind": experiment_kind,
    }


def _run_condition(relative_prompt_path, condition):
    """Run a resolved condition without reloading FLUX."""
    prompt_file = (STATE["prompts_dir"] / relative_prompt_path).resolve()
    if STATE["prompts_dir"] not in prompt_file.parents or not prompt_file.is_file():
        raise ValueError(f"invalid prompt path: {relative_prompt_path}")
    stats = STATE["condition_stats"][condition.key]
    relative = prompt_file.relative_to(STATE["prompts_dir"]).with_suffix("")
    relative_output = condition.output_prefix / relative
    output_dir = STATE["root"] / "results" / relative_output
    run_args = argparse.Namespace(**vars(STATE["config"]))
    run_args.seed = condition.seed
    run_args.action_gate_strength = getattr(condition, "action_gate_strength", 0.0)
    run_args.consistency_mode = getattr(condition, "consistency_mode", "full")
    run_args.use_interpolate = run_args.consistency_mode == "full"
    record = success_record(prompt_file, inference_settings(run_args))
    if marker_matches(output_dir, record):
        STATE["skipped"] += 1
        stats["skipped"] += 1
        _write_summary()
        archive_path = _archive_result(output_dir, relative_output)
        return {"status": "skipped", "relative_output": str(relative_output), "archive_path": str(archive_path)}

    staging_dir = STATE["root"] / "results_in_progress" / relative_output
    shutil.rmtree(staging_dir, ignore_errors=True)
    from characonsist.inference import reset_runtime_state, run_prompt_file
    try:

        run_args.prompts_file = str(prompt_file)
        run_args.out_dir = str(staging_dir)
        run_prompt_file(STATE["pipe"], run_args)
        (staging_dir / MARKER_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        staging_dir.replace(output_dir)
        archive_path = _archive_result(output_dir, relative_output)
        STATE["generated"] += 1
        stats["generated"] += 1
        return {"status": "success", "relative_output": str(relative_output), "archive_path": str(archive_path)}
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        STATE["failures"].append({
            "prompt_file": str(prompt_file.relative_to(STATE["prompts_dir"])),
            "condition": condition.key,
            "action_gate_strength": run_args.action_gate_strength,
            "consistency_mode": run_args.consistency_mode,
            "seed": condition.seed,
            "exit_code": 1,
            "error": f"{type(exc).__name__}: {exc}",
        })
        stats["failed"] += 1
        traceback.print_exc()
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        reset_runtime_state(STATE["pipe"], STATE["config"])
        _write_summary()


def run_one(relative_prompt_path: str, action_gate_strength: float, seed: int):
    """Run one lambda-sweep task without reloading FLUX."""
    if not STATE:
        raise RuntimeError("worker has not been initialized")
    if STATE["experiment_kind"] != "lambda_sweep":
        raise RuntimeError("worker was initialized for a component ablation")
    condition = find_condition(STATE["conditions"], action_gate_strength, seed)
    return _run_condition(relative_prompt_path, condition)


def run_component(relative_prompt_path: str, consistency_mode: str, seed: int):
    """Run one seed-matched component-ablation task without reloading FLUX."""
    if not STATE:
        raise RuntimeError("worker has not been initialized")
    if STATE["experiment_kind"] != "component_ablation":
        raise RuntimeError("worker was initialized for a lambda sweep")
    condition = find_component_condition(STATE["conditions"], consistency_mode, seed)
    return _run_condition(relative_prompt_path, condition)


def record_local_skip(relative_prompt_path: str, action_gate_strength: float, seed: int):
    """Include a locally verified resumed file in the remote batch summary."""
    if not STATE:
        raise RuntimeError("worker has not been initialized")
    prompt_file = (STATE["prompts_dir"] / relative_prompt_path).resolve()
    if STATE["prompts_dir"] not in prompt_file.parents or not prompt_file.is_file():
        raise ValueError(f"invalid prompt path: {relative_prompt_path}")
    condition = find_condition(STATE["conditions"], action_gate_strength, seed)
    STATE["skipped"] += 1
    STATE["condition_stats"][condition.key]["skipped"] += 1
    _write_summary()
    relative = prompt_file.relative_to(STATE["prompts_dir"]).with_suffix("")
    return {"status": "skipped", "relative_output": str(condition.output_prefix / relative)}


def record_component_local_skip(relative_prompt_path: str, consistency_mode: str, seed: int):
    """Include a locally verified component result in the remote summary."""
    if not STATE:
        raise RuntimeError("worker has not been initialized")
    prompt_file = (STATE["prompts_dir"] / relative_prompt_path).resolve()
    if STATE["prompts_dir"] not in prompt_file.parents or not prompt_file.is_file():
        raise ValueError(f"invalid prompt path: {relative_prompt_path}")
    condition = find_component_condition(STATE["conditions"], consistency_mode, seed)
    STATE["skipped"] += 1
    STATE["condition_stats"][condition.key]["skipped"] += 1
    _write_summary()
    relative = prompt_file.relative_to(STATE["prompts_dir"]).with_suffix("")
    return {"status": "skipped", "relative_output": str(condition.output_prefix / relative)}


def finish():
    if not STATE:
        return {"status": "not_started"}
    _write_summary()
    return {
        "status": "complete",
        "total": STATE["total_tasks"],
        "succeeded": STATE["generated"] + STATE["skipped"],
        "failed": len(STATE["failures"]),
    }
