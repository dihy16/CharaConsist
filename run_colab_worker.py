"""Persistent, one-model worker kept alive in the Colab kernel."""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

from run_batch_inference import MARKER_NAME, inference_settings, marker_matches, success_record, write_summary
from run_colab_remote import ensure_model_snapshot


STATE = {}


def _summary_path():
    return STATE["root"] / "results" / "batch-summary.json"


def _write_summary():
    write_summary(
        _summary_path(),
        len(STATE["prompt_files"]),
        STATE["generated"],
        STATE["skipped"],
        STATE["failures"],
    )


def _archive_result(output_dir: Path, relative: Path) -> Path:
    """Make a file transfer unit because colab-cli cannot download folders."""
    archive_path = STATE["root"] / "result_archives" / relative.with_suffix(".tar.gz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=str(Path("bg_fg") / relative))
    return archive_path


def start(root: str, model_path: str, model_repo: str, init_mode: int):
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

    from inference import initialize_pipeline

    config = argparse.Namespace(
        model_path=model_path,
        init_mode=init_mode,
        gpu_ids=[0],
        save_mask=True,
        save_points=True,
        height=1024,
        width=1024,
        seed=2025,
        use_interpolate=False,
        action_gate_strength=1.0,
        share_bg=False,
        save_all_steps=False,
        mix_mode=False,
    )
    prompt_files = sorted(prompts_dir.rglob("*.txt"))
    if not prompt_files:
        raise RuntimeError(f"No .txt files found in {prompts_dir}")
    STATE.update(
        root=root_path,
        prompts_dir=prompts_dir,
        config=config,
        settings=inference_settings(config),
        pipe=initialize_pipeline(config),
        prompt_files=prompt_files,
        generated=0,
        skipped=0,
        failures=[],
    )
    _write_summary()
    return {"status": "ready", "total": len(prompt_files)}


def run_one(relative_prompt_path: str):
    """Run and finalize one original prompt file without reloading FLUX."""
    if not STATE:
        raise RuntimeError("worker has not been initialized")

    prompt_file = (STATE["prompts_dir"] / relative_prompt_path).resolve()
    if STATE["prompts_dir"] not in prompt_file.parents or not prompt_file.is_file():
        raise ValueError(f"invalid prompt path: {relative_prompt_path}")
    relative = prompt_file.relative_to(STATE["prompts_dir"]).with_suffix("")
    output_dir = STATE["root"] / "results" / "bg_fg" / relative
    record = success_record(prompt_file, STATE["settings"])
    if marker_matches(output_dir, record):
        STATE["skipped"] += 1
        _write_summary()
        archive_path = _archive_result(output_dir, relative)
        return {"status": "skipped", "relative_output": str(relative), "archive_path": str(archive_path)}

    staging_dir = STATE["root"] / "results_in_progress" / relative
    shutil.rmtree(staging_dir, ignore_errors=True)
    from inference import reset_runtime_state, run_prompt_file
    try:

        run_args = argparse.Namespace(**vars(STATE["config"]))
        run_args.prompts_file = str(prompt_file)
        run_args.out_dir = str(staging_dir)
        run_prompt_file(STATE["pipe"], run_args)
        (staging_dir / MARKER_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        staging_dir.replace(output_dir)
        archive_path = _archive_result(output_dir, relative)
        STATE["generated"] += 1
        return {"status": "success", "relative_output": str(relative), "archive_path": str(archive_path)}
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        STATE["failures"].append({
            "prompt_file": str(prompt_file.relative_to(STATE["prompts_dir"])),
            "exit_code": 1,
            "error": f"{type(exc).__name__}: {exc}",
        })
        traceback.print_exc()
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        reset_runtime_state(STATE["pipe"], STATE["config"])
        _write_summary()


def record_local_skip(relative_prompt_path: str):
    """Include a locally verified resumed file in the remote batch summary."""
    if not STATE:
        raise RuntimeError("worker has not been initialized")
    prompt_file = (STATE["prompts_dir"] / relative_prompt_path).resolve()
    if STATE["prompts_dir"] not in prompt_file.parents or not prompt_file.is_file():
        raise ValueError(f"invalid prompt path: {relative_prompt_path}")
    STATE["skipped"] += 1
    _write_summary()
    return {"status": "skipped", "relative_output": str(prompt_file.relative_to(STATE["prompts_dir"]).with_suffix(""))}


def finish():
    if not STATE:
        return {"status": "not_started"}
    _write_summary()
    return {
        "status": "complete",
        "total": len(STATE["prompt_files"]),
        "succeeded": STATE["generated"] + STATE["skipped"],
        "failed": len(STATE["failures"]),
    }
