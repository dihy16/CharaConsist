import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from characonsist.runners.batch import (
    MARKER_NAME,
    main,
    marker_matches,
    merge_delivery_failures,
    success_record,
    write_summary,
)


class BatchInferenceTests(unittest.TestCase):
    def test_success_marker_requires_matching_prompt_and_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = root / "scene.txt"
            output = root / "output"
            prompt.write_text("background#subject#action\n", encoding="utf-8")
            record = success_record(prompt, {"seed": 2025})
            output.mkdir()
            (output / MARKER_NAME).write_text(json.dumps(record), encoding="utf-8")

            self.assertTrue(marker_matches(output, record))
            prompt.write_text("background#subject#different action\n", encoding="utf-8")
            self.assertFalse(marker_matches(output, success_record(prompt, {"seed": 2025})))
            self.assertFalse(marker_matches(output, success_record(prompt, {"seed": 7})))

    def test_summary_counts_generated_and_resumed_files_as_succeeded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "batch-summary.json"
            write_summary(summary_path, total=3, generated=1, skipped=1, failures=[{"prompt_file": "bad.txt"}])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["succeeded"], 2)
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(summary["failed"], [{"prompt_file": "bad.txt"}])

    def test_local_delivery_failure_makes_remote_success_retryable(self):
        merged = merge_delivery_failures(
            {"total": 3, "succeeded": 3, "generated": 3, "skipped": 0, "failed": []},
            [{
                "prompt_file": "scene.txt",
                "exit_code": 1,
                "source": "local_result_delivery",
                "error": "downloaded result did not pass marker verification",
            }],
        )

        self.assertEqual(merged["succeeded"], 2)
        self.assertEqual(merged["generated"], 2)
        self.assertEqual(merged["failed"][0]["prompt_file"], "scene.txt")

    def test_delivery_failure_updates_its_condition(self):
        merged = merge_delivery_failures(
            {
                "total": 2,
                "succeeded": 2,
                "generated": 2,
                "skipped": 0,
                "failed": [],
                "conditions": {
                    "lambda_0p50/seed_2025": {
                        "generated": 2,
                        "skipped": 0,
                        "failed": 0,
                    }
                },
            },
            [{
                "prompt_file": "lambda_0p50/seed_2025/bg_fg/scene",
                "condition": "lambda_0p50/seed_2025",
                "source": "local_result_delivery",
                "error": "archive missing",
            }],
        )

        condition = merged["conditions"]["lambda_0p50/seed_2025"]
        self.assertEqual(condition["generated"], 1)
        self.assertEqual(condition["failed"], 1)

    def test_main_runs_each_condition_with_interpolation_and_maps(self):
        calls = []
        resets = []

        def run_prompt_file(_pipe, args):
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            calls.append((
                args.action_gate_strength,
                args.seed,
                args.use_interpolate,
                args.save_action_maps,
                args.save_merge_maps,
            ))

        fake_inference = types.SimpleNamespace(
            initialize_pipeline=lambda _args: object(),
            reset_runtime_state=lambda _pipe, args: resets.append(getattr(args, "seed", None)),
            run_prompt_file=run_prompt_file,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "scene.txt").write_text("background#subject#action\n", encoding="utf-8")
            results = root / "results"
            summary = root / "summary.json"
            argv = [
                "run_batch_inference.py",
                "--root", str(root),
                "--prompts-dir", str(prompts),
                "--model-path", "model",
                "--init-mode", "0",
                "--results-dir", str(results),
                "--summary", str(summary),
                "--action-gate-strengths", "0,0.5",
                "--seeds", "7,8",
            ]

            with patch.dict(sys.modules, {"inference": fake_inference}), patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)

            self.assertEqual(
                calls,
                [
                    (0.0, 7, True, True, True),
                    (0.0, 8, True, True, True),
                    (0.5, 7, True, True, True),
                    (0.5, 8, True, True, True),
                ],
            )
            self.assertTrue(results.joinpath("lambda_0p50", "seed_8", "bg_fg", "scene", MARKER_NAME).is_file())
            self.assertGreaterEqual(len(resets), len(calls))

    def test_main_runs_component_modes_with_the_same_seed(self):
        calls = []

        def run_prompt_file(_pipe, args):
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            calls.append((args.consistency_mode, args.seed, args.use_interpolate, args.action_gate_strength))

        fake_inference = types.SimpleNamespace(
            initialize_pipeline=lambda _args: object(),
            reset_runtime_state=lambda _pipe, _args: None,
            run_prompt_file=run_prompt_file,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "scene.txt").write_text("background#subject#action\n", encoding="utf-8")
            argv = [
                "run_batch_inference.py",
                "--root", str(root),
                "--prompts-dir", str(prompts),
                "--model-path", "model",
                "--init-mode", "0",
                "--results-dir", str(root / "results"),
                "--summary", str(root / "summary.json"),
                "--consistency-modes", "prompt_only,attention_only,full",
                "--seeds", "2025",
            ]
            with patch.dict(sys.modules, {"inference": fake_inference}), patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)

            self.assertEqual(
                calls,
                [
                    ("prompt_only", 2025, False, 0.0),
                    ("attention_only", 2025, False, 0.0),
                    ("full", 2025, True, 0.0),
                ],
            )
            for mode in ("prompt_only", "attention_only", "full"):
                self.assertTrue(
                    root.joinpath(
                        "results", "component_ablation", mode, "seed_2025",
                        "bg_fg", "scene", MARKER_NAME,
                    ).is_file()
                )
