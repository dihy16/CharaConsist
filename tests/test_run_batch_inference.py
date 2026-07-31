import json
import tempfile
import unittest
from pathlib import Path

from run_batch_inference import MARKER_NAME, marker_matches, success_record, write_summary


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

