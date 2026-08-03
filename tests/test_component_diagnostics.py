import json
import tempfile
import unittest
from pathlib import Path

from characonsist.diagnostics.components import (
    finalize_component_state,
    normalize_consistency_mode,
    save_component_trace,
)


class ComponentDiagnosticsTests(unittest.TestCase):
    def test_legacy_interpolation_maps_to_expected_mode(self):
        self.assertEqual(normalize_consistency_mode(None, False), "attention_only")
        self.assertEqual(normalize_consistency_mode(None, True), "full")

    def test_mode_invariants(self):
        prompt = finalize_component_state({}, "prompt_only", 0)
        attention = finalize_component_state(
            {"identity_attention_invocations": 2}, "attention_only", 0
        )
        full = finalize_component_state(
            {"identity_attention_invocations": 2, "adaptive_merge_invocations": 1},
            "full",
            0,
        )
        self.assertEqual(prompt["status"], "PASS")
        self.assertEqual(attention["adaptive_merge_invocations"], 0)
        self.assertEqual(full["adaptive_merge_invocations"], 1)
        with self.assertRaises(RuntimeError):
            finalize_component_state({}, "attention_only", 0)
        with self.assertRaises(RuntimeError):
            finalize_component_state(
                {"identity_attention_invocations": 1}, "full", 0
            )

    def test_saves_prompt_trace_with_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = [{
                "frame": "0",
                "identity_attention_invocations": 2,
                "identity_token_applications": 10,
                "adaptive_merge_invocations": 0,
                "status": "PASS",
            }]
            save_component_trace(temp_dir, "attention_only", 2025, frames)
            data = json.loads(
                (Path(temp_dir) / "component_trace.json").read_text(encoding="utf-8")
            )
            self.assertEqual(data["seed"], 2025)
            self.assertEqual(data["consistency_mode"], "attention_only")


if __name__ == "__main__":
    unittest.main()
