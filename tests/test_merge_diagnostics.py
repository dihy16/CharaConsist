import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge_diagnostics import save_frame_merge_diagnostics, save_prompt_gate_audit


def diagnostic_state(gate_strength):
    base = np.array([[0.8, 0.0], [0.4, 0.0]], dtype=np.float32)
    suppression = base * gate_strength * np.array(
        [[0.5, 0.0], [1.0, 0.0]], dtype=np.float32
    )
    effective = base - suppression
    return {
        "invocations": {1: 19},
        "records": {
            1: {
                "alpha": 0.8,
                "maps": {
                    "similarities": base / 0.8,
                    "action_scores": np.array([[0.5, 0.0], [1.0, 0.0]], dtype=np.float32),
                    "gate_factors": np.array([[1 - 0.5 * gate_strength, 0.0], [1 - gate_strength, 0.0]], dtype=np.float32),
                    "base_weights": base,
                    "effective_weights": effective,
                    "suppressed_weights": suppression,
                    "valid_mask": np.array([[True, False], [True, False]]),
                },
            }
        },
    }


class MergeDiagnosticTests(unittest.TestCase):
    def test_saves_exact_arrays_trace_and_passing_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audit = save_frame_merge_diagnostics(diagnostic_state(1.0), root, 0, 1.0)
            save_prompt_gate_audit(root, [audit], 1.0, 2025)

            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["attention_layer_invocations"], 19)
            with np.load(root / "merge_diagnostics" / "0_weights.npz") as data:
                self.assertEqual(data["effective_weights"].shape, (1, 2, 2))
                self.assertAlmostEqual(float(data["suppressed_weights"].max()), 0.4)
            lines = (root / "action_gate_trace.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["modified_tokens"], 2)
            prompt_audit = json.loads((root / "action_gate_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(prompt_audit["status"], "PASS")

    def test_lambda_zero_is_control(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = save_frame_merge_diagnostics(
                diagnostic_state(0.0), Path(temp_dir), 0, 0.0
            )
            self.assertEqual(audit["status"], "CONTROL")
            self.assertEqual(audit["modified_token_applications"], 0)

    def test_no_records_warns_for_positive_lambda(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = save_frame_merge_diagnostics(
                {"records": {}, "invocations": {}}, Path(temp_dir), 0, 1.0
            )
            self.assertEqual(audit["status"], "WARN")


if __name__ == "__main__":
    unittest.main()
