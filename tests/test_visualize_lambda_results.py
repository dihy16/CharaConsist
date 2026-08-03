import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
from PIL import Image

from characonsist.visualization.lambda_comparison import (
    create_diagnostic_comparisons,
    create_lambda_comparison,
    find_lambda_directories,
)


PROMPT_LINES = [
    "A room,#a woman in blue,#the woman standing",
    "A room,#a woman in blue,#the woman sitting",
]


class LambdaComparisonTests(unittest.TestCase):
    def test_discovers_lambda_directories_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "lambda_1p00").mkdir()
            (root / "lambda_0p50").mkdir()
            (root / "unrelated").mkdir()

            discovered = find_lambda_directories(root)

            self.assertEqual([value for value, _ in discovered], [0.5, 1.0])

    def test_renders_available_and_missing_lambda_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt_file = root / "example.txt"
            prompt_file.write_text("\n".join(PROMPT_LINES) + "\n", encoding="utf-8")

            result = root / "lambda_0p50" / "seed_2025" / "bg_fg" / "example" / "prompt_0"
            result.mkdir(parents=True)
            Image.new("RGB", (40, 30), "red").save(result / "id.jpg")
            Image.new("RGB", (40, 30), "blue").save(result / "0.jpg")

            output = root / "comparison.jpg"
            create_lambda_comparison(
                root,
                prompt_file,
                output,
                strengths="0,0.5,1",
                panel_size=64,
            )

            self.assertTrue(output.is_file())
            with Image.open(output) as comparison:
                self.assertEqual(comparison.mode, "RGB")
                self.assertEqual(comparison.width, 16 * 2 + 150 + 2 * 64)

    def test_renders_separate_diagnostics_and_comparison_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt_file = root / "example.txt"
            prompt_file.write_text("\n".join(PROMPT_LINES) + "\n", encoding="utf-8")
            result = root / "lambda_1p00" / "seed_2025" / "bg_fg" / "example" / "prompt_0"
            (result / "mask").mkdir(parents=True)
            (result / "action_attention").mkdir()
            (result / "points").mkdir()
            (result / "merge_diagnostics").mkdir()
            Image.new("RGB", (40, 30), "red").save(result / "id.jpg")
            Image.new("RGB", (40, 30), "blue").save(result / "0.jpg")
            Image.new("RGB", (80, 30), "red").save(result / "mask" / "id_mask.jpg")
            Image.new("RGB", (80, 30), "red").save(result / "mask" / "0_mask.jpg")
            Image.new("RGB", (40, 30), "yellow").save(result / "action_attention" / "id_overlay.jpg")
            Image.new("RGB", (40, 30), "yellow").save(result / "action_attention" / "0_overlay.jpg")
            Image.new("RGB", (80, 30), "green").save(result / "points" / "0_dense.jpg")
            (result / "points" / "0_dense.json").write_text(
                json.dumps({"matches": [{"valid": True}, {"valid": False}]}),
                encoding="utf-8",
            )
            shape = (1, 2, 2)
            np.savez_compressed(
                result / "merge_diagnostics" / "0_weights.npz",
                schema_version=np.asarray(1),
                gate_strength=np.asarray(1.0),
                step_indices=np.asarray([1]),
                alphas=np.asarray([0.8]),
                valid_mask=np.ones(shape, dtype=bool),
                base_weights=np.full(shape, 0.8, dtype=np.float32),
                effective_weights=np.full(shape, 0.4, dtype=np.float32),
                suppressed_weights=np.full(shape, 0.4, dtype=np.float32),
                similarities=np.ones(shape, dtype=np.float32),
                action_scores=np.full(shape, 0.5, dtype=np.float32),
                gate_factors=np.full(shape, 0.5, dtype=np.float32),
            )
            (result / "action_gate_audit.json").write_text(
                json.dumps({
                    "status": "PASS",
                    "frames": [{
                        "frame": "0",
                        "status": "PASS",
                        "steps_recorded": 1,
                        "modified_token_applications": 4,
                        "suppression_max": 0.4,
                    }],
                }),
                encoding="utf-8",
            )

            output = root / "comparison.jpg"
            destinations, audit = create_diagnostic_comparisons(
                root,
                prompt_file,
                output,
                strengths="0,1",
                panel_size=64,
            )

            self.assertEqual(len(destinations), 10)
            self.assertTrue(all(path.is_file() for path in destinations))
            comparison_audit = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(comparison_audit["difference_reference"], 1.0)
            self.assertEqual(
                comparison_audit["conditions"]["lambda_1p00"]["frames"][0]["status"],
                "UNCHANGED",
            )


if __name__ == "__main__":
    unittest.main()
