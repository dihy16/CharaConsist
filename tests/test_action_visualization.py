import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from action_visualization import action_score_array, save_action_attention_artifacts


class ActionVisualizationTests(unittest.TestCase):
    def test_saves_raw_heatmap_and_overlay(self):
        image = Image.new("RGB", (32, 16), color=(100, 100, 100))
        scores = np.array([[[0.0, 0.5], [0.75, 1.0]]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_action_attention_artifacts(image, scores, output_dir, "0")

            stored = np.load(output_dir / "0_scores.npy")
            np.testing.assert_allclose(stored, scores[0])
            with Image.open(output_dir / "0_heatmap.png") as heatmap:
                self.assertEqual(heatmap.size, image.size)
            with Image.open(output_dir / "0_overlay.jpg") as overlay:
                self.assertEqual(overlay.size, image.size)

    def test_rejects_invalid_shape_and_values(self):
        with self.assertRaises(ValueError):
            action_score_array(np.zeros((2, 2, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            action_score_array(np.array([[0.0, np.nan]], dtype=np.float32))
        with self.assertRaises(ValueError):
            action_score_array(np.array([[0.0, 1.1]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
