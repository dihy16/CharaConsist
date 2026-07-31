import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from point_visualization import build_dense_correspondence, save_dense_correspondence


class PointVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "argmax_indices": np.array([[0, 2, 3, 1]]),
            "max_sim": np.array([[0.9, 0.8, 0.9, 0.9]], dtype=np.float32),
            "id_fg_mask": np.array([[[True, False], [True, True]]]),
            "curr_fg_mask": np.array([[[True, True], [False, True]]]),
        }

    def test_dense_records_map_current_tokens_to_identity_tokens(self):
        color_map, metadata = build_dense_correspondence(**self.snapshot)
        self.assertEqual(color_map.shape, (2, 2, 3))
        self.assertEqual(metadata["sampling_step_index"], 10)
        self.assertEqual(metadata["sampling_timestep_ordinal"], 11)
        self.assertEqual(len(metadata["matches"]), 4)
        self.assertEqual(metadata["matches"][1]["current"], [1, 0])
        self.assertEqual(metadata["matches"][1]["identity"], [0, 1])
        self.assertEqual([item["valid"] for item in metadata["matches"]], [True, True, False, False])

    def test_saved_comparison_uses_final_image_and_writes_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "0_dense.jpg"
            json_path = Path(temp_dir) / "0_dense.json"
            image = Image.new("RGB", (32, 24), "white")
            save_dense_correspondence(image, self.snapshot, image_path, json_path)
            with Image.open(image_path) as rendered:
                self.assertEqual(rendered.size, (64, 24))
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["extraction_pass"], "unmodified_pre_run")


if __name__ == "__main__":
    unittest.main()
