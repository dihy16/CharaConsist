import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from point_and_mask.visualize_gr import (
    CorrespondenceSession,
    discover_artifact_frames,
    load_action_heatmap,
    load_dense_correspondence,
    load_final_mask,
    select_correspondence,
)


def dense_metadata():
    return {
        "schema_version": 1,
        "similarity_threshold": 0.5,
        "current_grid": {"height": 2, "width": 2},
        "identity_grid": {"height": 2, "width": 2},
        "matches": [
            {"current": [0, 0], "identity": [1, 1], "similarity": 0.9, "current_foreground": True, "identity_foreground": True, "valid": True},
            {"current": [1, 0], "identity": [0, 0], "similarity": 0.4, "current_foreground": True, "identity_foreground": True, "valid": False},
            {"current": [0, 1], "identity": [0, 1], "similarity": 0.7, "current_foreground": False, "identity_foreground": False, "valid": False},
            {"current": [1, 1], "identity": [1, 0], "similarity": 0.8, "current_foreground": True, "identity_foreground": True, "valid": True},
        ],
    }


class VisualizeGrTests(unittest.TestCase):
    def test_dense_loader_and_valid_click(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "0_dense.json"
            path.write_text(json.dumps(dense_metadata()), encoding="utf-8")
            dense = load_dense_correspondence(path)
        self.assertEqual(dense["identity_x"].tolist(), [[1, 0], [0, 1]])
        session = CorrespondenceSession(
            Image.new("RGB", (20, 20), "white"), Image.new("RGB", (20, 20), "white"),
            dense["identity_mask"], dense["current_mask"], dense["identity_x"], dense["identity_y"],
            dense["similarity"], dense["valid"], dense["threshold"],
        )
        identity, current, status = select_correspondence(session, 1, 1)
        self.assertEqual(identity.size, (20, 20))
        self.assertEqual(current.size, (20, 20))
        self.assertIn("0.9000", status)
        self.assertIn("Valid foreground", status)

    def test_invalid_click_is_reported(self):
        session = CorrespondenceSession(
            Image.new("RGB", (20, 20), "white"), Image.new("RGB", (20, 20), "white"),
            np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=bool),
            np.zeros((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.int32),
            np.full((2, 2), 0.4, dtype=np.float32), np.zeros((2, 2), dtype=bool), 0.5,
        )
        _, _, status = select_correspondence(session, 15, 1)
        self.assertIn("Not a valid", status)

    def test_artifact_discovery_requires_complete_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            Image.new("RGB", (4, 4)).save(run_dir / "id.jpg")
            (run_dir / "points").mkdir()
            Image.new("RGB", (4, 4)).save(run_dir / "0.jpg")
            (run_dir / "points" / "0_dense.json").write_text(json.dumps(dense_metadata()), encoding="utf-8")
            (run_dir / "points" / "1_dense.json").write_text(json.dumps(dense_metadata()), encoding="utf-8")
            self.assertEqual(discover_artifact_frames(run_dir), [0])

    def test_action_heatmap_is_optional_for_saved_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.assertIsNone(load_action_heatmap(run_dir, 0))

            action_dir = run_dir / "action_attention"
            action_dir.mkdir()
            Image.new("RGB", (6, 4), "red").save(action_dir / "0_heatmap.png")
            heatmap = load_action_heatmap(run_dir, 0)

        self.assertEqual(heatmap.size, (6, 4))

    def test_final_mask_is_optional_and_extracts_mask_only_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.assertIsNone(load_final_mask(run_dir, 0, (6, 4)))

            mask_dir = run_dir / "mask"
            mask_dir.mkdir()
            comparison = Image.new("RGB", (12, 4), "blue")
            comparison.paste(Image.new("RGB", (6, 4), "red"), (6, 0))
            comparison.save(mask_dir / "0_mask.jpg", quality=100, subsampling=0)
            final_mask = load_final_mask(run_dir, 0, (6, 4))

        self.assertEqual(final_mask.size, (6, 4))
        red, green, blue = final_mask.getpixel((2, 2))
        self.assertGreater(red, 240)
        self.assertLess(green, 15)
        self.assertLess(blue, 15)

    def test_dense_loader_rejects_duplicate_current_coordinate(self):
        metadata = dense_metadata()
        metadata["matches"][1]["current"] = [0, 0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_dense_correspondence(path)


if __name__ == "__main__":
    unittest.main()
