import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from characonsist.visualization.role_action_comparison import render_role_action_comparison


class RoleActionComparisonTests(unittest.TestCase):
    def test_renders_matched_stories_and_audits_outputs_and_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for label, color in (("role_bias_0p00", 0), ("role_bias_1p00", 20)):
                folder = (
                    root / "role_action_ablation" / label / "seed_2025" /
                    "bg_fg" / "2b_transfer_roles" / "prompt_0"
                )
                map_dir = folder / "role_action" / "frame_0"
                map_dir.mkdir(parents=True)
                Image.new("RGB", (80, 30), (color, color, color)).save(folder / "story.jpg")
                Image.new("RGB", (16, 16), (color, color, color)).save(folder / "id.jpg")
                Image.new("RGB", (16, 16), (color, color, color)).save(folder / "0.jpg")
                np.save(map_dir / "subject.npy", np.ones((2, 2), dtype=np.float32))

            output = render_role_action_comparison(root, "2b_transfer_roles", 2025)
            self.assertTrue(output.is_file())
            audit = json.loads(output.with_name("role_action_audit.json").read_text())
            self.assertEqual(audit["pre_run_role_maps"][0]["mae"], 0.0)
            self.assertGreater(audit["outputs"][0]["pixel_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
