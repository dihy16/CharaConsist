import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from characonsist.visualization.component_comparison import (
    DEFAULT_COMPONENT_VIEWS,
    create_component_comparison,
    create_component_diagnostic_comparisons,
)


class ComponentComparisonTests(unittest.TestCase):
    def test_renders_seed_matched_modes_and_embeds_traces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = root / "example.txt"
            prompt.write_text(
                "A room,#a woman,#the woman standing\n"
                "A room,#a woman,#the woman throwing a book\n",
                encoding="utf-8",
            )
            for index, mode in enumerate(("prompt_only", "attention_only", "full")):
                result = (
                    root / "component_ablation" / mode / "seed_2025"
                    / "bg_fg" / "example" / "prompt_0"
                )
                result.mkdir(parents=True)
                Image.new("RGB", (32, 32), (index * 30, 0, 0)).save(result / "id.jpg")
                Image.new("RGB", (32, 32), (0, index * 30, 0)).save(result / "0.jpg")
                (result / "component_trace.json").write_text(
                    json.dumps({"consistency_mode": mode, "seed": 2025, "status": "PASS"}),
                    encoding="utf-8",
                )

            output = root / "comparison" / "outputs.jpg"
            image_path, audit_path = create_component_comparison(
                root, prompt, output, seed=2025, panel_size=64
            )

            self.assertTrue(image_path.is_file())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["seed"], 2025)
            self.assertEqual(audit["comparison_strategy"], "incremental_predecessor")
            self.assertEqual(audit["comparison_pairs"]["full"], "attention_only")
            self.assertEqual(
                audit["conditions"]["full"]["component_trace"]["consistency_mode"],
                "full",
            )

    def test_renders_component_diagnostics_and_expected_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = root / "example.txt"
            prompt.write_text(
                "A room,#a woman,#the woman standing\n"
                "A room,#a woman,#the woman throwing a book\n",
                encoding="utf-8",
            )
            modes = ("prompt_only", "attention_only", "full")
            for index, mode in enumerate(modes):
                result = (
                    root / "component_ablation" / mode / "seed_2025"
                    / "bg_fg" / "example" / "prompt_0"
                )
                result.mkdir(parents=True)
                Image.new("RGB", (32, 32), (index * 30, 0, 0)).save(result / "id.jpg")
                Image.new("RGB", (32, 32), (0, index * 30, 0)).save(result / "0.jpg")
                for stem in ("id", "0"):
                    (result / "mask").mkdir(exist_ok=True)
                    Image.new("RGB", (64, 32), "white").save(
                        result / "mask" / f"{stem}_mask.jpg"
                    )
                    (result / "action_attention").mkdir(exist_ok=True)
                    Image.new("RGB", (32, 32), "blue").save(
                        result / "action_attention" / f"{stem}_overlay.jpg"
                    )
                if mode != "prompt_only":
                    (result / "points").mkdir(exist_ok=True)
                    Image.new("RGB", (64, 32), "green").save(
                        result / "points" / "0_dense.jpg"
                    )
                    (result / "points" / "0_dense.json").write_text(
                        json.dumps({"matches": [{"valid": True}, {"valid": False}]}),
                        encoding="utf-8",
                    )
                if mode == "full":
                    (result / "merge_diagnostics").mkdir(exist_ok=True)
                    np.savez_compressed(
                        result / "merge_diagnostics" / "0_weights.npz",
                        step_indices=np.asarray([1, 2]),
                        effective_weights=np.ones((2, 2, 2), dtype=np.float32) * 0.4,
                    )
                enabled = mode != "prompt_only"
                merge_enabled = mode == "full"
                (result / "component_trace.json").write_text(
                    json.dumps({
                        "consistency_mode": mode,
                        "seed": 2025,
                        "status": "PASS",
                        "frames": [{
                            "frame": "0",
                            "identity_attention_invocations": 10 if enabled else 0,
                            "identity_token_applications": 20 if enabled else 0,
                            "adaptive_merge_invocations": 5 if merge_enabled else 0,
                            "status": "PASS",
                        }],
                        "identity_attention_invocations": 10 if enabled else 0,
                        "identity_token_applications": 20 if enabled else 0,
                        "adaptive_merge_invocations": 5 if merge_enabled else 0,
                    }),
                    encoding="utf-8",
                )

            output = root / "comparison" / "outputs.jpg"
            destinations, audit_path = create_component_diagnostic_comparisons(
                root, prompt, output, seed=2025, panel_size=64
            )

            self.assertEqual(
                {path.name for path in destinations},
                {f"{view}.jpg" for view in DEFAULT_COMPONENT_VIEWS},
            )
            self.assertTrue(all(path.is_file() for path in destinations))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["conditions"]["prompt_only"]["artifact_counts"]["points"], 0)
            self.assertEqual(audit["conditions"]["attention_only"]["artifact_counts"]["points"], 1)
            self.assertEqual(audit["conditions"]["full"]["artifact_counts"]["merge_diagnostics"], 1)
            self.assertEqual(audit["conditions"]["full"]["reference_mode"], "attention_only")
            self.assertEqual(audit["conditions"]["full"]["frames"][0]["status"], "CHANGED")


if __name__ == "__main__":
    unittest.main()
