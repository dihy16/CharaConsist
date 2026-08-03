import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from characonsist.visualization.component_comparison import create_component_comparison


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
            self.assertEqual(audit["difference_reference"], "prompt_only")
            self.assertEqual(
                audit["conditions"]["full"]["component_trace"]["consistency_mode"],
                "full",
            )


if __name__ == "__main__":
    unittest.main()
