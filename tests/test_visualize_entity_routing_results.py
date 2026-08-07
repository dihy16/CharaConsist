import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from characonsist.experiments.conditions import action_binding_label
from characonsist.visualization.entity_routing_comparison import (
    CONDITIONS,
    render_entity_routing_comparison,
)
from characonsist.visualization.entity_routing_viewer import (
    discover_routing_seeds,
    identity_image_path,
    result_image_path,
)


class EntityRoutingComparisonTests(unittest.TestCase):
    def test_viewer_discovers_seed_union_and_uses_matched_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir) / "results"
            beta, gamma = 1.0, 0.5
            off_path = result_image_path(
                results, "off", beta, gamma, 2025, "test_prompt"
            )
            hard_path = result_image_path(
                results, "hard", beta, gamma, 2026, "test_prompt"
            )
            off_path.parent.mkdir(parents=True)
            hard_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(off_path)
            Image.new("RGB", (8, 8), "red").save(hard_path)

            self.assertEqual(
                discover_routing_seeds(results, beta, gamma, "test_prompt"),
                [2025, 2026],
            )
            self.assertIn("routing_off", str(off_path))
            self.assertIn("beta_1p00_gamma_0p50", str(off_path))
            self.assertEqual(identity_image_path(
                results, "off", beta, gamma, 2025, "test_prompt"
            ).name, "id.jpg")

    def test_renders_matrix_and_preserves_manual_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results = root / "results"
            output = root / "comparison"
            for mode, beta, gamma in CONDITIONS:
                run = (
                    results / "entity_routing_ablation" / f"routing_{mode}"
                    / action_binding_label(beta, gamma) / "seed_2025" / "bg_fg"
                    / "2b_final_action_binding" / "prompt_0"
                )
                run.mkdir(parents=True)
                Image.new("RGB", (32, 32), "blue").save(run / "0.jpg")

            summary = render_entity_routing_comparison(results, output)
            self.assertTrue(Path(summary["comparison"]).is_file())
            with (output / "manual_scores.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            rows[0]["combined_binding_success"] = "1"
            with (output / "manual_scores.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            render_entity_routing_comparison(results, output)
            with (output / "manual_scores.csv").open(encoding="utf-8") as handle:
                preserved = next(csv.DictReader(handle))
            self.assertEqual(preserved["combined_binding_success"], "1")


if __name__ == "__main__":
    unittest.main()
