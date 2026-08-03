import unittest
from unittest.mock import patch

from characonsist.runners import colab_worker as worker
from characonsist.experiments.conditions import build_component_conditions


class ColabWorkerComponentTests(unittest.TestCase):
    def tearDown(self):
        worker.STATE.clear()

    def test_dispatches_only_configured_seed_matched_component(self):
        worker.STATE.update(
            experiment_kind="component_ablation",
            conditions=build_component_conditions(
                "prompt_only,attention_only,full", "2025"
            ),
        )
        with patch.object(worker, "_run_condition", return_value={"status": "success"}) as run:
            result = worker.run_component("1a_anchor_verb.txt", "attention_only", 2025)

        self.assertEqual(result["status"], "success")
        self.assertEqual(run.call_args.args[1].consistency_mode, "attention_only")
        self.assertEqual(run.call_args.args[1].seed, 2025)
        with self.assertRaises(ValueError):
            worker.run_component("1a_anchor_verb.txt", "full", 2026)
        with self.assertRaises(RuntimeError):
            worker.run_one("1a_anchor_verb.txt", 0, 2025)


if __name__ == "__main__":
    unittest.main()
