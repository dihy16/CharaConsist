import unittest
from unittest.mock import patch

from characonsist.runners import colab_worker as worker
from characonsist.experiments.conditions import (
    build_component_conditions,
    build_entity_routing_conditions,
    build_role_action_conditions,
)


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

    def test_dispatches_only_configured_role_action_condition(self):
        worker.STATE.update(
            experiment_kind="role_action_ablation",
            conditions=build_role_action_conditions("0,1", "2025"),
        )
        with patch.object(worker, "_run_condition", return_value={"status": "success"}) as run:
            result = worker.run_role_action("2b_transfer_roles.txt", 1, 2025)

        self.assertEqual(result["status"], "success")
        self.assertEqual(run.call_args.args[1].role_action_bias_strength, 1.0)
        self.assertEqual(run.call_args.args[1].seed, 2025)
        with self.assertRaises(ValueError):
            worker.run_role_action("2b_transfer_roles.txt", 0.5, 2025)

    def test_dispatches_only_configured_entity_routing_condition(self):
        worker.STATE.update(
            experiment_kind="entity_routing_ablation",
            conditions=build_entity_routing_conditions(
                "off,hard", "0:0,1:0.5", "2025"
            ),
        )
        with patch.object(worker, "_run_condition", return_value={"status": "success"}) as run:
            result = worker.run_entity_routing(
                "2b_final_action_binding.txt", "hard", 1, 0.5, 2025
            )

        self.assertEqual(result["status"], "success")
        condition = run.call_args.args[1]
        self.assertEqual(condition.entity_routing_mode, "hard")
        self.assertEqual((condition.beta, condition.gamma), (1.0, 0.5))
        with self.assertRaises(ValueError):
            worker.run_entity_routing(
                "2b_final_action_binding.txt", "hard", 2, 1, 2025
            )


if __name__ == "__main__":
    unittest.main()
