import unittest

import torch

from characonsist.diagnostics.entity_routing import (
    summarize_entity_attention_mass,
    validate_entity_routing_trace,
)
from models.entity_routing import (
    apply_entity_identity_mask,
    build_entity_labels,
    get_entity_restricted_matches,
    measure_entity_attention_mass,
    should_apply_entity_matching,
    should_build_entity_labels,
    validate_entity_labels,
)


class EntityRoutingTests(unittest.TestCase):
    def test_summarizes_attention_mass_across_layer_timestep_records(self):
        summary = summarize_entity_attention_mass([{
            "attention_mass_records": [
                {
                    "wrong_entity_mass_sum": 1.0,
                    "same_entity_mass_sum": 3.0,
                    "query_head_count": 4,
                    "wrong_entity_mass_mean": 0.25,
                },
                {
                    "wrong_entity_mass_sum": 2.0,
                    "same_entity_mass_sum": 2.0,
                    "query_head_count": 4,
                    "wrong_entity_mass_mean": 0.5,
                },
            ]
        }])
        self.assertTrue(summary["available"])
        self.assertEqual(summary["records"], 2)
        self.assertAlmostEqual(summary["wrong_entity_mass_mean"], 3.0 / 8.0)
        self.assertAlmostEqual(summary["wrong_fraction_of_identity_mass"], 3.0 / 8.0)
        self.assertAlmostEqual(summary["max_wrong_entity_mass_mean"], 0.5)

    def test_measures_pre_routing_wrong_entity_attention_mass(self):
        query = torch.zeros((1, 1, 2, 1))
        key = torch.zeros((1, 1, 4, 1))
        attention_mask = torch.zeros((1, 1, 2, 4))
        wrong = torch.tensor([[[[False, False, False, True],
                                [False, False, True, False]]]])
        same = torch.tensor([[[[False, False, True, False],
                               [False, False, False, True]]]])
        attention_mask = attention_mask.masked_fill(wrong, float("-inf"))

        report = measure_entity_attention_mass(
            query, key, attention_mask, wrong, same, query_chunk_size=1
        )

        self.assertAlmostEqual(report["wrong_entity_mass_mean"], 0.25)
        self.assertAlmostEqual(report["same_entity_mass_mean"], 0.25)
        self.assertAlmostEqual(report["wrong_fraction_of_identity_mass"], 0.5)

    def test_trace_validation_uses_topology_derived_invocation_count(self):
        reports = {
            "status": "pass",
            "entity_token_counts": {"1": 2, "2": 2},
            "labeled_background_tokens": 0,
        }
        frame = {
            "current_map_report": reports,
            "pre_run_match_report": {"status": "pass"},
            "match_reports": [{"status": "pass"}],
            "invocations": 760,
            "wrong_entity_allowed_pairs": 0,
        }
        self.assertEqual(
            validate_entity_routing_trace(reports, [frame], 760)["status"], "pass"
        )
        self.assertEqual(
            validate_entity_routing_trace(reports, [frame], 1520)["status"], "fail"
        )

    def test_pipeline_builds_labels_only_at_correspondence_step(self):
        self.assertFalse(should_build_entity_labels("hard", 0, 10))
        self.assertTrue(should_build_entity_labels("hard", 10, 10))
        self.assertFalse(should_build_entity_labels("off", 10, 10))

    def test_pre_run_waits_for_labels_before_restricted_matching(self):
        self.assertFalse(should_apply_entity_matching("hard", True, 0, 10))
        self.assertTrue(should_apply_entity_matching("hard", True, 10, 10))
        self.assertTrue(should_apply_entity_matching("hard", False, 0, 10))
        self.assertFalse(should_apply_entity_matching("off", False, 10, 10))

    def test_builds_categorical_labels_only_inside_foreground(self):
        foreground = torch.tensor([[[True, True], [True, False]]])
        labels = build_entity_labels(
            {
                "1": torch.tensor([[[1.0, 0.8], [0.1, 0.0]]]),
                "2": torch.tensor([[[0.0, 0.2], [1.0, 0.0]]]),
            },
            foreground,
        )
        self.assertTrue(torch.equal(labels, torch.tensor([[[1, 1], [2, 0]]])))
        self.assertEqual(validate_entity_labels(labels, foreground)["status"], "pass")

    def test_restricted_matches_never_cross_entity(self):
        similarities = torch.tensor([[[0.9, 0.99], [0.98, 0.8]]])
        id_labels = torch.tensor([[1, 2]])
        curr_labels = torch.tensor([[1, 2]])
        id_indices, curr_indices, max_sim, argmax, report = get_entity_restricted_matches(
            similarities, id_labels, curr_labels, 0.5
        )
        self.assertTrue(torch.equal(id_indices, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(curr_indices, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(argmax, torch.tensor([[0, 1]])))
        self.assertTrue(torch.allclose(max_sim, torch.tensor([[0.9, 0.8]])))
        self.assertEqual(report["total_valid_matches"], 2)

    def test_restricted_matching_rejects_entity_without_valid_match(self):
        with self.assertRaisesRegex(ValueError, "C2 has no matches"):
            get_entity_restricted_matches(
                torch.tensor([[[0.9, 0.9], [0.9, 0.1]]]),
                torch.tensor([[1, 2]]),
                torch.tensor([[1, 2]]),
                0.5,
            )

    def test_hard_mask_blocks_wrong_bank_and_traces_zero_leakage(self):
        base = torch.zeros((1, 1, 3, 2), dtype=torch.bool)
        trace = {}
        routed = apply_entity_identity_mask(
            base,
            appended_entity_labels=torch.tensor([1, 2]),
            curr_entity_labels=torch.tensor([1, 2, 0]),
            trace=trace,
        )
        self.assertTrue(torch.equal(
            routed[0, 0],
            torch.tensor([[False, True], [True, False], [True, True]]),
        ))
        self.assertEqual(trace["wrong_entity_allowed_pairs"], 0)
        self.assertEqual(trace["allowed_same_entity_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
