import unittest
from pathlib import Path

from sweep_utils import (
    build_sweep_conditions,
    lambda_label,
    parse_action_gate_strengths,
    parse_seeds,
)


class SweepUtilsTests(unittest.TestCase):
    def test_parses_unique_values_in_stable_order(self):
        self.assertEqual(parse_action_gate_strengths("0,0.5,0.5,1"), [0.0, 0.5, 1.0])
        self.assertEqual(parse_seeds("2025,7,2025"), [2025, 7])

    def test_rejects_invalid_values(self):
        for value in ("", "0,,1", "nan", "1.1", "-0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_action_gate_strengths(value)
        for value in ("", "1,,2", "1.5", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_seeds(value)

    def test_builds_lambda_major_condition_paths(self):
        conditions = build_sweep_conditions("0,0.25", "2025,2026")

        self.assertEqual(
            [(item.action_gate_strength, item.seed) for item in conditions],
            [(0.0, 2025), (0.0, 2026), (0.25, 2025), (0.25, 2026)],
        )
        self.assertEqual(lambda_label(0.5), "lambda_0p50")
        self.assertEqual(
            conditions[-1].output_prefix,
            Path("lambda_0p25") / "seed_2026" / "bg_fg",
        )


if __name__ == "__main__":
    unittest.main()

