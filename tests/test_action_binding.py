import unittest
from pathlib import Path

import torch

from characonsist.prompts import build_prompt_spans_roles_and_bindings, parse_indexed_tags
from models.action_binding import (
    apply_character_action_bias,
    build_character_maps,
    validate_character_maps,
)


class FakeTokenResult:
    def __init__(self, count):
        self.attention_mask = torch.zeros((1, 512), dtype=torch.long)
        self.attention_mask[:, : min(count + 1, 512)] = 1


class FakeTokenizer:
    def __call__(self, text, **_kwargs):
        return FakeTokenResult(len(text.split()))


class FakePipe:
    tokenizer_2 = FakeTokenizer()


class IndexedPromptTests(unittest.TestCase):
    def test_full_identity_and_action_tags_strip_without_reordering(self):
        foreground = "[C2]a silver-haired woman[/C2] and [C1]a red-bearded man[/C1]"
        action = "[A1]the man drinks[/A1] while [A2]the woman watches[/A2]"
        prompt, _, action_start, _, _, spans = build_prompt_spans_roles_and_bindings(
            "Kitchen", foreground, action, FakePipe()
        )
        self.assertEqual(
            prompt,
            "Kitchen a silver-haired woman and a red-bearded man the man drinks while the woman watches",
        )
        self.assertEqual(set(spans["characters"]), {"1", "2"})
        self.assertEqual(spans["actions"]["1"], (action_start, action_start + 3))
        self.assertEqual(spans["actions"]["2"], (action_start + 4, action_start + 7))

    def test_malformed_or_wrong_section_tags_fail(self):
        for text, kind in (("[C1]man", "C"), ("[/A1]drinks", "A"), ("[A1]man[/A1]", "C")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_indexed_tags(text, kind)

    def test_tagged_minimal_prompt_cleans_to_exact_baseline_text(self):
        folder = Path(__file__).resolve().parents[1] / "prompts" / "stress_test"
        baseline = (folder / "2b_final_baseline.txt").read_text(encoding="utf-8").splitlines()
        tagged = (folder / "2b_final_action_binding.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(baseline), len(tagged))
        for baseline_line, tagged_line in zip(baseline, tagged):
            baseline_bg, baseline_fg, baseline_action = baseline_line.split("#", 2)
            tagged_bg, tagged_fg, tagged_action = tagged_line.split("#", 2)
            clean_fg, _ = parse_indexed_tags(tagged_fg, "C")
            clean_action, _ = parse_indexed_tags(tagged_action, "A")
            self.assertEqual((tagged_bg, clean_fg, clean_action), (
                baseline_bg, baseline_fg, baseline_action
            ))


class ActionBindingTensorTests(unittest.TestCase):
    def test_maps_normalize_independently_and_pass_preflight(self):
        foreground = torch.ones((1, 2, 2), dtype=torch.bool)
        maps = build_character_maps(
            {
                "1": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
                "2": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
            },
            foreground,
        )
        self.assertEqual(validate_character_maps(maps, foreground)["status"], "pass")
        self.assertEqual(maps["1"].max().item(), 1.0)
        self.assertEqual(maps["2"].max().item(), 1.0)

    def test_zero_strength_is_exact_no_op_without_allocation(self):
        query = torch.zeros((1, 2, 6, 4))
        key = torch.zeros((1, 2, 6, 4))
        mask = torch.zeros((1, 1, 6, 6))
        result = apply_character_action_bias(
            mask, 2, 4, {"1": (0, 1)}, {"1": torch.ones(1, 2, 2)},
            0.0, 0.0, query, key,
        )
        self.assertIs(result, mask)

    def test_contrastive_bias_targets_only_action_keys_and_preserves_infinity(self):
        query = torch.zeros((1, 2, 6, 4))
        key = torch.zeros((1, 2, 8, 4))
        mask = torch.zeros((1, 1, 6, 8))
        mask[..., 2, 0] = float("-inf")
        trace = {}
        result = apply_character_action_bias(
            mask,
            2,
            4,
            {"1": (0, 1), "2": (1, 2)},
            {
                "1": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                "2": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
            },
            1.0,
            0.5,
            query,
            key,
            trace,
        )
        self.assertTrue(torch.isneginf(result[..., 2, 0]).item())
        self.assertEqual(result[..., 2, 1].item(), -0.5)
        self.assertEqual(result[..., 3, 0].item(), -0.5)
        self.assertEqual(result[..., 3, 1].item(), 1.0)
        self.assertEqual(result[..., 2, 2].item(), 0.0)
        self.assertEqual(trace["invocations"], 1)


if __name__ == "__main__":
    unittest.main()
