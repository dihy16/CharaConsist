import unittest
from pathlib import Path

import torch

from characonsist.prompts import build_prompt_spans_and_roles, parse_role_tags
from models.role_action_routing import apply_role_attention_bias, build_soft_role_maps


class FakeTokenResult:
    def __init__(self, count):
        self.attention_mask = torch.zeros((1, 512), dtype=torch.long)
        self.attention_mask[:, : min(count + 1, 512)] = 1


class FakeTokenizer:
    def __call__(self, text, **_kwargs):
        return FakeTokenResult(len(text.split()))


class FakePipe:
    tokenizer_2 = FakeTokenizer()


class RoleTagTests(unittest.TestCase):
    def test_tags_are_stripped_and_spans_are_cumulative(self):
        action = "[S]The man[/S] [A]hands[/A] [O]the mug[/O] to [R]the woman[/R]"
        clean, char_spans = parse_role_tags(action)
        self.assertEqual(clean, "The man hands the mug to the woman")
        self.assertEqual(clean[slice(*char_spans["predicate"])], "hands")

        prompt, _, action_start, real_end, spans = build_prompt_spans_and_roles(
            "Kitchen", "A man and a woman", action, FakePipe()
        )
        self.assertNotIn("[S]", prompt)
        self.assertEqual(spans["subject"], (action_start, action_start + 2))
        self.assertEqual(spans["predicate"], (action_start + 2, action_start + 3))
        self.assertEqual(spans["object"], (action_start + 3, action_start + 5))
        self.assertEqual(spans["recipient"], (action_start + 6, real_end))

    def test_untagged_action_is_backward_compatible(self):
        clean, spans = parse_role_tags("The man holds a mug")
        self.assertEqual(clean, "The man holds a mug")
        self.assertEqual(spans, {})

    def test_malformed_tags_fail(self):
        for action in ("[S]man", "[/S]man", "[S][A]man[/A][/S]"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                parse_role_tags(action)

    def test_unknown_brackets_remain_literal_text(self):
        clean, spans = parse_role_tags("A person beside [EXIT]")
        self.assertEqual(clean, "A person beside [EXIT]")
        self.assertEqual(spans, {})

    def test_annotated_transfer_prompt_cleans_to_original_actions(self):
        root = Path(__file__).resolve().parents[1] / "prompts" / "stress_test"
        original = (root / "2b_transfer.txt").read_text(encoding="utf-8").splitlines()
        annotated = (root / "2b_transfer_roles.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(original), len(annotated))
        for original_line, annotated_line in zip(original, annotated):
            original_parts = original_line.split("#", 2)
            annotated_parts = annotated_line.split("#", 2)
            clean_action, _ = parse_role_tags(annotated_parts[2])
            self.assertEqual(annotated_parts[:2], original_parts[:2])
            self.assertEqual(clean_action, original_parts[2])


class RoleMapTests(unittest.TestCase):
    def test_soft_maps_are_foreground_localized(self):
        foreground = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.bool)
        raw = {
            "subject": torch.tensor([[[1.0, 0.2], [8.0, 8.0]]]),
            "predicate": torch.tensor([[[0.3, 1.0], [8.0, 8.0]]]),
            "object": torch.tensor([[[0.1, 1.0], [8.0, 8.0]]]),
            "recipient": torch.tensor([[[0.2, 1.0], [8.0, 8.0]]]),
        }
        maps = build_soft_role_maps(raw, foreground)
        self.assertGreater(maps["subject"][0, 0, 0], maps["recipient"][0, 0, 0])
        self.assertGreater(maps["recipient"][0, 0, 1], maps["subject"][0, 0, 1])
        self.assertTrue(torch.equal(maps["interaction"][:, 1], torch.zeros((1, 2))))

    def test_zero_strength_is_exact_object_no_op(self):
        mask = torch.zeros((1, 1, 6, 6))
        result = apply_role_attention_bias(
            mask, 2, 4, {"subject": (0, 1)}, {"subject": torch.ones(1, 2, 2)}, 0.0
        )
        self.assertIs(result, mask)

    def test_bias_is_role_local_and_preserves_negative_infinity(self):
        mask = torch.zeros((1, 1, 6, 7))
        mask[..., 2, 0] = float("-inf")
        trace = {}
        result = apply_role_attention_bias(
            mask,
            text_seq_len=2,
            visual_seq_len=4,
            role_spans={"subject": (0, 1), "predicate": (1, 2)},
            role_maps={
                "subject": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                "interaction": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
            },
            strength=1.0,
            trace_state=trace,
        )
        self.assertTrue(torch.isneginf(result[..., 2, 0]).item())
        self.assertEqual(result[..., 3, 1].item(), 1.0)
        self.assertEqual(result[..., 3, 0].item(), 0.0)
        self.assertEqual(trace["invocations"], 1)


if __name__ == "__main__":
    unittest.main()
