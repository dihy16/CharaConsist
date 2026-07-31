import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.action_gating import (  # noqa: E402
    action_gated_merge_weights,
    normalize_action_attention,
)
from prompt_utils import build_prompt_and_spans  # noqa: E402


class FakeTokenizer:
    """Whitespace tokenizer with one terminal token for span tests."""

    def __call__(self, text, max_length, **kwargs):
        token_count = min(len(text.split()) + 1, max_length)
        attention_mask = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask[:, :token_count] = 1
        return SimpleNamespace(attention_mask=attention_mask)


class PromptSpanTests(unittest.TestCase):
    def setUp(self):
        self.pipe = SimpleNamespace(tokenizer_2=FakeTokenizer())

    def test_action_span_uses_cumulative_prefixes(self):
        prompt, background_end, action_start, real_end = build_prompt_and_spans(
            "in a library",
            "a woman",
            "reading a book",
            self.pipe,
        )

        self.assertEqual(prompt, "in a library a woman reading a book")
        self.assertEqual(background_end, 3)
        self.assertEqual(action_start, 5)
        self.assertEqual(real_end, 8)

    def test_truncated_action_span_is_empty(self):
        long_foreground = " ".join(["person"] * 600)
        _, _, action_start, real_end = build_prompt_and_spans(
            "room",
            long_foreground,
            "running",
            self.pipe,
        )

        self.assertEqual(action_start, real_end)
        self.assertEqual(real_end, 511)


class ActionAttentionTests(unittest.TestCase):
    def test_normalizes_only_foreground_positions(self):
        attention = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
        foreground = torch.tensor([[[False, True], [True, True]]])

        scores = normalize_action_attention(attention, foreground)

        expected = torch.tensor([[[0.0, 0.0], [1.0 / 3.0, 1.0]]])
        torch.testing.assert_close(scores, expected)

    def test_constant_or_empty_foreground_is_neutral(self):
        attention = torch.ones((2, 2, 2))
        foreground = torch.tensor(
            [
                [[True, True], [False, False]],
                [[False, False], [False, False]],
            ]
        )

        scores = normalize_action_attention(attention, foreground)

        torch.testing.assert_close(scores, torch.zeros_like(attention))

    def test_gate_strength_zero_matches_baseline(self):
        similarities = torch.tensor([0.25, 0.75])
        action_scores = torch.tensor([0.0, 1.0])

        weights = action_gated_merge_weights(
            0.8,
            similarities,
            action_scores,
            gate_strength=0.0,
        )

        torch.testing.assert_close(weights, similarities * 0.8)

    def test_full_action_score_suppresses_merge(self):
        similarities = torch.ones(3)
        action_scores = torch.tensor([0.0, 0.5, 1.0])

        weights = action_gated_merge_weights(
            0.8,
            similarities,
            action_scores,
            gate_strength=1.0,
        )

        torch.testing.assert_close(weights, torch.tensor([0.8, 0.4, 0.0]))

    def test_missing_action_scores_matches_baseline(self):
        similarities = torch.tensor([0.4, 0.8])

        weights = action_gated_merge_weights(
            0.5,
            similarities,
            action_scores=None,
        )

        torch.testing.assert_close(weights, similarities * 0.5)

    def test_invalid_strength_is_rejected(self):
        with self.assertRaises(ValueError):
            action_gated_merge_weights(
                0.5,
                torch.ones(1),
                torch.zeros(1),
                gate_strength=1.1,
            )


if __name__ == "__main__":
    unittest.main()
