#!/usr/bin/env python3
"""Render the matched character-conditioned action-binding experiment."""

import argparse

from characonsist.visualization.action_binding_comparison import (
    render_action_binding_comparison,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results_colab")
    parser.add_argument("--output-dir", default="comparisons/action_binding/2b_final_action_binding")
    parser.add_argument("--prompt", default="2b_final_action_binding")
    args = parser.parse_args()
    summary = render_action_binding_comparison(args.results_root, args.output_dir, args.prompt)
    print(summary["comparison"])
    print(summary["manual_scores"])


if __name__ == "__main__":
    main()
