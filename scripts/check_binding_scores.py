#!/usr/bin/env python3
"""Gate full-sequence runs on completed manual actor-action scores."""

import argparse
import csv
from pathlib import Path

from characonsist.experiments.conditions import action_binding_label, parse_action_binding_conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    beta, gamma = parse_action_binding_conditions([args.condition])[0]
    winner = action_binding_label(beta, gamma)
    baseline = action_binding_label(0, 0)
    rows = list(csv.DictReader(args.scores.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["combined_binding_success"] not in {"0", "1"}:
            raise SystemExit("Complete all combined_binding_success cells with 0 or 1 first.")
    rates = {}
    for label in (baseline, winner):
        values = [int(row["combined_binding_success"]) for row in rows if row["condition"] == label]
        if len(values) != 5:
            raise SystemExit(f"Expected five scored seeds for {label}.")
        rates[label] = sum(values) / len(values)
    print(f"baseline={rates[baseline]:.3f} selected={rates[winner]:.3f}")
    if rates[winner] <= rates[baseline]:
        raise SystemExit("Selected condition does not beat the matched baseline.")


if __name__ == "__main__":
    main()
