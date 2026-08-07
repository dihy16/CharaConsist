#!/usr/bin/env python3
"""Gate the full-sequence routing run on the predeclared five-seed criteria."""

import argparse
import csv
from pathlib import Path


def _as_bit(row, field):
    value = row[field].strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{field} must be filled with 0 or 1 for every row.")
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        default="comparisons/entity_routing/2b_final_action_binding/manual_scores.csv",
    )
    args = parser.parse_args()
    with Path(args.scores).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def select(mode, beta, gamma):
        selected = [
            row for row in rows
            if row["routing_mode"] == mode
            and float(row["beta"]) == beta
            and float(row["gamma"]) == gamma
        ]
        if len(selected) != 5:
            raise ValueError(f"Expected five rows for {mode}, {beta}:{gamma}.")
        return selected

    baseline = select("off", 0.0, 0.0)
    action_only = select("off", 1.0, 0.5)
    combined = select("hard", 1.0, 0.5)
    baseline_success = sum(_as_bit(row, "combined_binding_success") for row in baseline)
    action_success = sum(_as_bit(row, "combined_binding_success") for row in action_only)
    combined_success = sum(_as_bit(row, "combined_binding_success") for row in combined)
    baseline_identity = sum(_as_bit(row, "identities_correct") for row in baseline)
    combined_identity = sum(_as_bit(row, "identities_correct") for row in combined)

    passed = (
        combined_success > baseline_success
        and combined_success > action_success
        and combined_identity >= baseline_identity
    )
    print(
        f"baseline={baseline_success}/5 action_only={action_success}/5 "
        f"hard_plus_action={combined_success}/5 identity="
        f"{combined_identity}/5_vs_{baseline_identity}/5"
    )
    if not passed:
        raise SystemExit(
            "Entity routing did not pass the semantic lift gate; do not run the full sequence."
        )


if __name__ == "__main__":
    main()
