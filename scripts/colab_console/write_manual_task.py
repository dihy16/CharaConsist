"""Write a validated temporary task specification for the remote worker."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from characonsist.experiments.conditions import (
    find_component_condition,
    find_action_binding_condition,
    find_entity_routing_condition,
    find_condition,
    find_role_action_condition,
    build_component_conditions,
    build_action_binding_conditions,
    build_entity_routing_conditions,
    build_role_action_conditions,
    build_sweep_conditions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("lambda", "component", "role", "binding", "routing"), required=True
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.mode == "lambda":
        condition = find_condition(build_sweep_conditions([args.value], [args.seed]), args.value, args.seed)
        task = {"method": "run_one", "value": condition.action_gate_strength}
    elif args.mode == "component":
        condition = find_component_condition(
            build_component_conditions([args.value], [args.seed]), args.value, args.seed
        )
        task = {"method": "run_component", "value": condition.consistency_mode}
    elif args.mode == "role":
        condition = find_role_action_condition(
            build_role_action_conditions([args.value], [args.seed]), args.value, args.seed
        )
        task = {"method": "run_role_action", "value": condition.role_action_bias_strength}
    elif args.mode == "binding":
        condition = build_action_binding_conditions([args.value], [args.seed])[0]
        condition = find_action_binding_condition(
            [condition], condition.beta, condition.gamma, args.seed
        )
        task = {
            "method": "run_action_binding",
            "value": [condition.beta, condition.gamma],
        }
    else:
        parts = args.value.split(":")
        if len(parts) != 3:
            raise ValueError("Routing value must be mode:beta:gamma.")
        mode, beta, gamma = parts
        condition = build_entity_routing_conditions(
            [mode], [f"{beta}:{gamma}"], [args.seed]
        )[0]
        condition = find_entity_routing_condition(
            [condition], condition.entity_routing_mode,
            condition.beta, condition.gamma, args.seed,
        )
        task = {
            "method": "run_entity_routing",
            "value": [condition.entity_routing_mode, condition.beta, condition.gamma],
        }

    task.update(
        prompt=args.prompt,
        seed=condition.seed,
        relative_output=(condition.output_prefix / Path(args.prompt).with_suffix("")).as_posix(),
    )
    Path(args.output).write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
