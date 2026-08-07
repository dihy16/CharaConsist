"""Pure helpers for reproducible action-gating sweep conditions."""

from dataclasses import dataclass
import math
from pathlib import Path


DEFAULT_ACTION_GATE_STRENGTHS = "0,0.25,0.5,0.75,1"
DEFAULT_SEEDS = "2025"
DEFAULT_CONSISTENCY_MODES = "prompt_only,attention_only,full"
DEFAULT_ROLE_ACTION_BIAS_STRENGTHS = "0,1"
DEFAULT_ACTION_BINDING_CONDITIONS = "0:0,1:0,1:0.5,2:1"
CONSISTENCY_MODES = ("prompt_only", "attention_only", "full")
ENTITY_ROUTING_MODES = ("off", "hard")


def _csv_items(value, name):
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = list(value)
    if not items or any(item == "" for item in items):
        raise ValueError(f"{name} must be a non-empty comma-separated list.")
    return items


def parse_action_gate_strengths(value):
    """Parse unique finite lambda values in [0, 1], preserving order."""
    strengths = []
    for item in _csv_items(value, "action gate strengths"):
        try:
            strength = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid action gate strength: {item!r}.") from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(f"Action gate strength must be within [0, 1], got {item!r}.")
        if strength not in strengths:
            strengths.append(strength)
    return strengths


def parse_role_action_bias_strengths(value):
    """Parse unique finite non-negative role-bias strengths."""
    strengths = []
    for item in _csv_items(value, "role-action bias strengths"):
        try:
            strength = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid role-action bias strength: {item!r}.") from exc
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"Role-action bias strength must be non-negative, got {item!r}.")
        if strength not in strengths:
            strengths.append(strength)
    return strengths


def parse_action_binding_conditions(value):
    """Parse unique beta:gamma pairs with finite non-negative values."""
    conditions = []
    for item in _csv_items(value, "action-binding conditions"):
        parts = item if isinstance(item, (tuple, list)) else str(item).split(":")
        if len(parts) != 2:
            raise ValueError(f"Action-binding condition must be beta:gamma, got {item!r}.")
        try:
            beta, gamma = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid action-binding condition: {item!r}.") from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in (beta, gamma)):
            raise ValueError(f"Action-binding strengths must be finite and non-negative: {item!r}.")
        pair = (beta, gamma)
        if pair not in conditions:
            conditions.append(pair)
    return conditions


def parse_seeds(value):
    """Parse unique non-negative integer seeds, preserving order."""
    seeds = []
    for item in _csv_items(value, "seeds"):
        try:
            seed = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid seed: {item!r}.") from exc
        if seed < 0:
            raise ValueError(f"Seeds must be non-negative, got {seed}.")
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def parse_consistency_modes(value):
    """Parse unique component-ablation modes in stable order."""
    modes = []
    for item in _csv_items(value, "consistency modes"):
        mode = str(item).strip().lower().replace("-", "_")
        if mode not in CONSISTENCY_MODES:
            allowed = ", ".join(CONSISTENCY_MODES)
            raise ValueError(f"Unknown consistency mode {item!r}; choose from {allowed}.")
        if mode not in modes:
            modes.append(mode)
    return modes


def parse_entity_routing_modes(value):
    """Parse unique entity-routing modes in stable order."""
    modes = []
    for item in _csv_items(value, "entity routing modes"):
        mode = str(item).strip().lower()
        if mode not in ENTITY_ROUTING_MODES:
            raise ValueError(
                f"Unknown entity routing mode {item!r}; choose from off, hard."
            )
        if mode not in modes:
            modes.append(mode)
    return modes


def lambda_label(strength):
    """Return a stable, sortable directory label such as lambda_0p50."""
    strength = parse_action_gate_strengths([strength])[0]
    text = f"{strength:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".00"
    elif len(text.rsplit(".", 1)[1]) == 1:
        text += "0"
    return f"lambda_{text.replace('.', 'p')}"


@dataclass(frozen=True)
class SweepCondition:
    action_gate_strength: float
    seed: int

    @property
    def lambda_label(self):
        return lambda_label(self.action_gate_strength)

    @property
    def key(self):
        return f"{self.lambda_label}/seed_{self.seed}"

    @property
    def output_prefix(self):
        return Path(self.lambda_label) / f"seed_{self.seed}" / "bg_fg"


def build_sweep_conditions(action_gate_strengths, seeds):
    """Build the lambda-major condition order used by local and Colab runs."""
    strengths = parse_action_gate_strengths(action_gate_strengths)
    parsed_seeds = parse_seeds(seeds)
    return [
        SweepCondition(action_gate_strength=strength, seed=seed)
        for strength in strengths
        for seed in parsed_seeds
    ]


def find_condition(conditions, action_gate_strength, seed):
    """Return an allowed condition or reject an unexpected worker request."""
    strength = parse_action_gate_strengths([action_gate_strength])[0]
    parsed_seed = parse_seeds([seed])[0]
    for condition in conditions:
        if condition.action_gate_strength == strength and condition.seed == parsed_seed:
            return condition
    raise ValueError(f"Sweep condition lambda={strength}, seed={parsed_seed} was not configured.")


@dataclass(frozen=True)
class ComponentCondition:
    consistency_mode: str
    seed: int

    @property
    def key(self):
        return f"component_ablation/{self.consistency_mode}/seed_{self.seed}"

    @property
    def output_prefix(self):
        return Path("component_ablation") / self.consistency_mode / f"seed_{self.seed}" / "bg_fg"


def build_component_conditions(consistency_modes, seeds):
    """Build mode-major, seed-matched component-ablation conditions."""
    modes = parse_consistency_modes(consistency_modes)
    parsed_seeds = parse_seeds(seeds)
    return [
        ComponentCondition(consistency_mode=mode, seed=seed)
        for mode in modes
        for seed in parsed_seeds
    ]


def find_component_condition(conditions, consistency_mode, seed):
    """Return an allowed component condition or reject an unexpected request."""
    mode = parse_consistency_modes([consistency_mode])[0]
    parsed_seed = parse_seeds([seed])[0]
    for condition in conditions:
        if condition.consistency_mode == mode and condition.seed == parsed_seed:
            return condition
    raise ValueError(f"Component condition mode={mode}, seed={parsed_seed} was not configured.")


def role_bias_label(strength):
    strength = parse_role_action_bias_strengths([strength])[0]
    text = f"{strength:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".00"
    elif len(text.rsplit(".", 1)[1]) == 1:
        text += "0"
    return f"role_bias_{text.replace('.', 'p')}"


@dataclass(frozen=True)
class RoleActionCondition:
    role_action_bias_strength: float
    seed: int

    @property
    def key(self):
        return f"role_action_ablation/{role_bias_label(self.role_action_bias_strength)}/seed_{self.seed}"

    @property
    def output_prefix(self):
        return Path(self.key) / "bg_fg"


def build_role_action_conditions(strengths, seeds):
    return [
        RoleActionCondition(role_action_bias_strength=strength, seed=seed)
        for strength in parse_role_action_bias_strengths(strengths)
        for seed in parse_seeds(seeds)
    ]


def find_role_action_condition(conditions, strength, seed):
    strength = parse_role_action_bias_strengths([strength])[0]
    parsed_seed = parse_seeds([seed])[0]
    for condition in conditions:
        if condition.role_action_bias_strength == strength and condition.seed == parsed_seed:
            return condition
    raise ValueError(f"Role-action condition strength={strength}, seed={parsed_seed} was not configured.")


def action_binding_label(beta, gamma):
    beta, gamma = parse_action_binding_conditions([f"{beta}:{gamma}"])[0]
    encode = lambda value: f"{value:.2f}".replace(".", "p")
    return f"beta_{encode(beta)}_gamma_{encode(gamma)}"


@dataclass(frozen=True)
class ActionBindingCondition:
    beta: float
    gamma: float
    seed: int

    @property
    def key(self):
        return f"action_binding_ablation/{action_binding_label(self.beta, self.gamma)}/seed_{self.seed}"

    @property
    def output_prefix(self):
        return Path(self.key) / "bg_fg"


def build_action_binding_conditions(conditions, seeds):
    return [
        ActionBindingCondition(beta=beta, gamma=gamma, seed=seed)
        for beta, gamma in parse_action_binding_conditions(conditions)
        for seed in parse_seeds(seeds)
    ]


def find_action_binding_condition(conditions, beta, gamma, seed):
    pair = parse_action_binding_conditions([f"{beta}:{gamma}"])[0]
    parsed_seed = parse_seeds([seed])[0]
    for condition in conditions:
        if (condition.beta, condition.gamma) == pair and condition.seed == parsed_seed:
            return condition
    raise ValueError(
        f"Action-binding condition beta={pair[0]}, gamma={pair[1]}, seed={parsed_seed} was not configured."
    )


@dataclass(frozen=True)
class EntityRoutingCondition:
    entity_routing_mode: str
    beta: float
    gamma: float
    seed: int

    @property
    def key(self):
        return (
            f"entity_routing_ablation/routing_{self.entity_routing_mode}/"
            f"{action_binding_label(self.beta, self.gamma)}/seed_{self.seed}"
        )

    @property
    def output_prefix(self):
        return Path(self.key) / "bg_fg"


def build_entity_routing_conditions(modes, action_conditions, seeds):
    return [
        EntityRoutingCondition(
            entity_routing_mode=mode, beta=beta, gamma=gamma, seed=seed
        )
        for mode in parse_entity_routing_modes(modes)
        for beta, gamma in parse_action_binding_conditions(action_conditions)
        for seed in parse_seeds(seeds)
    ]


def find_entity_routing_condition(conditions, mode, beta, gamma, seed):
    mode = parse_entity_routing_modes([mode])[0]
    beta, gamma = parse_action_binding_conditions([f"{beta}:{gamma}"])[0]
    seed = parse_seeds([seed])[0]
    for condition in conditions:
        if (
            condition.entity_routing_mode == mode
            and condition.beta == beta
            and condition.gamma == gamma
            and condition.seed == seed
        ):
            return condition
    raise ValueError(
        f"Entity-routing condition mode={mode}, beta={beta}, gamma={gamma}, "
        f"seed={seed} was not configured."
    )
