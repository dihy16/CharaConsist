"""Pure helpers for reproducible action-gating sweep conditions."""

from dataclasses import dataclass
import math
from pathlib import Path


DEFAULT_ACTION_GATE_STRENGTHS = "0,0.25,0.5,0.75,1"
DEFAULT_SEEDS = "2025"
DEFAULT_CONSISTENCY_MODES = "prompt_only,attention_only,full"
CONSISTENCY_MODES = ("prompt_only", "attention_only", "full")


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
