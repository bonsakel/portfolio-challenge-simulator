"""Registry of trained PPO/DDPG models and the compatibility check run before any model is used.

A model is only valid for the universe, constraints and feature layout it was trained on. The
research models' metadata does not record the universe or cost settings, so those come from the
research configuration they were trained with (`configs/experiment.yaml`); models trained in the
app record everything in their own `metadata.json`. Nothing is ever matched loosely: a model is
either compatible or the reasons it is not are returned.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

OBSERVATION_SIZE = 191  # 9 assets x 20-day returns + 9 weights + VIX + 20-day VIX change
ACTION_SIZE = 9
RESEARCH_PREFIX = "research"
USER_PREFIX = "user"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    algorithm: str  # "PPO" | "DDPG"
    label: str
    source: str  # "research" | "user"
    path: Path
    universe: tuple[str, ...]
    max_weight: float
    transaction_cost_rate: float
    observation_size: int
    action_size: int
    training_period: tuple[str, str]
    validation_period: tuple[str, str]
    seed: int | None
    trained_on: str  # event name or "custom"
    timesteps: int | None
    package_versions: dict[str, str] = field(default_factory=dict)

    @property
    def model_file(self) -> Path:
        return self.path / "selected" / "best_model.zip"

    @property
    def normalization_file(self) -> Path:
        return self.path / "selected" / "best_vecnormalize.pkl"

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["universe"] = list(self.universe)
        data["training_period"] = list(self.training_period)
        data["validation_period"] = list(self.validation_period)
        return data


@dataclass(frozen=True)
class Compatibility:
    compatible: bool
    issues: list[str]  # reasons the model must not be used
    warnings: list[str]  # allowed, but the user should know

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def check_compatibility(
    spec: ModelSpec,
    *,
    assets: list[str],
    max_weight: float,
    transaction_cost_rate: float,
    first_decision: date,
    last_decision: date,
    equal_initial_weights: bool = True,
) -> Compatibility:
    """Decide whether `spec` may be evaluated on this portfolio and window."""
    issues: list[str] = []
    warnings: list[str] = []
    universe = list(spec.universe)

    if assets != universe:
        if sorted(assets) == sorted(universe):
            issues.append(
                "The portfolio has the same ETFs in a different order. The model's inputs are "
                f"position-based, so the order must be exactly {', '.join(universe)}."
            )
        else:
            missing = sorted(set(universe) - set(assets))
            extra = sorted(set(assets) - set(universe))
            parts = []
            if missing:
                parts.append(f"missing {', '.join(missing)}")
            if extra:
                parts.append(f"extra {', '.join(extra)}")
            issues.append(
                f"Different asset universe ({'; '.join(parts)}). The model was trained on "
                f"{', '.join(universe)} and cannot be applied to another universe; train a new model."
            )
    if spec.observation_size != OBSERVATION_SIZE or spec.action_size != ACTION_SIZE:
        issues.append(
            f"The model expects {spec.observation_size} inputs and {spec.action_size} outputs, but "
            f"the environment provides {OBSERVATION_SIZE} and {ACTION_SIZE}."
        )
    if not math.isclose(spec.max_weight, max_weight, abs_tol=1e-9):
        issues.append(
            f"The model was trained with a {spec.max_weight:.0%} position cap, but this run uses "
            f"{max_weight:.0%}."
        )
    if not math.isclose(spec.transaction_cost_rate, transaction_cost_rate, abs_tol=1e-12):
        issues.append(
            f"The model was trained with a {spec.transaction_cost_rate:.4%} transaction cost, but "
            f"this run uses {transaction_cost_rate:.4%}."
        )
    if not equal_initial_weights:
        issues.append(
            "RL agents start from equal weights, as in the research. Use equal initial weights "
            "for the portfolio to evaluate PPO or DDPG."
        )

    train_start = date.fromisoformat(spec.training_period[0])
    validation_end = date.fromisoformat(spec.validation_period[1])
    if first_decision <= validation_end and last_decision >= train_start:
        issues.append(
            f"The window {first_decision} to {last_decision} overlaps the model's training or "
            f"validation data ({train_start} to {validation_end}). Results would be in-sample and "
            "must not be treated as a test."
        )
    elif last_decision < train_start:
        warnings.append(
            f"The window ends before the model's training data begins ({train_start}), so the model "
            "learned from market history that came after this window."
        )
    return Compatibility(compatible=not issues, issues=issues, warnings=warnings)


def _research_specs(results_dir: Path, config_path: Path) -> list[ModelSpec]:
    if not config_path.is_file() or not (results_dir / "models").is_dir():
        return []
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    for meta_path in sorted((results_dir / "models").glob("*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        event, algorithm = meta["event"], meta["algorithm"]
        specs.append(
            ModelSpec(
                id=f"{RESEARCH_PREFIX}/{event}/{algorithm.lower()}",
                algorithm=algorithm,
                label=f"{algorithm} - {event.replace('_', ' ')} (research)",
                source="research",
                path=meta_path.parent,
                universe=tuple(config["assets"]),
                max_weight=float(config["portfolio"]["max_weight"]),
                transaction_cost_rate=float(config["portfolio"]["transaction_cost_rate"]),
                observation_size=int(config["reinforcement_learning"]["observation"]["expected_size"]),
                action_size=len(config["assets"]),
                training_period=tuple(meta["training_period"]),
                validation_period=tuple(meta["validation_period"]),
                seed=meta.get("seed"),
                trained_on=event,
                timesteps=meta.get("training_timesteps_requested"),
                package_versions=meta.get("package_versions", {}),
            )
        )
    return specs


def _user_specs(user_dir: Path) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    paths = sorted(user_dir.glob("*/metadata.json")) if user_dir.is_dir() else []
    for meta_path in paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        env = meta["environment"]
        algorithm = meta["algorithm"]
        specs.append(
            ModelSpec(
                id=f"{USER_PREFIX}/{meta_path.parent.name}",
                algorithm=algorithm,
                label=f"{algorithm} - trained in PortfolioLab ({meta_path.parent.name})",
                source="user",
                path=meta_path.parent,
                universe=tuple(env["universe"]),
                max_weight=float(env["max_weight"]),
                transaction_cost_rate=float(env["transaction_cost_rate"]),
                observation_size=int(env["observation_size"]),
                action_size=int(env["action_size"]),
                training_period=tuple(meta["training_period"]),
                validation_period=tuple(meta["validation_period"]),
                seed=meta.get("seed"),
                trained_on="custom",
                timesteps=meta.get("training_timesteps_requested"),
                package_versions=meta.get("package_versions", {}),
            )
        )
    return specs


class ModelRegistry:
    def __init__(self, results_dir: Path, config_path: Path, user_dir: Path) -> None:
        self._results_dir, self._config_path, self._user_dir = results_dir, config_path, user_dir

    def list(self) -> list[ModelSpec]:
        specs = [*_research_specs(self._results_dir, self._config_path), *_user_specs(self._user_dir)]
        return [s for s in specs if s.model_file.is_file() and s.normalization_file.is_file()]

    def get(self, model_id: str) -> ModelSpec | None:
        return next((s for s in self.list() if s.id == model_id), None)
