"""Subprocess entry point that trains one PPO/DDPG model with the research training code.

Run as `python -m app.rl.train_worker <job_dir>`. The job directory holds `spec.json`,
`returns.csv` and `vix.csv` written by the training manager. Training itself is
`src.experiments.train_agents.train_agent`, unchanged, so the environment, algorithm settings,
normalization and validation-based checkpoint selection are exactly the research ones.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from app.config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rl.registry import ACTION_SIZE, OBSERVATION_SIZE  # noqa: E402
from src.experiments.train_agents import train_agent  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def build_config(spec: dict) -> dict:
    """Research configuration with the user's portfolio constraints and custom date windows."""
    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    config["assets"] = list(spec["universe"])
    config["portfolio"]["max_weight"] = float(spec["max_weight"])
    config["portfolio"]["transaction_cost_rate"] = float(spec["transaction_cost_rate"])
    config["portfolio"]["initial_capital"] = float(spec["initial_capital"])
    config["reproducibility"]["random_seed"] = int(spec["seed"])
    config["events"] = {
        "custom": {"training": spec["training_period"], "validation": spec["validation_period"]}
    }
    return config


def run(job_dir: Path) -> dict:
    spec = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
    returns = pd.read_csv(job_dir / "returns.csv", index_col="Date", parse_dates=True)
    vix = pd.read_csv(job_dir / "vix.csv", index_col="Date", parse_dates=True)["VIX"]
    config = build_config(spec)
    output_dir = Path(spec["output_dir"])
    metadata = train_agent(
        algorithm=spec["algorithm"],
        event_name="custom",
        config=config,
        returns=returns[spec["universe"]],
        vix=vix,
        output_dir=output_dir,
        total_timesteps=int(spec["total_timesteps"]),
        eval_freq=int(spec["eval_freq"]),
    )
    metadata["environment"] = {
        "universe": list(spec["universe"]),
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "max_weight": float(spec["max_weight"]),
        "transaction_cost_rate": float(spec["transaction_cost_rate"]),
        "initial_capital": float(spec["initial_capital"]),
        "return_lookback_trading_days": 20,
        "features": "20-day asset returns, current weights, VIX close, 20-day VIX change (points)",
        "episode_length_trading_days": int(config["reinforcement_learning"]["episode_length_trading_days"]),
        "action": "affine map of [-1, 1] to [0, 1], then projection onto the capped simplex",
        "reward": config["reinforcement_learning"]["reward"],
        "price_field": spec.get("price_field", "Adj Close"),
        "vix_source": "CBOE VIX history (research dataset)",
    }
    metadata["normalization"]["clip_obs"] = float(config["reinforcement_learning"]["normalization"]["clip_obs"])
    metadata["normalization"]["clip_reward"] = float(config["reinforcement_learning"]["normalization"]["clip_reward"])
    metadata["normalization"]["statistics_file"] = "selected/best_vecnormalize.pkl"
    metadata["trained_on"] = "custom"
    metadata["data_rows"] = {"first": str(returns.index[0].date()), "last": str(returns.index[-1].date())}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    run(Path(sys.argv[1]))
