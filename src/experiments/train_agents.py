"""Train PPO or DDPG and select the best checkpoint on validation data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.download import ROOT, load_config
from src.environment.portfolio_env import PortfolioEnv


Algorithm = Literal["PPO", "DDPG"]


class SaveBestNormalizationCallback(EvalCallback):
    """Keep VecNormalize statistics paired with the best validation model."""

    def _on_step(self) -> bool:
        previous_best = self.best_mean_reward
        should_continue = super()._on_step()
        if self.best_mean_reward > previous_best:
            if not isinstance(self.eval_env, VecNormalize):
                raise TypeError("Evaluation environment must use VecNormalize")
            self.eval_env.save(str(Path(self.best_model_save_path) / "best_vecnormalize.pkl"))
        return should_continue


def _environment_factory(
    returns: pd.DataFrame,
    vix: pd.Series,
    *,
    period: list[str],
    config: dict[str, Any],
    training: bool,
) -> PortfolioEnv:
    portfolio = config["portfolio"]
    rl = config["reinforcement_learning"]
    return PortfolioEnv(
        returns,
        vix,
        start=period[0],
        end=period[1],
        episode_length=(
            int(rl["episode_length_trading_days"]) if training else None
        ),
        initial_capital=float(portfolio["initial_capital"]),
        transaction_cost_rate=float(portfolio["transaction_cost_rate"]),
        max_weight=float(portfolio["max_weight"]),
    )


def make_model(
    algorithm: Algorithm,
    env: VecNormalize,
    *,
    config: dict[str, Any],
    seed: int,
) -> PPO | DDPG:
    """Build the configured SB3 policy with reproducible algorithm settings."""
    settings = config["reinforcement_learning"]["training"][algorithm.lower()]
    if algorithm == "PPO":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=float(settings["learning_rate"]),
            n_steps=int(settings["n_steps"]),
            batch_size=int(settings["batch_size"]),
            gamma=float(settings["gamma"]),
            seed=seed,
            device="cpu",
            verbose=0,
        )

    action_dim = int(env.action_space.shape[0])
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim, dtype=np.float32),
        sigma=float(settings["action_noise_std"])
        * np.ones(action_dim, dtype=np.float32),
    )
    return DDPG(
        "MlpPolicy",
        env,
        learning_rate=float(settings["learning_rate"]),
        buffer_size=int(settings["buffer_size"]),
        learning_starts=int(settings["learning_starts"]),
        batch_size=int(settings["batch_size"]),
        tau=float(settings["tau"]),
        gamma=float(settings["gamma"]),
        train_freq=(int(settings["train_freq"]), "step"),
        gradient_steps=int(settings["gradient_steps"]),
        action_noise=action_noise,
        seed=seed,
        device="cpu",
        verbose=0,
    )


def train_agent(
    *,
    algorithm: Algorithm,
    event_name: str,
    config: dict[str, Any],
    returns: pd.DataFrame,
    vix: pd.Series,
    output_dir: Path,
    seed: int | None = None,
    total_timesteps: int | None = None,
    eval_freq: int | None = None,
) -> dict[str, Any]:
    """Train one event/algorithm pair and persist its validation-best model."""
    if algorithm not in {"PPO", "DDPG"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if event_name not in config["events"]:
        raise ValueError(f"Unknown event: {event_name}")
    rl_settings = config["reinforcement_learning"]
    training_settings = rl_settings["training"][algorithm.lower()]
    timesteps = int(
        total_timesteps
        if total_timesteps is not None
        else training_settings["total_timesteps"]
    )
    validation_frequency = int(
        eval_freq if eval_freq is not None else rl_settings["validation"]["eval_freq"]
    )
    if timesteps <= 0 or validation_frequency <= 0:
        raise ValueError("Training timesteps and evaluation frequency must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Training output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    event = config["events"][event_name]
    train_base = DummyVecEnv(
        [
            lambda: Monitor(
                _environment_factory(
                    returns,
                    vix,
                    period=event["training"],
                    config=config,
                    training=True,
                )
            )
        ]
    )
    train_env = VecNormalize(
        train_base,
        norm_obs=True,
        norm_reward=True,
        clip_obs=float(rl_settings["normalization"]["clip_obs"]),
        clip_reward=float(rl_settings["normalization"]["clip_reward"]),
        gamma=float(training_settings["gamma"]),
        training=True,
    )
    eval_base = DummyVecEnv(
        [
            lambda: Monitor(
                _environment_factory(
                    returns,
                    vix,
                    period=event["validation"],
                    config=config,
                    training=False,
                )
            )
        ]
    )
    eval_env = VecNormalize(
        eval_base,
        norm_obs=True,
        norm_reward=False,
        clip_obs=float(rl_settings["normalization"]["clip_obs"]),
        training=False,
    )

    effective_seed = int(
        config["reproducibility"]["random_seed"] if seed is None else seed
    )
    model = make_model(algorithm, train_env, config=config, seed=effective_seed)
    checkpoint_dir = output_dir / "selected"
    checkpoint_dir.mkdir()
    callback = SaveBestNormalizationCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(output_dir / "evaluation"),
        eval_freq=validation_frequency,
        n_eval_episodes=int(rl_settings["validation"]["n_eval_episodes"]),
        deterministic=True,
        render=False,
        warn=True,
    )
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
        if callback.best_mean_reward == -np.inf:
            raise RuntimeError(
                "No validation checkpoint was evaluated; increase training "
                "timesteps or lower the evaluation frequency."
            )
        final_model_path = output_dir / "final_model"
        model.save(str(final_model_path))
        train_env.save(str(output_dir / "final_vecnormalize.pkl"))
        model_path = checkpoint_dir / "best_model.zip"
        normalization_path = checkpoint_dir / "best_vecnormalize.pkl"
        if not model_path.exists() or not normalization_path.exists():
            raise RuntimeError("The best validation model or normalization state was not saved")
        metadata = {
            "algorithm": algorithm,
            "event": event_name,
            "selected_by": "highest_mean_validation_episode_reward",
            "validation_reward_units": "sum_of_net_daily_log_returns",
            "validation_mean_reward": float(callback.best_mean_reward),
            "validation_episodes": 1,
            "training_timesteps_requested": timesteps,
            "evaluation_frequency_steps": validation_frequency,
            "algorithm_settings": training_settings,
            "seed": effective_seed,
            "training_period": event["training"],
            "validation_period": event["validation"],
            "normalization": {
                "observations": True,
                "rewards_during_training": True,
                "statistics_fitted_on": "training_only",
                "validation_reward_normalization": False,
            },
            "package_versions": {
                "stable-baselines3": version("stable-baselines3"),
                "gymnasium": version("gymnasium"),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
            },
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_model": "selected/best_model.zip",
            "selected_normalization": "selected/best_vecnormalize.pkl",
            "last_model": "final_model.zip",
            "last_training_normalization": "final_vecnormalize.pkl",
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
        return metadata
    finally:
        train_env.close()
        eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--algorithm", required=True, choices=["PPO", "DDPG"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument(
        "--returns",
        type=Path,
        default=ROOT / "data/processed/daily_returns/daily_returns.csv",
    )
    parser.add_argument(
        "--vix",
        type=Path,
        default=ROOT / "data/processed/daily_returns/vix.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.event not in config["events"]:
        parser.error(f"unknown event {args.event!r}; configured events: {', '.join(config['events'])}")
    returns = pd.read_csv(args.returns, index_col="Date", parse_dates=True)
    vix = pd.read_csv(args.vix, index_col="Date", parse_dates=True)["VIX"]
    output_dir = args.output_dir or (
        ROOT / "results/models" / args.event / args.algorithm.lower()
    )
    metadata = train_agent(
        algorithm=args.algorithm,
        event_name=args.event,
        config=config,
        returns=returns,
        vix=vix,
        output_dir=output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
    )
    print(
        f"Selected {args.algorithm} for {args.event}: validation reward "
        f"{metadata['validation_mean_reward']:.6f}; model saved to "
        f"{output_dir / metadata['selected_model']}"
    )


if __name__ == "__main__":
    main()
