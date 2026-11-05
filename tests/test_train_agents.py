import json

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import DDPG, PPO

from src.data.download import load_config
from src.experiments.train_agents import Algorithm, make_model, train_agent


@pytest.fixture
def training_inputs() -> tuple[dict, pd.DataFrame, pd.Series]:
    config = load_config()
    dates = pd.bdate_range("2000-01-03", periods=80)
    assets = config["assets"]
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    vix = pd.Series(20.0 + rng.normal(0.0, 1.0, len(dates)), index=dates, name="VIX")
    config["events"] = {
        "smoke": {
            "training": [dates[30].date().isoformat(), dates[45].date().isoformat()],
            "validation": [dates[46].date().isoformat(), dates[50].date().isoformat()],
            "downturn_test": [dates[51].date().isoformat(), dates[55].date().isoformat()],
            "recovery_observation": [
                dates[56].date().isoformat(),
                dates[60].date().isoformat(),
            ],
        }
    }
    config["reinforcement_learning"]["episode_length_trading_days"] = 4
    config["reinforcement_learning"]["validation"]["eval_freq"] = 4
    config["reinforcement_learning"]["training"]["ppo"].update(
        {"total_timesteps": 8, "n_steps": 4, "batch_size": 4}
    )
    config["reinforcement_learning"]["training"]["ddpg"].update(
        {
            "total_timesteps": 8,
            "buffer_size": 32,
            "learning_starts": 2,
            "batch_size": 4,
        }
    )
    return config, returns, vix


@pytest.mark.parametrize(
    ("algorithm", "expected_type"),
    [("PPO", PPO), ("DDPG", DDPG)],
)
def test_make_model_builds_configured_sb3_algorithm(
    algorithm: str,
    expected_type: type[PPO] | type[DDPG],
    training_inputs: tuple[dict, pd.DataFrame, pd.Series],
) -> None:
    config, returns, vix = training_inputs
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.environment.portfolio_env import PortfolioEnv

    env = VecNormalize(
        DummyVecEnv(
            [
                lambda: PortfolioEnv(
                    returns,
                    vix,
                    start=config["events"]["smoke"]["training"][0],
                    end=config["events"]["smoke"]["training"][1],
                    episode_length=4,
                )
            ]
        )
    )

    model = make_model(algorithm, env, config=config, seed=42)

    assert isinstance(model, expected_type)
    env.close()


@pytest.mark.parametrize("algorithm", ["PPO", "DDPG"])
def test_train_agent_saves_validation_best_model_and_normalization(
    algorithm: Algorithm,
    training_inputs: tuple[dict, pd.DataFrame, pd.Series],
    tmp_path,
) -> None:
    config, returns, vix = training_inputs
    output_dir = tmp_path / algorithm.lower()

    metadata = train_agent(
        algorithm=algorithm,
        event_name="smoke",
        config=config,
        returns=returns,
        vix=vix,
        output_dir=output_dir,
        seed=123,
    )

    assert metadata["selected_by"] == "highest_mean_validation_episode_reward"
    assert metadata["normalization"]["statistics_fitted_on"] == "training_only"
    assert (output_dir / "selected/best_model.zip").is_file()
    assert (output_dir / "selected/best_vecnormalize.pkl").is_file()
    assert (output_dir / "final_model.zip").is_file()
    assert (output_dir / "final_vecnormalize.pkl").is_file()
    saved_metadata = json.loads((output_dir / "metadata.json").read_text())
    assert saved_metadata["algorithm"] == algorithm
    assert saved_metadata["event"] == "smoke"
    assert saved_metadata["seed"] == 123


def test_train_agent_does_not_overwrite_existing_run_directory(
    training_inputs: tuple[dict, pd.DataFrame, pd.Series],
    tmp_path,
) -> None:
    config, returns, vix = training_inputs
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("existing output")

    with pytest.raises(FileExistsError, match="not empty"):
        train_agent(
            algorithm="PPO",
            event_name="smoke",
            config=config,
            returns=returns,
            vix=vix,
            output_dir=output_dir,
        )

    assert marker.read_text() == "existing output"
