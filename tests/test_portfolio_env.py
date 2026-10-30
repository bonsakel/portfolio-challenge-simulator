from math import isclose, log

import numpy as np
import pandas as pd
import pytest

from src.environment.portfolio_env import PortfolioEnv


@pytest.fixture
def market_data() -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.bdate_range("2020-01-01", periods=80)
    assets = [f"ETF_{index}" for index in range(9)]
    returns = pd.DataFrame(0.0, index=dates, columns=assets)
    returns.iloc[:, 0] = np.arange(len(dates), dtype=float) / 10_000.0
    vix = pd.Series(15.0 + np.arange(len(dates)), index=dates, name="VIX")
    return returns, vix


def make_env(
    market_data: tuple[pd.DataFrame, pd.Series],
    *,
    episode_length: int | None = 5,
) -> PortfolioEnv:
    returns, vix = market_data
    return PortfolioEnv(
        returns,
        vix,
        start=returns.index[30],
        end=returns.index[50],
        episode_length=episode_length,
    )


def test_observation_contains_only_pre_decision_returns_and_vix(
    market_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, vix = market_data
    env = make_env(market_data)

    observation, info = env.reset(options={"start_index": 30})

    assert observation.shape == (191,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(
        observation[: 20 * 9].reshape(20, 9),
        returns.iloc[10:30].to_numpy(dtype=np.float32),
    )
    np.testing.assert_allclose(observation[180:189], np.full(9, 1 / 9))
    assert observation[189] == pytest.approx(vix.iloc[29])
    assert observation[190] == pytest.approx(vix.iloc[29] - vix.iloc[9])
    assert info["date"] == returns.index[30].date().isoformat()
    env.close()


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (np.ones(9, dtype=np.float32), np.full(9, 1 / 9)),
        (np.zeros(9, dtype=np.float32), np.full(9, 1 / 9)),
    ],
)
def test_action_projection_produces_valid_portfolio_weights(
    action: np.ndarray, expected: np.ndarray
) -> None:
    weights = PortfolioEnv.project_action(action, 0.30)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.30 + 1e-8).all()
    np.testing.assert_allclose(weights, expected, atol=1e-8)


def test_action_projection_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        PortfolioEnv.project_action(np.array([np.nan] * 9), 0.30)
    with pytest.raises(ValueError, match="action space"):
        PortfolioEnv.project_action(np.array([-0.1] * 9), 0.30)
    with pytest.raises(ValueError, match="support a fully invested"):
        PortfolioEnv.project_action(np.array([0.5] * 9), 0.10)


def test_step_applies_cost_once_and_returns_net_log_reward(
    market_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, _ = market_data
    env = make_env(market_data)
    env.reset(options={"start_index": 30})
    action = np.full(9, -1.0, dtype=np.float32)
    action[0] = 1.0

    observation, reward, terminated, truncated, info = env.step(action)

    target = np.asarray(list(info["target_weights"].values()))
    turnover = np.abs(target - np.full(9, 1 / 9)).sum()
    fee = 0.001 * 100_000.0 * turnover
    gross_return = float(target @ returns.iloc[30].to_numpy())
    expected_value = (100_000.0 - fee) * (1.0 + gross_return)
    expected_net_return = expected_value / 100_000.0 - 1.0

    assert not terminated
    assert not truncated
    assert isclose(info["turnover"], turnover)
    assert isclose(info["transaction_cost"], fee)
    assert isclose(info["traded_value"], 100_000.0 * turnover)
    assert isclose(info["gross_return"], gross_return)
    assert isclose(info["net_return"], expected_net_return)
    assert isclose(reward, log(1.0 + expected_net_return))
    assert info["portfolio_value"] == pytest.approx(expected_value)
    assert env.observation_space.contains(observation)
    assert observation[: 20 * 9].reshape(20, 9)[-1, 0] == pytest.approx(
        returns.iloc[30, 0]
    )
    env.close()


def test_reset_is_seeded_and_episode_ends_after_configured_steps(
    market_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    env = make_env(market_data, episode_length=5)

    _, first_info = env.reset(seed=123)
    _, second_info = env.reset(seed=123)
    assert first_info["date"] == second_info["date"]

    _, info = env.reset(options={"start_index": 30})
    assert info["date"] == market_data[0].index[30].date().isoformat()
    for step in range(5):
        _, _, terminated, truncated, _ = env.step(
            np.full(9, 1 / 9, dtype=np.float32)
        )
        assert terminated is (step == 4)
        assert not truncated
    with pytest.raises(RuntimeError, match="after episode termination"):
        env.step(np.full(9, 1 / 9, dtype=np.float32))
    env.close()


def test_full_period_episode_runs_to_inclusive_end(
    market_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    returns, vix = market_data
    env = PortfolioEnv(
        returns,
        vix,
        start=returns.index[30],
        end=returns.index[35],
        episode_length=None,
    )
    env.reset()
    for step in range(6):
        _, _, terminated, _, info = env.step(
            np.full(9, 1 / 9, dtype=np.float32)
        )
        assert terminated is (step == 5)
    assert info["date"] == returns.index[35].date().isoformat()
    env.close()


def test_gymnasium_environment_checker_accepts_portfolio_env(
    market_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    from gymnasium.utils.env_checker import check_env

    env = make_env(market_data)

    with pytest.warns(UserWarning, match="Box observation space"):
        check_env(env, skip_render_check=True)

    env.close()
