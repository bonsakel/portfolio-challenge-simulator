"""Gymnasium environment for cost-aware, long-only ETF portfolios."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from numpy.typing import NDArray


class PortfolioEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Make a portfolio decision before each day's ETF return is realized.

    An observation at date ``t`` contains ETF returns through the prior
    trading day, portfolio weights before trading, the prior close's VIX,
    and the absolute VIX point change over 20 trading-day intervals.
    Actions are projected onto the fully invested capped simplex. The
    transaction fee is deducted once from pre-trade value; reward is the
    natural log of the resulting net daily portfolio return.
    """

    metadata = {"render_modes": []}
    RETURN_LOOKBACK = 20
    EXPECTED_OBSERVATION_SIZE = 191

    def __init__(
        self,
        daily_returns: pd.DataFrame,
        vix: pd.Series | pd.DataFrame,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        episode_length: int | None = 252,
        initial_capital: float = 100_000.0,
        transaction_cost_rate: float = 0.001,
        max_weight: float = 0.30,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None:
            raise ValueError("This environment does not support rendering")
        if daily_returns.empty:
            raise ValueError("Daily returns must not be empty")
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            raise TypeError("Daily returns must have a DatetimeIndex")
        if (
            daily_returns.index.has_duplicates
            or not daily_returns.index.is_monotonic_increasing
        ):
            raise ValueError("Daily return dates must be unique and chronological")
        if daily_returns.columns.has_duplicates or len(daily_returns.columns) == 0:
            raise ValueError("Daily returns must have unique asset columns")
        if not np.isfinite(daily_returns.to_numpy(dtype=float)).all():
            raise ValueError("Daily returns must not contain missing or infinite values")
        if (daily_returns.to_numpy(dtype=float) < -1.0).any():
            raise ValueError("An asset return cannot be less than -100%")
        if not np.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("Initial capital must be finite and positive")
        if (
            not np.isfinite(transaction_cost_rate)
            or transaction_cost_rate < 0
            or transaction_cost_rate >= 1
        ):
            raise ValueError("Transaction cost rate must be in [0, 1)")
        if not np.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
            raise ValueError("Maximum weight must be in (0, 1]")
        if max_weight * len(daily_returns.columns) < 1.0 - 1e-12:
            raise ValueError(
                "Maximum weight cannot support a fully invested portfolio"
            )
        if episode_length is not None and (
            isinstance(episode_length, bool)
            or not isinstance(episode_length, (int, np.integer))
            or episode_length <= 0
        ):
            raise ValueError(
                "Episode length must be a positive integer number of trading days"
            )

        if isinstance(vix, pd.DataFrame):
            if list(vix.columns) != ["VIX"]:
                raise ValueError("VIX DataFrame must contain exactly one 'VIX' column")
            vix_series = vix["VIX"]
        elif isinstance(vix, pd.Series):
            vix_series = vix.rename("VIX")
        else:
            raise TypeError("VIX must be a pandas Series or one-column DataFrame")
        if not isinstance(vix_series.index, pd.DatetimeIndex):
            raise TypeError("VIX data must have a DatetimeIndex")
        if (
            vix_series.index.has_duplicates
            or not vix_series.index.is_monotonic_increasing
        ):
            raise ValueError("VIX dates must be unique and chronological")
        if not np.isfinite(vix_series.to_numpy(dtype=float)).all():
            raise ValueError("VIX values must be finite")
        if (vix_series.to_numpy(dtype=float) <= 0).any():
            raise ValueError("VIX values must be positive")

        self.assets = tuple(str(asset) for asset in daily_returns.columns)
        if len(self.assets) != 9 or len(set(self.assets)) != 9:
            raise ValueError("Expected nine ETF assets with unique string names")
        self._returns = daily_returns.astype(float).copy()
        self._returns.index = pd.DatetimeIndex(self._returns.index)
        self._vix = vix_series.astype(float).reindex(self._returns.index)
        if self._vix.isna().any():
            missing = self._vix.index[self._vix.isna()][0].date()
            raise ValueError(f"VIX is missing an ETF return date: {missing}")

        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.max_weight = float(max_weight)
        self.episode_length = episode_length
        self._start_date = pd.Timestamp(start)
        self._end_date = pd.Timestamp(end)
        if self._start_date > self._end_date:
            raise ValueError("Environment start must not be after end")
        self._period_start_index = int(self._returns.index.searchsorted(self._start_date))
        end_position = int(self._returns.index.searchsorted(self._end_date, side="right"))
        self._period_end_index = end_position - 1
        if (
            self._period_start_index >= len(self._returns)
            or self._period_end_index < self._period_start_index
        ):
            raise ValueError("No daily returns fall within the configured period")
        if self._period_start_index < self.RETURN_LOOKBACK + 1:
            raise ValueError(
                "At least 21 prior trading-day rows are required before the "
                "configured start for observations and VIX change."
            )
        self._max_episode_start = self._period_end_index
        if episode_length is not None:
            self._max_episode_start -= episode_length - 1
            if self._max_episode_start < self._period_start_index:
                available = self._period_end_index - self._period_start_index + 1
                raise ValueError(
                    f"Period has {available} trading days, fewer than the "
                    f"{episode_length}-day episode length"
                )

        self.action_space = spaces.Box(
            low=-np.ones(len(self.assets), dtype=np.float32),
            high=np.ones(len(self.assets), dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.EXPECTED_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._current_index = self._period_start_index
        self._episode_end_index = self._period_end_index
        self._portfolio_value = self.initial_capital
        self._weights = np.full(
            len(self.assets), 1.0 / len(self.assets), dtype=np.float64
        )
        self._terminated = False

    @staticmethod
    def project_action(
        action: NDArray[np.floating], max_weight: float
    ) -> NDArray[np.float64]:
        """Project desired weights onto the nonnegative capped unit simplex."""
        values = np.asarray(action, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Action must be a non-empty one-dimensional array")
        if not np.isfinite(values).all():
            raise ValueError("Action values must be finite")
        if (values < 0.0).any() or (values > 1.0).any():
            raise ValueError("Action values must be within the action space [0, 1]")
        if not np.isfinite(max_weight) or max_weight <= 0:
            raise ValueError("Maximum weight must be positive and finite")
        if max_weight * len(values) < 1.0 - 1e-12:
            raise ValueError("Maximum weight cannot support a fully invested portfolio")

        low = float(np.min(values) - max_weight)
        high = float(np.max(values))
        for _ in range(80):
            threshold = (low + high) / 2.0
            projected = np.clip(values - threshold, 0.0, max_weight)
            if projected.sum() > 1.0:
                low = threshold
            else:
                high = threshold
        projected = np.clip(values - (low + high) / 2.0, 0.0, max_weight)

        residual = 1.0 - float(projected.sum())
        if residual > 0.0:
            for index in range(len(projected)):
                addition = min(residual, max_weight - projected[index])
                projected[index] += addition
                residual -= addition
                if residual <= 1e-12:
                    break
        elif residual < 0.0:
            for index in range(len(projected)):
                reduction = min(-residual, projected[index])
                projected[index] -= reduction
                residual += reduction
                if residual >= -1e-12:
                    break
        if abs(projected.sum() - 1.0) > 1e-8:
            raise ArithmeticError("Action projection did not produce unit-sum weights")
        return projected

    def _observation(self, decision_index: int) -> NDArray[np.float32]:
        history_start = decision_index - self.RETURN_LOOKBACK
        history = self._returns.iloc[history_start:decision_index].to_numpy(
            dtype=np.float64
        )
        prior_vix_index = decision_index - 1
        prior_vix = float(self._vix.iloc[prior_vix_index])
        vix_change = prior_vix - float(
            self._vix.iloc[prior_vix_index - self.RETURN_LOOKBACK]
        )
        observation = np.concatenate(
            (
                history.reshape(-1),
                self._weights,
                np.asarray([prior_vix, vix_change], dtype=np.float64),
            )
        ).astype(np.float32)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Expected observation shape {self.observation_space.shape}, "
                f"got {observation.shape}"
            )
        if not np.isfinite(observation).all():
            raise ArithmeticError("Environment produced a non-finite observation")
        return observation

    def _info(
        self,
        *,
        decision_index: int,
        gross_return: float | None = None,
        net_return: float | None = None,
        turnover: float = 0.0,
        traded_value: float = 0.0,
        transaction_cost: float = 0.0,
        target_weights: NDArray[np.float64] | None = None,
    ) -> dict[str, object]:
        info: dict[str, object] = {
            "date": self._returns.index[decision_index].date().isoformat(),
            "portfolio_value": self._portfolio_value,
            "turnover": turnover,
            "traded_value": traded_value,
            "transaction_cost": transaction_cost,
            "weights": dict(zip(self.assets, self._weights.tolist(), strict=True)),
        }
        if gross_return is not None:
            info["gross_return"] = gross_return
        if net_return is not None:
            info["net_return"] = net_return
        if target_weights is not None:
            info["target_weights"] = dict(
                zip(self.assets, target_weights.tolist(), strict=True)
            )
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        super().reset(seed=seed)
        options = options or {}
        unknown_options = set(options).difference({"start_index"})
        if unknown_options:
            raise ValueError(f"Unknown reset options: {sorted(unknown_options)}")
        requested_start = options.get("start_index")
        if requested_start is None:
            if self.episode_length is None:
                start_index = self._period_start_index
            else:
                start_index = int(
                    self.np_random.integers(
                        self._period_start_index, self._max_episode_start + 1
                    )
                )
        else:
            if isinstance(requested_start, bool) or not isinstance(
                requested_start, (int, np.integer)
            ):
                raise TypeError("reset option 'start_index' must be an integer")
            start_index = int(requested_start)
            if not self._period_start_index <= start_index <= self._max_episode_start:
                raise ValueError(
                    "Requested episode start is outside the configured period"
                )

        self._current_index = start_index
        if self.episode_length is None:
            self._episode_end_index = self._period_end_index
        else:
            self._episode_end_index = start_index + self.episode_length - 1
        self._portfolio_value = self.initial_capital
        self._weights = np.full(
            len(self.assets), 1.0 / len(self.assets), dtype=np.float64
        )
        self._terminated = False
        observation = self._observation(self._current_index)
        return observation, self._info(decision_index=self._current_index)

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        if self._terminated:
            raise RuntimeError("step() called after episode termination; call reset()")

        decision_index = self._current_index
        action_values = np.asarray(action, dtype=np.float64)
        if action_values.shape != self.action_space.shape:
            raise ValueError(
                f"Action must have shape {self.action_space.shape}; "
                f"received {action_values.shape}"
            )
        if not np.isfinite(action_values).all():
            raise ValueError("Action values must be finite")
        if (action_values < -1.0).any() or (action_values > 1.0).any():
            raise ValueError("Action values must be within the action space [-1, 1]")
        desired_weights = (action_values + 1.0) / 2.0
        target_weights = self.project_action(desired_weights, self.max_weight)
        turnover = float(np.abs(target_weights - self._weights).sum())
        traded_value = self._portfolio_value * turnover
        transaction_cost = self.transaction_cost_rate * traded_value
        value_after_cost = self._portfolio_value - transaction_cost
        asset_returns = self._returns.iloc[decision_index].to_numpy(dtype=np.float64)
        gross_return = float(target_weights @ asset_returns)
        next_value = value_after_cost * (1.0 + gross_return)
        net_return = next_value / self._portfolio_value - 1.0
        if not np.isfinite(next_value) or next_value <= 0.0 or net_return <= -1.0:
            raise ArithmeticError(
                f"Portfolio value became invalid on "
                f"{self._returns.index[decision_index].date()}"
            )

        ending_weights = target_weights * (1.0 + asset_returns) / (1.0 + gross_return)
        reward = float(np.log1p(net_return))
        self._portfolio_value = float(next_value)
        self._weights = ending_weights
        terminated = decision_index >= self._episode_end_index
        self._terminated = terminated
        next_index = decision_index + 1
        if next_index <= self._period_end_index:
            observation = self._observation(next_index)
        else:
            observation = self._observation(decision_index)
        self._current_index = min(next_index, self._period_end_index)
        info = self._info(
            decision_index=decision_index,
            gross_return=gross_return,
            net_return=net_return,
            turnover=turnover,
            traded_value=traded_value,
            transaction_cost=transaction_cost,
            target_weights=target_weights,
        )
        return observation, reward, terminated, False, info


def load_environment_data(
    returns_path: str,
    vix_path: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed daily returns and VIX CSV files by their Date column."""
    returns = pd.read_csv(returns_path, index_col="Date", parse_dates=True)
    vix = pd.read_csv(vix_path, index_col="Date", parse_dates=True)["VIX"]
    return returns, vix
