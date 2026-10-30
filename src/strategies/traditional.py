"""Equal-weight and rolling mean-variance portfolio policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from scipy.optimize import minimize

from src.backtest.engine import WeightPolicy, equal_weights


Objective = Literal["maximum_sharpe", "minimum_variance"]


def _maximize_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate_daily: float,
    max_weight: float,
    decision_date: pd.Timestamp,
) -> pd.Series:
    assets = list(expected_returns.index)
    means = expected_returns.to_numpy(dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    excess_returns = means - risk_free_rate_daily
    if max_weight >= 1.0 / len(assets):
        equal_start = np.full(len(assets), 1.0 / len(assets))
    else:
        equal_start = np.empty(len(assets), dtype=float)
        remaining = 1.0
        for index in range(len(assets)):
            equal_start[index] = min(
                max_weight, remaining / (len(assets) - index)
            )
            remaining -= equal_start[index]

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ covariance_values @ weights)
        if variance <= 0:
            return np.finfo(float).max
        return -float(excess_returns @ weights) / np.sqrt(variance)

    def gradient(weights: np.ndarray) -> np.ndarray:
        variance = float(weights @ covariance_values @ weights)
        if variance <= 0:
            raise ArithmeticError(
                f"Portfolio variance is not positive on {decision_date.date()}"
            )
        volatility = np.sqrt(variance)
        excess_return = float(excess_returns @ weights)
        return (
            -excess_returns / volatility
            + excess_return * (covariance_values @ weights) / variance**1.5
        )

    starts = [equal_start]
    for offset in range(len(assets)):
        candidate = np.zeros(len(assets), dtype=float)
        remaining = 1.0
        for step in range(len(assets)):
            index = (offset + step) % len(assets)
            candidate[index] = min(max_weight, remaining)
            remaining -= candidate[index]
            if remaining <= 1e-12:
                break
        if not any(np.allclose(candidate, start) for start in starts):
            starts.append(candidate)

    solutions = []
    for start in starts:
        solution = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * len(assets),
            constraints={
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
                "jac": lambda weights: np.ones_like(weights),
            },
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if solution.success:
            solutions.append(solution)
    if not solutions:
        raise RuntimeError(
            f"Maximum-Sharpe optimization failed on {decision_date.date()}: "
            f"{solution.message}"
        )
    solution = min(solutions, key=lambda candidate: candidate.fun)
    return pd.Series(solution.x, index=assets, dtype=float)


def equal_weight_policy(
    decision_date: pd.Timestamp,
    history: pd.DataFrame,
    current_weights: pd.Series,
) -> pd.Series:
    """Target equal weights; use with a monthly backtest schedule."""
    return equal_weights(current_weights.index)


def _validate_lookback(
    history: pd.DataFrame,
    assets: Sequence[str],
    lookback_trading_days: int,
) -> pd.DataFrame:
    if lookback_trading_days <= 1:
        raise ValueError("Lookback must contain at least two trading days")
    if history.columns.has_duplicates or set(history.columns) != set(assets):
        raise ValueError("Historical returns must contain each configured asset once")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("Historical returns must have a DatetimeIndex")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("Historical returns must be unique and chronological")
    if not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError("Historical returns must not contain missing or infinite values")
    if len(history) < lookback_trading_days:
        raise ValueError(
            f"Need {lookback_trading_days} historical returns; found {len(history)}"
        )
    return history.reindex(columns=assets).iloc[-lookback_trading_days:]


def make_rolling_optimizer_policy(
    assets: Sequence[str],
    *,
    objective: Objective,
    lookback_trading_days: int = 252,
    max_weight: float = 0.30,
    risk_free_rate_daily: float | None = None,
) -> WeightPolicy:
    """Create a policy using only the preceding fixed-length return window.

    The risk-free rate is expressed as a daily decimal return and is required
    for maximum-Sharpe optimization. Minimum-variance optimization does not
    use it.
    """
    ordered_assets = tuple(assets)
    if not ordered_assets or len(set(ordered_assets)) != len(ordered_assets):
        raise ValueError("Assets must be a non-empty sequence of unique names")
    if objective not in {"maximum_sharpe", "minimum_variance"}:
        raise ValueError(f"Unsupported portfolio objective: {objective}")
    if not np.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
        raise ValueError("Maximum weight must be in (0, 1]")
    if max_weight * len(ordered_assets) < 1.0 - 1e-12:
        raise ValueError("Maximum weight is too low to construct a fully invested portfolio")
    if objective == "maximum_sharpe":
        if risk_free_rate_daily is None:
            raise ValueError(
                "Set risk_free_rate_daily explicitly before creating an "
                "MVO policy; the project has not selected this assumption."
            )
        if not np.isfinite(risk_free_rate_daily):
            raise ValueError("Daily risk-free rate must be finite")

    def policy(
        decision_date: pd.Timestamp,
        history: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        if not isinstance(decision_date, pd.Timestamp):
            decision_date = pd.Timestamp(decision_date)
        if not history.empty and history.index.max() >= decision_date:
            raise ValueError("Optimizer history must be strictly before decision date")
        window = _validate_lookback(history, ordered_assets, lookback_trading_days)
        expected_returns = window.mean()
        covariance = window.cov()
        if objective == "maximum_sharpe":
            return _maximize_sharpe(
                expected_returns,
                covariance,
                float(risk_free_rate_daily),
                max_weight,
                decision_date,
            )
        optimizer = EfficientFrontier(
            None,
            covariance,
            weight_bounds=(0.0, max_weight),
            solver="CLARABEL",
            solver_options={
                "tol_gap_abs": 1e-10,
                "tol_feas": 1e-10,
                "tol_gap_rel": 1e-10,
            },
        )
        optimizer.min_volatility()
        weights = optimizer.weights
        if weights is None:
            raise RuntimeError(
                f"PyPortfolioOpt returned no weights for {objective} on "
                f"{decision_date.date()}"
            )
        return pd.Series(weights, index=ordered_assets, dtype=float)

    return policy


def make_mean_variance_policy(
    assets: Sequence[str],
    *,
    lookback_trading_days: int = 252,
    max_weight: float = 0.30,
    risk_free_rate_daily: float | None = None,
) -> WeightPolicy:
    """Create a rolling maximum-Sharpe policy."""
    return make_rolling_optimizer_policy(
        assets,
        objective="maximum_sharpe",
        lookback_trading_days=lookback_trading_days,
        max_weight=max_weight,
        risk_free_rate_daily=risk_free_rate_daily,
    )


def make_minimum_variance_policy(
    assets: Sequence[str],
    *,
    lookback_trading_days: int = 252,
    max_weight: float = 0.30,
) -> WeightPolicy:
    """Create a rolling minimum-variance policy."""
    return make_rolling_optimizer_policy(
        assets,
        objective="minimum_variance",
        lookback_trading_days=lookback_trading_days,
        max_weight=max_weight,
    )
