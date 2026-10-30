"""Pure backtest orchestration over the research engine in `src/` (which is reused unchanged).

Accounting (fee on pre-trade value, fee deducted once before the day's return, decisions use
only returns strictly before the decision date) lives in `src.backtest.engine`; this module
only prepares inputs, validates the window and collects results.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import TRADING_DAYS_PER_YEAR, period_metrics  # noqa: E402
from src.strategies.traditional import (  # noqa: E402
    equal_weight_policy,
    make_mean_variance_policy,
    make_minimum_variance_policy,
)

STRATEGY_LABELS = {
    "equal_weight": "Equal Weight",
    "mean_variance": "Mean Variance (max Sharpe)",
    "minimum_variance": "Minimum Variance",
    "ppo": "PPO",
    "ddpg": "DDPG",
}
# Research defaults: (rebalance frequency, turnover threshold).
STRATEGY_DEFAULTS = {
    "equal_weight": ("monthly", 0.0),
    "mean_variance": ("daily", 0.02),
    "minimum_variance": ("daily", 0.02),
    "ppo": ("daily", 0.0),
    "ddpg": ("daily", 0.0),
}
OPTIMIZERS = {"mean_variance", "minimum_variance"}
RL_STRATEGIES = {"ppo", "ddpg"}

# (strategy, progress callback) -> (result, model description). Supplied by the job layer,
# which owns the model registry, the compatibility check and the VIX data.
AgentRunner = Callable[[str, "Callable[[float], None] | None"], tuple[Any, dict[str, Any]]]


class BacktestError(ValueError):
    """A configuration or data problem the user can act on."""


@dataclass(frozen=True)
class RunConfig:
    strategies: tuple[str, ...]
    start: date
    end: date
    initial_capital: float
    max_weight: float
    lookback_days: int = 252
    transaction_cost_rate: float = 0.001
    rebalance: str | None = None  # None -> per-strategy research default
    turnover_threshold: float | None = None  # None -> per-strategy research default
    risk_free_rate_annual: float = 0.0
    initial_weights: dict[str, float] | None = None  # None -> equal weight, as in the research

    @property
    def risk_free_rate_daily(self) -> float:
        return self.risk_free_rate_annual / TRADING_DAYS_PER_YEAR

    def resolved(self, strategy: str) -> tuple[str, float]:
        frequency, threshold = STRATEGY_DEFAULTS[strategy]
        return (
            self.rebalance or frequency,
            self.turnover_threshold if self.turnover_threshold is not None else threshold,
        )


@dataclass
class StrategyRun:
    key: str
    label: str
    rebalance: str
    turnover_threshold: float
    summary: dict[str, Any]
    series: dict[str, Any]
    model: dict[str, Any] | None = None  # RL strategies: which trained model produced this run


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns between consecutive common dates. No filling: a date missing for any
    asset was already removed, so a return across such a gap spans the gap."""
    if prices.empty or len(prices) < 2:
        raise BacktestError("Not enough common price history to compute returns.")
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise BacktestError("Prices must be positive and finite.")
    return prices.pct_change().iloc[1:]


def check_window(returns: pd.DataFrame, config: RunConfig) -> pd.Timestamp:
    """Validate the date window and the warm-up history; returns the first decision date."""
    if config.start >= config.end:
        raise BacktestError("Start date must be before the end date.")
    dates = returns.index
    in_window = dates[(dates >= pd.Timestamp(config.start)) & (dates <= pd.Timestamp(config.end))]
    if len(in_window) < 2:
        raise BacktestError(
            f"Fewer than 2 trading days of common data between {config.start} and {config.end} "
            f"(common returns run {dates[0].date()} to {dates[-1].date()})."
        )
    first = in_window[0]
    if OPTIMIZERS & set(config.strategies):
        before = int((dates < first).sum())
        if before < config.lookback_days:
            earliest = dates[config.lookback_days].date() if len(dates) > config.lookback_days else None
            hint = (
                f" Start on or after {earliest}, or reduce the lookback."
                if earliest
                else " The common history is shorter than the lookback."
            )
            raise BacktestError(
                f"Optimization strategies need {config.lookback_days} daily returns before the first "
                f"decision on {first.date()}, but only {before} are available (common history begins "
                f"{dates[0].date()}).{hint}"
            )
    return first


def _policy(strategy: str, assets: list[str], config: RunConfig) -> Callable:
    if strategy == "equal_weight":
        return equal_weight_policy
    if strategy == "mean_variance":
        return make_mean_variance_policy(
            assets,
            lookback_trading_days=config.lookback_days,
            max_weight=config.max_weight,
            risk_free_rate_daily=config.risk_free_rate_daily,
        )
    return make_minimum_variance_policy(
        assets, lookback_trading_days=config.lookback_days, max_weight=config.max_weight
    )


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def run_strategies(
    returns: pd.DataFrame,
    config: RunConfig,
    progress: Callable[[float], None] | None = None,
    agent_runner: AgentRunner | None = None,
) -> dict[str, StrategyRun]:
    check_window(returns, config)
    assets = list(returns.columns)
    start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
    span = max((end - start).days, 1)
    initial = pd.Series(config.initial_weights, dtype=float) if config.initial_weights else None
    runs: dict[str, StrategyRun] = {}
    total = len(config.strategies)

    for position, strategy in enumerate(config.strategies):
        if strategy not in STRATEGY_LABELS:
            raise BacktestError(f"Unknown strategy: {strategy!r}.")
        frequency, threshold = config.resolved(strategy)
        model: dict[str, Any] | None = None

        try:
            if strategy in RL_STRATEGIES:
                if agent_runner is None:
                    raise BacktestError(f"{STRATEGY_LABELS[strategy]} needs a trained model.")
                # Agents decide every day with no threshold, as in the research.
                frequency, threshold = STRATEGY_DEFAULTS[strategy]
                result, model = agent_runner(
                    strategy, (lambda f, _p=position: progress((_p + f) / total)) if progress else None
                )
            else:
                base = _policy(strategy, assets, config)

                def policy(day: pd.Timestamp, history: pd.DataFrame, current: pd.Series,
                           _b=base, _p=position):
                    if progress:
                        within = min(max((day - start).days / span, 0.0), 1.0)
                        progress((_p + within) / total)
                    return _b(day, history, current)

                result = run_backtest(
                    returns,
                    policy,
                    start=start,
                    end=end,
                    initial_value=config.initial_capital,
                    initial_weights=initial,
                    transaction_cost_rate=config.transaction_cost_rate,
                    max_weight=config.max_weight,
                    rebalance=frequency,
                    turnover_threshold=threshold,
                )
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            if isinstance(exc, BacktestError):
                raise
            raise BacktestError(f"{STRATEGY_LABELS[strategy]} failed: {exc}") from exc

        daily = result.daily
        first_day, last_day = daily.index[0].date().isoformat(), daily.index[-1].date().isoformat()
        metrics = period_metrics(
            result, start=first_day, end=last_day, risk_free_rate_daily=config.risk_free_rate_daily
        )
        metrics.update(
            total_turnover=float(daily["turnover"].sum()),
            rebalance_count=int(daily["rebalance"].sum()),
            initial_value=result.initial_value,
        )
        series = {
            "dates": [d.date().isoformat() for d in daily.index],
            "portfolio_value": daily["portfolio_value"].tolist(),
            "net_return": daily["net_return"].tolist(),
            "turnover": daily["turnover"].tolist(),
            "transaction_cost": daily["transaction_cost"].tolist(),
            "weights": {a: daily[f"weight_{a}"].tolist() for a in assets},
        }
        runs[strategy] = StrategyRun(
            strategy, STRATEGY_LABELS[strategy], frequency, threshold, _clean(metrics), series, model
        )
    if progress:
        progress(1.0)
    return runs
