"""Portfolio accounting for chronological, cost-aware backtests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


WeightPolicy = Callable[
    [pd.Timestamp, pd.DataFrame, pd.Series],
    pd.Series,
]


@dataclass(frozen=True)
class BacktestResult:
    """Daily portfolio history and cumulative transaction-cost accounting."""

    daily: pd.DataFrame
    initial_value: float
    final_value: float
    total_transaction_costs: float


def equal_weights(assets: Sequence[str]) -> pd.Series:
    """Return a fully invested equal-weight portfolio."""
    if len(assets) == 0 or len(set(assets)) != len(assets):
        raise ValueError("Assets must be a non-empty sequence of unique names")
    return pd.Series(1.0 / len(assets), index=list(assets), dtype=float)


def validate_weights(
    weights: pd.Series,
    assets: Sequence[str],
    *,
    max_weight: float = 1.0,
    fully_invested: bool = True,
    tolerance: float = 1e-8,
) -> pd.Series:
    """Validate and order portfolio weights without silently repairing them."""
    if not isinstance(weights, pd.Series):
        raise TypeError("A weight policy must return a pandas Series")
    if weights.index.has_duplicates or set(weights.index) != set(assets):
        raise ValueError("Weights must contain each portfolio asset exactly once")
    ordered = weights.reindex(assets).astype(float)
    if not np.isfinite(ordered.to_numpy()).all():
        raise ValueError("Portfolio weights must be finite")
    if (ordered < -tolerance).any():
        raise ValueError("Portfolio weights must be nonnegative")
    if (ordered > max_weight + tolerance).any():
        raise ValueError(f"Portfolio weights must not exceed {max_weight}")
    if fully_invested and not np.isclose(
        ordered.sum(), 1.0, atol=tolerance, rtol=0.0
    ):
        raise ValueError("Fully invested portfolio weights must sum to 1")
    if not fully_invested and ordered.sum() > 1.0 + tolerance:
        raise ValueError("Portfolio weights must not sum to more than 1")
    return ordered


def _validate_returns(returns: pd.DataFrame) -> list[str]:
    if returns.empty:
        raise ValueError("Daily returns must not be empty")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Daily returns must have a DatetimeIndex")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Daily return dates must be unique and chronological")
    if returns.columns.has_duplicates or len(returns.columns) == 0:
        raise ValueError("Daily returns must have unique asset columns")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Daily returns must not contain missing or infinite values")
    if (values < -1.0).any():
        raise ValueError("An asset return cannot be less than -100%")
    return list(returns.columns)


def run_backtest(
    daily_returns: pd.DataFrame,
    weight_policy: WeightPolicy,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    initial_value: float = 100_000.0,
    initial_weights: pd.Series | None = None,
    transaction_cost_rate: float = 0.001,
    max_weight: float = 1.0,
    fully_invested: bool = True,
    rebalance: str = "daily",
    turnover_threshold: float = 0.0,
) -> BacktestResult:
    """Run a cost-aware backtest using only returns before each decision.

    On each date, the policy sees returns strictly before that date and the
    holdings immediately before any trade. Rebalance trades incur a fee equal
    to `transaction_cost_rate * pre-trade portfolio value * sum(abs(weight
    changes))`. The fee is deducted once before the target portfolio earns
    that date's return. The reported net return and log reward both include
    the fee; callers must not deduct it a second time.
    """
    assets = _validate_returns(daily_returns)
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Initial portfolio value must be finite and positive")
    if not np.isfinite(transaction_cost_rate) or not 0 <= transaction_cost_rate < 1:
        raise ValueError("Transaction cost rate must be in [0, 1)")
    if not np.isfinite(turnover_threshold) or turnover_threshold < 0:
        raise ValueError("Turnover threshold must be finite and nonnegative")
    if rebalance not in {"daily", "monthly"}:
        raise ValueError("Rebalance frequency must be 'daily' or 'monthly'")
    if not np.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
        raise ValueError("Maximum weight must be in (0, 1]")

    holdings = validate_weights(
        initial_weights if initial_weights is not None else equal_weights(assets),
        assets,
        max_weight=max_weight,
        fully_invested=fully_invested,
    )
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("Backtest start must not be after end")
    test_returns = daily_returns.loc[
        (daily_returns.index >= start_date) & (daily_returns.index <= end_date)
    ]
    if test_returns.empty:
        raise ValueError("No daily returns fall within the requested backtest period")

    value = float(initial_value)
    last_rebalance_month: tuple[int, int] | None = None
    records: list[dict[str, float | bool]] = []
    for day in test_returns.index:
        current_month = (day.year, day.month)
        scheduled = rebalance == "daily" or current_month != last_rebalance_month
        if scheduled:
            last_rebalance_month = current_month
            target = validate_weights(
                weight_policy(
                    day,
                    daily_returns.loc[daily_returns.index < day],
                    holdings.copy(),
                ),
                assets,
                max_weight=max_weight,
                fully_invested=fully_invested,
            )
            turnover = float((target - holdings).abs().sum())
            traded_value = value * turnover
            transaction_cost = transaction_cost_rate * traded_value
            holdings_within_constraints = (
                (holdings >= -1e-8).all()
                and (holdings <= max_weight + 1e-8).all()
                and (
                    np.isclose(holdings.sum(), 1.0, atol=1e-8, rtol=0.0)
                    if fully_invested
                    else holdings.sum() <= 1.0 + 1e-8
                )
            )
            if (
                turnover + 1e-12 < turnover_threshold
                and holdings_within_constraints
            ):
                target = holdings.copy()
                turnover = 0.0
                traded_value = 0.0
                transaction_cost = 0.0
                did_rebalance = False
            else:
                did_rebalance = turnover > 1e-12
        else:
            target = holdings.copy()
            turnover = 0.0
            traded_value = 0.0
            transaction_cost = 0.0
            did_rebalance = False

        pre_trade_value = value
        value_after_cost = pre_trade_value - transaction_cost
        asset_returns = test_returns.loc[day]
        gross_return = float(target.dot(asset_returns))
        value = value_after_cost * (1.0 + gross_return)
        net_return = value / pre_trade_value - 1.0
        if not np.isfinite(value) or value <= 0 or net_return <= -1.0:
            raise ArithmeticError(f"Portfolio value became invalid on {day.date()}")

        ending_holdings = target * (1.0 + asset_returns) / (1.0 + gross_return)
        record: dict[str, float | bool] = {
            "portfolio_value": value,
            "gross_return": gross_return,
            "net_return": net_return,
            "log_reward": float(np.log1p(net_return)),
            "turnover": turnover,
            "traded_value": traded_value,
            "transaction_cost": transaction_cost,
            "rebalance": did_rebalance,
        }
        record.update({f"weight_{asset}": float(target[asset]) for asset in assets})
        record.update(
            {
                f"ending_weight_{asset}": float(ending_holdings[asset])
                for asset in assets
            }
        )
        records.append(record)
        holdings = ending_holdings

    daily = pd.DataFrame(records, index=test_returns.index)
    daily.index.name = "Date"
    return BacktestResult(
        daily=daily,
        initial_value=float(initial_value),
        final_value=value,
        total_transaction_costs=float(daily["transaction_cost"].sum()),
    )
