"""Comparison plots for saved daily strategy backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestResult


STRATEGY_LABELS = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "mean_variance": "Mean Variance",
    "PPO": "PPO",
    "DDPG": "DDPG",
}


def _drawdown(result: BacktestResult) -> pd.Series:
    wealth = pd.concat(
        [
            pd.Series([result.initial_value], index=[result.daily.index[0] - pd.Timedelta(days=1)]),
            result.daily["portfolio_value"],
        ]
    )
    return wealth / wealth.cummax() - 1.0


def plot_event_values(
    results: Mapping[str, BacktestResult],
    *,
    event_name: str,
    output_path: Path,
) -> None:
    if not results:
        raise ValueError("At least one strategy result is required")
    figure, axis = plt.subplots(figsize=(12, 6))
    for strategy, result in results.items():
        label = STRATEGY_LABELS.get(strategy, strategy)
        axis.plot(
            result.daily.index,
            result.daily["portfolio_value"],
            label=label,
            linewidth=1.5,
        )
    axis.set_title(f"Portfolio Value During Downturn and Recovery: {event_name}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio value (USD)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_all_drawdowns(
    results_by_event: Mapping[str, Mapping[str, BacktestResult]],
    *,
    output_path: Path,
) -> None:
    if not results_by_event:
        raise ValueError("At least one event result is required")
    figure, axes = plt.subplots(
        len(results_by_event), 1, figsize=(12, 4 * len(results_by_event)), squeeze=False
    )
    for axis, (event_name, results) in zip(axes[:, 0], results_by_event.items(), strict=True):
        for strategy, result in results.items():
            drawdown = _drawdown(result)
            axis.plot(
                drawdown.index,
                drawdown,
                label=STRATEGY_LABELS.get(strategy, strategy),
                linewidth=1.2,
            )
        axis.set_title(event_name)
        axis.set_ylabel("Drawdown")
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Date")
    axes[0, 0].legend(loc="lower left", ncol=3)
    figure.suptitle("Strategy Drawdowns During Downturn and Recovery")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
