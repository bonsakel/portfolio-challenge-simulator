"""Train all agents, run every strategy/event backtest, and save reports."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.metrics import period_metrics, recovery_metrics
from src.data.download import ROOT, load_config
from src.environment.portfolio_env import PortfolioEnv
from src.experiments.train_agents import Algorithm, train_agent
from src.strategies.traditional import (
    equal_weight_policy,
    make_mean_variance_policy,
    make_minimum_variance_policy,
)
from src.visualization.plots import plot_all_drawdowns, plot_event_values


STRATEGIES = (
    "equal_weight",
    "minimum_variance",
    "mean_variance",
    "PPO",
    "DDPG",
)


def _environment(
    returns: pd.DataFrame,
    vix: pd.Series,
    event: dict[str, Any],
    config: dict[str, Any],
) -> PortfolioEnv:
    portfolio = config["portfolio"]
    return PortfolioEnv(
        returns,
        vix,
        start=event["downturn_test"][0],
        end=event["recovery_observation"][1],
        episode_length=None,
        initial_capital=float(portfolio["initial_capital"]),
        transaction_cost_rate=float(portfolio["transaction_cost_rate"]),
        max_weight=float(portfolio["max_weight"]),
    )


def evaluate_agent(
    *,
    algorithm: Algorithm,
    event_name: str,
    event: dict[str, Any],
    config: dict[str, Any],
    returns: pd.DataFrame,
    vix: pd.Series,
    model_dir: Path,
) -> BacktestResult:
    """Evaluate the selected, frozen checkpoint through test and recovery."""
    checkpoint_dir = model_dir / "selected"
    model_path = checkpoint_dir / "best_model.zip"
    normalization_path = checkpoint_dir / "best_vecnormalize.pkl"
    if not model_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError(
            f"Missing validation-selected {algorithm} model for {event_name}: "
            f"{checkpoint_dir}"
        )
    env = _environment(returns, vix, event, config)
    vector_env = DummyVecEnv([lambda: Monitor(env)])
    normalized_env = VecNormalize.load(str(normalization_path), vector_env)
    normalized_env.training = False
    normalized_env.norm_reward = False
    model_class = PPO if algorithm == "PPO" else DDPG
    model = model_class.load(str(model_path), env=normalized_env, device="cpu")
    observation = normalized_env.reset()
    records: list[dict[str, Any]] = []
    try:
        for _ in range(len(returns)):
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = normalized_env.step(action)
            info = infos[0]
            row: dict[str, Any] = {
                "Date": pd.Timestamp(info["date"]),
                "portfolio_value": float(info["portfolio_value"]),
                "gross_return": float(info["gross_return"]),
                "net_return": float(info["net_return"]),
                "log_reward": float(rewards[0]),
                "turnover": float(info["turnover"]),
                "traded_value": float(info["traded_value"]),
                "transaction_cost": float(info["transaction_cost"]),
                "rebalance": float(info["turnover"]) > 1e-12,
            }
            row.update(
                {
                    f"weight_{asset}": float(info["target_weights"][asset])
                    for asset in env.assets
                }
            )
            row.update(
                {
                    f"ending_weight_{asset}": float(info["weights"][asset])
                    for asset in env.assets
                }
            )
            records.append(row)
            if bool(dones[0]):
                break
    finally:
        normalized_env.close()

    daily = pd.DataFrame(records).set_index("Date").sort_index()
    if daily.empty:
        raise RuntimeError(f"{algorithm} produced no evaluation steps for {event_name}")
    expected_dates = returns.loc[event["downturn_test"][0] : event["recovery_observation"][1]].index
    if not daily.index.equals(expected_dates):
        raise RuntimeError(
            f"{algorithm} evaluation dates do not match configured dates for "
            f"{event_name}: expected {len(expected_dates)}, got {len(daily)}"
        )
    return BacktestResult(
        daily=daily,
        initial_value=float(config["portfolio"]["initial_capital"]),
        final_value=float(daily["portfolio_value"].iloc[-1]),
        total_transaction_costs=float(daily["transaction_cost"].sum()),
    )


def _run_traditional(
    strategy: str,
    *,
    event: dict[str, Any],
    config: dict[str, Any],
    returns: pd.DataFrame,
) -> BacktestResult:
    portfolio = config["portfolio"]
    settings = config["traditional_strategies"]
    if strategy == "equal_weight":
        policy = equal_weight_policy
        strategy_settings = settings["equal_weight"]
        turnover_threshold = 0.0
    elif strategy == "minimum_variance":
        strategy_settings = settings["minimum_variance"]
        policy = make_minimum_variance_policy(
            returns.columns,
            lookback_trading_days=int(strategy_settings["lookback_trading_days"]),
            max_weight=float(strategy_settings["max_weight"]),
        )
        turnover_threshold = float(strategy_settings["turnover_threshold"])
    elif strategy == "mean_variance":
        strategy_settings = settings["mean_variance"]
        policy = make_mean_variance_policy(
            returns.columns,
            lookback_trading_days=int(strategy_settings["lookback_trading_days"]),
            max_weight=float(strategy_settings["max_weight"]),
            risk_free_rate_daily=float(strategy_settings["risk_free_rate_daily"]),
        )
        turnover_threshold = float(strategy_settings["turnover_threshold"])
    else:
        raise ValueError(f"Unknown traditional strategy: {strategy}")
    return run_backtest(
        returns,
        policy,
        start=event["downturn_test"][0],
        end=event["recovery_observation"][1],
        initial_value=float(portfolio["initial_capital"]),
        transaction_cost_rate=float(portfolio["transaction_cost_rate"]),
        max_weight=float(portfolio["max_weight"]),
        rebalance=strategy_settings["rebalance"],
        turnover_threshold=turnover_threshold,
    )


def _write_findings(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Experiment Results Summary",
        "",
        "This report summarizes one seeded PPO/DDPG training run per event and",
        "the three configured traditional strategies. It is descriptive, not",
        "a claim of statistical significance or evidence that one strategy is",
        "generally superior.",
        "",
    ]
    for event_name, event_rows in summary.groupby("event", sort=False):
        downturn = event_rows[event_rows["section"] == "downturn"]
        recovery = event_rows[event_rows["section"] == "recovery"]
        best_sharpe = downturn.loc[downturn["sharpe_ratio"].idxmax()]
        best_total = downturn.loc[downturn["total_return"].idxmax()]
        lines.extend(
            [
                f"## {event_name}",
                "",
                f"- Highest downturn total return: **{best_total['strategy']}** "
                f"({best_total['total_return']:.2%}).",
                f"- Highest downturn Sharpe ratio: **{best_sharpe['strategy']}** "
                f"({best_sharpe['sharpe_ratio']:.3f}).",
                "- Recovered within 24 months: "
                + ", ".join(
                    f"{row.strategy} ({'yes' if row.recovered_within_24_months else 'no'})"
                    for row in recovery.itertuples()
                )
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Results reflect the specified historical windows, one random seed,",
            "one validation episode per checkpoint, the chosen transaction-cost",
            "model, and a zero daily risk-free-rate assumption. Interpret strategy",
            "rankings accordingly; PPO and DDPG performance may vary with seeds",
            "and hyperparameters. No result should be generalized beyond the",
            "tested assets and periods without additional robustness analysis.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    config: dict[str, Any],
    returns: pd.DataFrame,
    vix: pd.Series,
    root: Path = ROOT,
    train: bool = True,
    total_timesteps: int | None = None,
    eval_freq: int | None = None,
) -> pd.DataFrame:
    """Run all five strategies for all events and save reproducible outputs."""
    assets = list(config["assets"])
    if list(returns.columns) != assets:
        raise ValueError("Daily-return columns do not match configured asset order")
    aligned_vix = vix.reindex(returns.index)
    if aligned_vix.isna().any():
        raise ValueError("Daily returns and VIX must be aligned on every return date")
    if config["traditional_strategies"]["mean_variance"]["risk_free_rate_daily"] is None:
        raise ValueError("MVO daily risk-free rate must be configured")

    results_by_event: dict[str, dict[str, BacktestResult]] = {}
    metric_rows: list[dict[str, Any]] = []
    model_root = root / "results/models"
    table_dir = root / "results/tables"
    figure_dir = root / "results/figures"
    model_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for event_name, event in config["events"].items():
        event_results: dict[str, BacktestResult] = {}
        for algorithm in config["reinforcement_learning"]["algorithms"]:
            model_dir = model_root / event_name / algorithm.lower()
            selected_model = model_dir / "selected/best_model.zip"
            if train and not selected_model.exists():
                if model_dir.exists() and any(model_dir.iterdir()):
                    raise FileExistsError(
                        f"Partial RL output exists for {event_name}/{algorithm}; "
                        f"inspect or archive {model_dir} before rerunning."
                    )
                train_agent(
                    algorithm=algorithm,
                    event_name=event_name,
                    config=config,
                    returns=returns,
                    vix=vix,
                    output_dir=model_dir,
                    total_timesteps=total_timesteps,
                    eval_freq=eval_freq,
                )
            elif not selected_model.exists():
                raise FileNotFoundError(
                    f"Selected {algorithm} model missing for {event_name}: "
                    f"{selected_model}"
                )
            event_results[algorithm] = evaluate_agent(
                algorithm=algorithm,
                event_name=event_name,
                event=event,
                config=config,
                returns=returns,
                vix=vix,
                model_dir=model_dir,
            )

        for strategy in STRATEGIES[:3]:
            event_results[strategy] = _run_traditional(
                strategy,
                event=event,
                config=config,
                returns=returns,
            )

        event_results = {strategy: event_results[strategy] for strategy in STRATEGIES}
        results_by_event[event_name] = event_results
        test_start, test_end = event["downturn_test"]
        recovery_start, recovery_end = event["recovery_observation"]
        expected_dates = returns.loc[test_start:recovery_end].index
        for strategy, result in event_results.items():
            if not result.daily.index.equals(expected_dates):
                raise RuntimeError(
                    f"{strategy} dates mismatch for {event_name}: "
                    f"expected {len(expected_dates)}, got {len(result.daily)}"
                )
            result.daily.to_csv(
                table_dir / f"daily_{event_name}_{strategy.lower()}.csv",
                index_label="Date",
            )
            risk_free = float(
                config["traditional_strategies"]["mean_variance"][
                    "risk_free_rate_daily"
                ]
            )
            downturn_metrics = period_metrics(
                result,
                start=test_start,
                end=test_end,
                risk_free_rate_daily=risk_free,
            )
            recovery_summary = recovery_metrics(
                result,
                downturn_start=test_start,
                downturn_end=test_end,
                recovery_start=recovery_start,
                recovery_end=recovery_end,
            )
            metric_rows.append(
                {
                    "event": event_name,
                    "strategy": strategy,
                    "section": "downturn",
                    **downturn_metrics,
                }
            )
            metric_rows.append(
                {
                    "event": event_name,
                    "strategy": strategy,
                    "section": "recovery",
                    **recovery_summary,
                    "transaction_costs": float(
                        result.daily.loc[recovery_start:recovery_end][
                            "transaction_cost"
                        ].sum()
                    ),
                }
            )
        plot_event_values(
            event_results,
            event_name=event_name,
            output_path=figure_dir / f"portfolio_values_{event_name}.png",
        )

    summary = pd.DataFrame(metric_rows)
    summary_path = table_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    summary.loc[summary["section"] == "recovery"].to_csv(
        table_dir / "recovery_metrics.csv", index=False
    )
    plot_all_drawdowns(
        results_by_event,
        output_path=figure_dir / "drawdowns_all_events.png",
    )
    shutil.copy2(root / "configs/experiment.yaml", root / "results/experiment.yaml")
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": list(config["events"]),
        "strategies": list(STRATEGIES),
        "seed": config["reproducibility"]["random_seed"],
        "data_manifest": "data/processed/metadata.json",
        "package_versions": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "PyPortfolioOpt": version("PyPortfolioOpt"),
            "gymnasium": version("gymnasium"),
            "stable-baselines3": version("stable-baselines3"),
            "matplotlib": version("matplotlib"),
        },
        "risk_free_rate_daily": config["traditional_strategies"][
            "mean_variance"
        ]["risk_free_rate_daily"],
        "outputs": {
            "summary_metrics": "results/tables/summary_metrics.csv",
            "recovery_metrics": "results/tables/recovery_metrics.csv",
            "daily_strategy_results": "results/tables/daily_<event>_<strategy>.csv",
            "plots": "results/figures/",
        },
    }
    with (root / "results/run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    _write_findings(summary, root / "reports/experiment_summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Evaluate existing validation-selected RL models without training",
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    returns = pd.read_csv(args.returns, index_col="Date", parse_dates=True)
    vix = pd.read_csv(args.vix, index_col="Date", parse_dates=True)["VIX"]
    summary = run_experiment(
        config=config,
        returns=returns,
        vix=vix,
        train=not args.skip_training,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
    )
    print(summary.to_string(index=False))
    print(f"\nWrote summary to {ROOT / 'results/tables/summary_metrics.csv'}")


if __name__ == "__main__":
    main()
