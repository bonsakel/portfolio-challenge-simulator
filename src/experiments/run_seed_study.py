"""Run a multi-seed PPO/DDPG robustness study without replacing baseline artifacts."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.metrics import period_metrics, recovery_metrics
from src.data.download import ROOT, load_config
from src.experiments.run_experiment import STRATEGIES, _run_traditional, evaluate_agent
from src.experiments.train_agents import train_agent


DEFAULT_SEED_COUNT = 5


def _result_rows(
    *,
    event_name: str,
    strategy: str,
    seed: int | str,
    daily: pd.DataFrame,
    event: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    from src.backtest.engine import BacktestResult

    result = BacktestResult(
        daily=daily,
        initial_value=float(config["portfolio"]["initial_capital"]),
        final_value=float(daily["portfolio_value"].iloc[-1]),
        total_transaction_costs=float(daily["transaction_cost"].sum()),
    )
    risk_free = float(
        config["traditional_strategies"]["mean_variance"]["risk_free_rate_daily"]
    )
    test_start, test_end = event["downturn_test"]
    recovery_start, recovery_end = event["recovery_observation"]
    downturn = period_metrics(
        result,
        start=test_start,
        end=test_end,
        risk_free_rate_daily=risk_free,
    )
    recovery = recovery_metrics(
        result,
        downturn_start=test_start,
        downturn_end=test_end,
        recovery_start=recovery_start,
        recovery_end=recovery_end,
    )
    return [
        {
            "event": event_name,
            "strategy": strategy.lower(),
            "seed": seed,
            "section": "downturn",
            **downturn,
        },
        {
            "event": event_name,
            "strategy": strategy.lower(),
            "seed": seed,
            "section": "recovery",
            **recovery,
        },
    ]


def run_seed_study(
    *,
    config: dict[str, Any],
    returns: pd.DataFrame,
    vix: pd.Series,
    root: Path = ROOT,
    seeds: list[int] | None = None,
    total_timesteps: int | None = None,
    eval_freq: int | None = None,
) -> pd.DataFrame:
    """Train and evaluate independent model seeds, retaining all original results."""
    seeds = seeds or list(
        range(
            int(config["reproducibility"]["random_seed"]),
            int(config["reproducibility"]["random_seed"]) + DEFAULT_SEED_COUNT,
        )
    )
    if len(seeds) != DEFAULT_SEED_COUNT or len(set(seeds)) != len(seeds):
        raise ValueError(f"Provide exactly {DEFAULT_SEED_COUNT} distinct random seeds")
    if list(returns.columns) != list(config["assets"]):
        raise ValueError("Daily-return columns do not match configured asset order")
    aligned_vix = vix.reindex(returns.index)
    if aligned_vix.isna().any():
        raise ValueError("Daily returns and VIX must be aligned on every return date")

    study_dir = root / "results" / "robustness"
    model_root = study_dir / "models"
    table_dir = study_dir / "tables"
    manifest_path = study_dir / "study.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"A completed seed study already exists at {study_dir}; preserve it or choose another output root."
        )
    table_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    events = config["events"]

    for event_name, event in events.items():
        for algorithm in config["reinforcement_learning"]["algorithms"]:
            for seed in seeds:
                model_dir = model_root / event_name / algorithm.lower() / f"seed_{seed}"
                metadata_path = model_dir / "metadata.json"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("seed") != seed:
                        raise ValueError(f"Stored seed metadata does not match {model_dir}")
                else:
                    if model_dir.exists() and any(model_dir.iterdir()):
                        raise FileExistsError(
                            f"Incomplete model output exists at {model_dir}; inspect it before rerunning."
                        )
                    print(
                        f"Training {algorithm} / {event_name} / seed {seed}",
                        flush=True,
                    )
                    train_config = deepcopy(config)
                    train_config["reproducibility"]["random_seed"] = seed
                    train_agent(
                        algorithm=algorithm,
                        event_name=event_name,
                        config=train_config,
                        returns=returns,
                        vix=vix,
                        output_dir=model_dir,
                        seed=seed,
                        total_timesteps=total_timesteps,
                        eval_freq=eval_freq,
                    )
                result = evaluate_agent(
                    algorithm=algorithm,
                    event_name=event_name,
                    event=event,
                    config=config,
                    returns=returns,
                    vix=vix,
                    model_dir=model_dir,
                )
                strategy = algorithm.lower()
                daily_path = table_dir / (
                    f"daily_{event_name}_{strategy}_seed_{seed}.csv"
                )
                result.daily.to_csv(daily_path, index_label="Date")
                metric_rows.extend(
                    _result_rows(
                        event_name=event_name,
                        strategy=strategy,
                        seed=seed,
                        daily=result.daily,
                        event=event,
                        config=config,
                    )
                )

        for strategy in STRATEGIES[:3]:
            result = _run_traditional(
                strategy,
                event=event,
                config=config,
                returns=returns,
            )
            result.daily.to_csv(
                table_dir / f"daily_{event_name}_{strategy.lower()}.csv",
                index_label="Date",
            )
            metric_rows.extend(
                _result_rows(
                    event_name=event_name,
                    strategy=strategy,
                    seed="baseline",
                    daily=result.daily,
                    event=event,
                    config=config,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(table_dir / "metrics_by_seed.csv", index=False)
    manifest = {
        "complete": True,
        "seeds": seeds,
        "seed_count": len(seeds),
        "events": list(events),
        "algorithms": config["reinforcement_learning"]["algorithms"],
        "timesteps_per_model": total_timesteps
        or max(
            int(config["reinforcement_learning"]["training"][name.lower()]["total_timesteps"])
            for name in config["reinforcement_learning"]["algorithms"]
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metrics


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
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    args = parser.parse_args()
    if args.seed_count != DEFAULT_SEED_COUNT:
        parser.error(f"--seed-count must be {DEFAULT_SEED_COUNT}")

    config = load_config(args.config)
    returns = pd.read_csv(args.returns, index_col="Date", parse_dates=True)
    vix = pd.read_csv(args.vix, index_col="Date", parse_dates=True)["VIX"]
    metrics = run_seed_study(
        config=config,
        returns=returns,
        vix=vix,
        seeds=list(
            range(
                int(config["reproducibility"]["random_seed"]),
                int(config["reproducibility"]["random_seed"]) + args.seed_count,
            )
        ),
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
    )
    print(
        f"Completed {len(metrics[metrics['section'] == 'downturn'])} seed-level "
        "downturn evaluations; original results were not changed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
