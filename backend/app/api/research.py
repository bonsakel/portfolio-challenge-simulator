"""Read-only access to the existing CSC-475 research artifacts (never writes)."""

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.backtests.runner import BacktestError, RunConfig, run_strategies
from app.config import PROJECT_ROOT, get_settings
from app.deps import Services, get_services
from app.rl.inference import RLError, load_vix, run_agent

router = APIRouter(prefix="/research", tags=["research"])


class DownturnRow(BaseModel):
    event: str
    strategy: str
    period_start: str
    period_end: str
    trading_days: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    transaction_costs: float


class ResearchSummary(BaseModel):
    source: str
    rows: list[DownturnRow]


@router.get("/summary", response_model=ResearchSummary)
def research_summary() -> ResearchSummary:
    path = get_settings().results_dir / "tables" / "summary_metrics.csv"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Research results not found. Run `python -m src.experiments.run_experiment` first.",
        )
    frame = pd.read_csv(path)
    frame = frame[frame["section"] == "downturn"]
    rows = []
    for record in frame.to_dict("records"):
        rows.append(
            DownturnRow(
                event=record["event"],
                strategy=record["strategy"],
                period_start=record["period_start"],
                period_end=record["period_end"],
                trading_days=int(record["trading_days"]),
                total_return=record["total_return"],
                annualized_return=record["annualized_return"],
                annualized_volatility=record["annualized_volatility"],
                sharpe_ratio=record["sharpe_ratio"],
                maximum_drawdown=record["maximum_drawdown"],
                transaction_costs=record["transaction_costs"],
            )
        )
    return ResearchSummary(source=str(path.name), rows=rows)


def _dashboard_stats(rows: pd.DataFrame, column: str) -> dict[str, float | None]:
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _dashboard_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _dashboard_series(
    event: str,
    strategy: str,
    *,
    seeds: list[int],
    seed_count: int,
    results_dir: Path,
    assets: list[str],
    multi_seed: bool,
) -> dict[str, Any]:
    table_dir = results_dir / ("robustness/tables" if multi_seed else "tables")
    if multi_seed and strategy in ("ppo", "ddpg"):
        paths = [
            table_dir / f"daily_{event}_{strategy}_seed_{seed}.csv"
            for seed in seeds
        ]
    else:
        paths = [table_dir / f"daily_{event}_{strategy}.csv"]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Daily research results are missing: {', '.join(missing)}.",
        )

    frames = [pd.read_csv(path, parse_dates=["Date"]) for path in paths]
    dates = frames[0]["Date"]
    if any(not frame["Date"].equals(dates) for frame in frames[1:]):
        raise HTTPException(
            status_code=500,
            detail=f"Daily result dates do not align for {event}/{strategy}.",
        )

    values = [frame["portfolio_value"].to_numpy(dtype=float) for frame in frames]
    drawdowns = [
        value / pd.Series(value).cummax().to_numpy() - 1.0 for value in values
    ]
    costs = [
        frame["transaction_cost"].to_numpy(dtype=float).cumsum() for frame in frames
    ]
    turnovers = [frame["turnover"].to_numpy(dtype=float) for frame in frames]

    def average(items: list[Any]) -> list[float]:
        return [float(v) for v in pd.DataFrame(items).mean(axis=0).to_numpy()]

    def low(items: list[Any]) -> list[float]:
        return [float(v) for v in pd.DataFrame(items).min(axis=0).to_numpy()]

    def high(items: list[Any]) -> list[float]:
        return [float(v) for v in pd.DataFrame(items).max(axis=0).to_numpy()]

    weight_values = {
        asset: average(
            [frame[f"weight_{asset}"].to_numpy(dtype=float) for frame in frames]
        )
        for asset in assets
    }
    return {
        "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
        "sample_returns": [
            frame["net_return"].to_numpy(dtype=float).tolist() for frame in frames
        ],
        "sample_transaction_costs": [
            frame["transaction_cost"].to_numpy(dtype=float).tolist()
            for frame in frames
        ],
        "portfolio_value": average(values),
        "value_min": low(values),
        "value_max": high(values),
        "drawdown": average(drawdowns),
        "drawdown_min": low(drawdowns),
        "drawdown_max": high(drawdowns),
        "cumulative_cost": average(costs),
        "turnover": average(turnovers),
        "weights": weight_values,
        "seed_count": seed_count if multi_seed and strategy in ("ppo", "ddpg") else 1,
    }


@router.get("/dashboard")
def research_dashboard(
    events: list[str] | None = Query(default=None),
    strategies: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    """Return only selected research data, preferring a complete multi-seed study."""
    config = _read_config()
    selected_events = list(dict.fromkeys(events or EVENT_ORDER))
    selected_strategies = list(dict.fromkeys(strategies or STRATEGY_ORDER))
    invalid_events = [event for event in selected_events if event not in config["events"]]
    invalid_strategies = [
        strategy for strategy in selected_strategies if strategy not in STRATEGY_ORDER
    ]
    if invalid_events or invalid_strategies:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown event(s): {invalid_events}; "
                f"unknown strategy/strategies: {invalid_strategies}."
            ),
        )
    if not selected_events or not selected_strategies:
        raise HTTPException(
            status_code=422,
            detail="Select at least one downturn and one strategy.",
        )

    results_dir = get_settings().results_dir
    manifest_path = results_dir / "robustness" / "study.json"
    robust_metrics_path = results_dir / "robustness" / "tables" / "metrics_by_seed.csv"
    multi_seed = False
    seeds: list[int] = []
    if manifest_path.is_file() and robust_metrics_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in manifest.get("seeds", [])]
        multi_seed = bool(manifest.get("complete")) and len(seeds) >= 2
        if multi_seed:
            expected = [
                results_dir
                / "robustness"
                / "tables"
                / f"daily_{event}_{strategy}_seed_{seed}.csv"
                for event in config["events"]
                for strategy in ("ppo", "ddpg")
                for seed in seeds
            ]
            multi_seed = all(path.is_file() for path in expected)

    baseline_path = results_dir / "tables" / "summary_metrics.csv"
    if not multi_seed and not baseline_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Research results not found. Run `python -m src.experiments.run_experiment` first.",
        )
    metric_path = robust_metrics_path if multi_seed else baseline_path
    metrics_frame = pd.read_csv(metric_path)
    if "seed" not in metrics_frame:
        metrics_frame["seed"] = "baseline"

    events_result: list[dict[str, Any]] = []
    for event_name in selected_events:
        event_metrics: dict[str, Any] = {}
        for strategy in selected_strategies:
            rows = metrics_frame[
                (metrics_frame["event"] == event_name)
                & (metrics_frame["strategy"].str.lower() == strategy)
                & (metrics_frame["section"] == "downturn")
            ]
            if multi_seed and strategy in ("ppo", "ddpg"):
                rows = rows[rows["seed"] != "baseline"]
                if rows["seed"].nunique() != len(seeds):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Expected {len(seeds)} seeds for {event_name}/{strategy}.",
                    )
            else:
                rows = rows[rows["seed"] == "baseline"] if multi_seed else rows
            if rows.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"Summary metrics are missing for {event_name}/{strategy}.",
                )

            recovery_rows = metrics_frame[
                (metrics_frame["event"] == event_name)
                & (metrics_frame["strategy"].str.lower() == strategy)
                & (metrics_frame["section"] == "recovery")
            ]
            if multi_seed and strategy in ("ppo", "ddpg"):
                recovery_rows = recovery_rows[recovery_rows["seed"] != "baseline"]
            elif multi_seed:
                recovery_rows = recovery_rows[recovery_rows["seed"] == "baseline"]

            metric_fields = (
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
                "transaction_costs",
            )
            metric_summary = {
                field: _dashboard_stats(rows, field) for field in metric_fields
            }
            recovered = (
                sum(
                    _dashboard_bool(value)
                    for value in recovery_rows["recovered_within_24_months"]
                )
                if "recovered_within_24_months" in recovery_rows
                else 0
            )
            event_metrics[strategy] = {
                "metrics": metric_summary,
                "recovery": {
                    "recovery_return_from_trough": _dashboard_stats(
                        recovery_rows, "recovery_return_from_trough"
                    ),
                    "recovery_time_trading_days": _dashboard_stats(
                        recovery_rows, "recovery_time_trading_days"
                    ),
                    "ending_value": _dashboard_stats(
                        recovery_rows,
                        "recovery_ending_value"
                        if "recovery_ending_value" in recovery_rows
                        else "ending_value",
                    ),
                    "recovered_count": int(recovered),
                    "sample_count": int(len(recovery_rows)),
                },
                "series": _dashboard_series(
                    event_name,
                    strategy,
                    seeds=seeds,
                    seed_count=len(seeds),
                    results_dir=results_dir,
                    assets=config["assets"],
                    multi_seed=multi_seed,
                ),
            }
        events_result.append(
            {
                "event": event_name,
                "windows": config["events"][event_name],
                "strategies": event_metrics,
            }
        )

    return {
        "source": "five_seed_study" if multi_seed else "original_single_seed",
        "seed_count": len(seeds) if multi_seed else 1,
        "reference_capital": float(config["portfolio"]["initial_capital"]),
        "annualization_trading_days": int(
            config["backtest"]["metrics"]["annualization_trading_days"]
        ),
        "risk_free_rate_daily": float(
            config["backtest"]["metrics"]["risk_free_rate_daily"]
        ),
        "strategies": selected_strategies,
        "events": events_result,
    }


@router.get("/pipeline")
def research_pipeline() -> dict[str, Any]:
    """Report durable progress for the configured multi-seed artifact workflow."""
    config = _read_config()
    results_dir = get_settings().results_dir
    robustness_dir = results_dir / "robustness"
    model_root = robustness_dir / "models"
    table_dir = robustness_dir / "tables"
    summary_path = results_dir / "tables" / "summary_metrics.csv"
    configured_seed = int(config["reproducibility"]["random_seed"])
    seeds = list(range(configured_seed, configured_seed + 5))
    algorithms = [algorithm.lower() for algorithm in config["reinforcement_learning"]["algorithms"]]
    expected_models = len(config["events"]) * len(algorithms) * len(seeds)
    completed_models = 0
    completed_simulations = 0
    for event in config["events"]:
        for algorithm in algorithms:
            for seed in seeds:
                model_metadata = (
                    model_root / event / algorithm / f"seed_{seed}" / "metadata.json"
                )
                if model_metadata.is_file():
                    metadata = json.loads(model_metadata.read_text(encoding="utf-8"))
                    if metadata.get("event") == event and metadata.get("algorithm", "").lower() == algorithm and metadata.get("seed") == seed:
                        completed_models += 1
                if (
                    table_dir
                    / f"daily_{event}_{algorithm}_seed_{seed}.csv"
                ).is_file():
                    completed_simulations += 1

    manifest_path = robustness_dir / "study.json"
    metrics_path = table_dir / "metrics_by_seed.csv"
    robustness_ready = False
    if manifest_path.is_file() and metrics_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        robustness_ready = (
            bool(manifest.get("complete"))
            and manifest.get("seed_count") == len(seeds)
            and completed_models == expected_models
            and completed_simulations == expected_models
        )
    run_started = model_root.is_dir() and any(model_root.rglob("metadata.json"))
    if robustness_ready:
        status = "complete"
    elif run_started:
        status = "running"
    else:
        status = "not_started"

    returns_path = PROJECT_ROOT / "data" / "processed" / "daily_returns" / "daily_returns.csv"
    vix_path = PROJECT_ROOT / "data" / "processed" / "daily_returns" / "vix.csv"
    data_ready = returns_path.is_file() and vix_path.is_file()
    steps = [
        {
            "id": "data",
            "label": "Market data and periods",
            "status": "complete" if data_ready else "pending",
            "detail": "Nine ETFs and VIX aligned to the configured windows"
            if data_ready
            else "Processed returns and VIX are required",
        },
        {
            "id": "training",
            "label": "Train and validate PPO / DDPG",
            "status": "complete" if robustness_ready else "in_progress" if run_started else "pending",
            "detail": f"{completed_models}/{expected_models} models"
            if run_started
            else "Five seeds per algorithm and downturn",
        },
        {
            "id": "simulation",
            "label": "Test downturn and recovery",
            "status": "complete" if robustness_ready else "in_progress" if completed_simulations else "pending",
            "detail": f"{completed_simulations}/{expected_models} RL simulations; baseline results available"
            if summary_path.is_file()
            else f"{completed_simulations}/{expected_models} RL simulations",
        },
        {
            "id": "comparison",
            "label": "Compare strategies",
            "status": "current",
            "detail": "Review selected events, strategies, risk and recovery",
        },
    ]
    return {
        "status": status,
        "completed_models": completed_models,
        "completed_simulations": completed_simulations,
        "total_models": expected_models,
        "robustness_ready": robustness_ready,
        "baseline_available": summary_path.is_file(),
        "steps": steps,
    }


# Research Mode: a locked, read-only view of the CSC-475 experiment. Nothing here writes to
# results/, and nothing done with custom portfolios can change what it shows.

STRATEGY_ORDER = ["equal_weight", "mean_variance", "minimum_variance", "ppo", "ddpg"]
EVENT_ORDER = ["global_financial_crisis", "covid_crash", "2022_downturn"]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None  # e.g. the artifact's "not recovered within 24 months" text
    return None if math.isnan(number) else number


def _read_table() -> tuple[pd.DataFrame, str]:
    path = get_settings().results_dir / "tables" / "summary_metrics.csv"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Research results not found. Run `python -m src.experiments.run_experiment` first.",
        )
    frame = pd.read_csv(path)
    frame["strategy"] = frame["strategy"].str.lower()
    return frame, path.name


def _read_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise HTTPException(status_code=404, detail="configs/experiment.yaml not found.")
    return yaml.safe_load(CONFIG_PATH.read_text())


class RecoveryRow(BaseModel):
    event: str
    strategy: str
    trough_date: str
    trough_value: float
    recovery_return_from_trough: float
    recovered_within_24_months: bool
    recovery_time_trading_days: int | None
    recovery_date: str | None
    ending_value: float


class Finding(BaseModel):
    text: str
    evidence: str


class ResearchOverview(BaseModel):
    source: str
    locked: bool
    settings: dict[str, Any]
    events: dict[str, dict[str, list[str]]]
    downturn: list[DownturnRow]
    recovery: list[RecoveryRow]
    findings: list[Finding]
    caveats: list[str]


def _sorted(rows: list) -> list:
    def key(r):
        return (
            EVENT_ORDER.index(r.event) if r.event in EVENT_ORDER else 99,
            STRATEGY_ORDER.index(r.strategy) if r.strategy in STRATEGY_ORDER else 99,
        )

    return sorted(rows, key=key)


def _name(strategy: str) -> str:
    return strategy.upper() if strategy in ("ppo", "ddpg") else strategy.replace("_", " ")


def _findings(downturn: list[DownturnRow], recovery: list[RecoveryRow]) -> list[Finding]:
    """Observations computed from the artifacts; nothing here is a stored claim."""
    events = [e for e in EVENT_ORDER if any(r.event == e for r in downturn)]
    if not events:
        return []
    ename = lambda e: e.replace("_", " ")  # noqa: E731
    out: list[Finding] = []

    best_return = {e: max((r for r in downturn if r.event == e), key=lambda r: r.total_return) for e in events}
    best_sharpe = {e: max((r for r in downturn if r.event == e), key=lambda r: r.sharpe_ratio) for e in events}
    winners = {best_return[e].strategy for e in events}
    out.append(Finding(
        text=("No strategy had the best downturn total return in every event."
              if len(winners) > 1 else
              f"{_name(next(iter(winners)))} had the best downturn total return in every event."),
        evidence="; ".join(
            f"{ename(e)}: {_name(best_return[e].strategy)} ({best_return[e].total_return:+.1%})" for e in events),
    ))
    out.append(Finding(
        text="Best downturn Sharpe ratio by event.",
        evidence="; ".join(
            f"{ename(e)}: {_name(best_sharpe[e].strategy)} ({best_sharpe[e].sharpe_ratio:.3f})" for e in events),
    ))
    rl_best = [e for e in events if best_return[e].strategy in ("ppo", "ddpg")]
    out.append(Finding(
        text=("PPO or DDPG had the best downturn total return in " + ", ".join(ename(e) for e in rl_best) + "."
              if rl_best else
              "PPO and DDPG did not have the best downturn total return in any event."),
        evidence=f"{len(events)} events compared",
    ))
    lost = sum(1 for r in downturn if r.total_return < 0)
    out.append(Finding(
        text=("Every strategy lost money in every downturn test."
              if lost == len(downturn)
              else f"{lost} of {len(downturn)} strategy/event runs lost money in the downturn."),
        evidence=f"{lost} of {len(downturn)} downturn total returns are negative",
    ))
    for e in events:
        rows = [r for r in recovery if r.event == e]
        if not rows:
            continue
        crossed = [r for r in rows if r.recovered_within_24_months]
        above = [r for r in rows if r.ending_value >= 100_000]
        below = ", ".join(f"{_name(r.strategy)} ({r.ending_value:,.0f})" for r in rows if r.ending_value < 100_000)
        out.append(Finding(
            text=(f"{ename(e)}: {len(crossed)} of {len(rows)} strategies regained $100,000 at some point "
                  f"within 24 months; {len(above)} ended the window at or above $100,000."),
            evidence="ended below $100,000: " + (below or "none"),
        ))
    return out


@router.get("/overview", response_model=ResearchOverview)
def research_overview() -> ResearchOverview:
    frame, source = _read_table()
    config = _read_config()
    downturn = _sorted([
        DownturnRow(
            event=r["event"], strategy=r["strategy"], period_start=r["period_start"],
            period_end=r["period_end"], trading_days=int(r["trading_days"]),
            total_return=r["total_return"], annualized_return=r["annualized_return"],
            annualized_volatility=r["annualized_volatility"], sharpe_ratio=r["sharpe_ratio"],
            maximum_drawdown=r["maximum_drawdown"], transaction_costs=r["transaction_costs"],
        )
        for r in frame[frame["section"] == "downturn"].to_dict("records")
    ])
    recovery = _sorted([
        RecoveryRow(
            event=r["event"], strategy=r["strategy"], trough_date=r["downturn_trough_date"],
            trough_value=r["downturn_trough_value"],
            recovery_return_from_trough=r["recovery_return_from_trough"],
            recovered_within_24_months=bool(r["recovered_within_24_months"]),
            recovery_time_trading_days=(None if _num(r["recovery_time_trading_days"]) is None
                                        else int(float(r["recovery_time_trading_days"]))),
            recovery_date=r["recovery_date"] if isinstance(r["recovery_date"], str) else None,
            ending_value=r["recovery_ending_value"],
        )
        for r in frame[frame["section"] == "recovery"].to_dict("records")
    ])
    portfolio, backtest = config["portfolio"], config["backtest"]
    traditional = config["traditional_strategies"]
    settings = {
        "assets": config["assets"],
        "initial_capital": portfolio["initial_capital"],
        "max_weight": portfolio["max_weight"],
        "transaction_cost_rate": portfolio["transaction_cost_rate"],
        "long_only": portfolio["long_only"],
        "fully_invested": portfolio["fully_invested"],
        "price_field": config["data"]["price_field"],
        "volatility_source": config["data"]["volatility_source"],
        "risk_free_rate_daily": backtest["metrics"]["risk_free_rate_daily"],
        "annualization_trading_days": backtest["metrics"]["annualization_trading_days"],
        "initial_weights": backtest["initial_weights"],
        "equal_weight": traditional["equal_weight"],
        "mean_variance": traditional["mean_variance"],
        "minimum_variance": traditional["minimum_variance"],
        "observation_size": config["reinforcement_learning"]["observation"]["expected_size"],
        "random_seed": config["reproducibility"]["random_seed"],
    }
    events = {k: {p: list(v) for p, v in value.items()} for k, value in config["events"].items()}
    order = lambda kv: EVENT_ORDER.index(kv[0]) if kv[0] in EVENT_ORDER else 99  # noqa: E731
    return ResearchOverview(
        source=source,
        locked=True,
        settings=settings,
        events=dict(sorted(events.items(), key=order)),
        downturn=downturn,
        recovery=recovery,
        findings=_findings(downturn, recovery),
        caveats=[
            "One random seed per RL model and one validation episode per checkpoint.",
            "Nine sector ETFs and a 0% risk-free rate; rankings are exploratory, not statistically conclusive.",
            "The three downturns differ in length, so compare annualized and risk-adjusted measures, "
            "not only raw returns.",
            "Results are historical simulations, not investment advice.",
        ],
    )


class CheckRow(BaseModel):
    strategy: str
    metric: str
    stored: float
    recomputed: float
    matches: bool


class VerifyResult(BaseModel):
    event: str
    period_start: str
    period_end: str
    all_match: bool
    tolerance: float
    checks: list[CheckRow]


VERIFY_METRICS = ("total_return", "annualized_return", "annualized_volatility", "sharpe_ratio",
                  "maximum_drawdown", "transaction_costs")
VERIFY_TOLERANCE = 1e-6


@router.post("/verify/{event}", response_model=VerifyResult)
def verify_event(event: str, services: Services = Depends(get_services)) -> VerifyResult:
    """Recompute one event's downturn results with PortfolioLab's engine and the frozen research
    models, then compare with the stored artifacts. Read-only: nothing is saved."""
    frame, _ = _read_table()
    config = _read_config()
    if event not in config["events"]:
        raise HTTPException(status_code=404, detail=f"Unknown research event {event!r}.")
    returns_path = PROJECT_ROOT / "data" / "processed" / "daily_returns" / "daily_returns.csv"
    if not returns_path.is_file():
        raise HTTPException(status_code=404, detail="Research returns data not found.")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    start, end = (date.fromisoformat(d) for d in config["events"][event]["downturn_test"])
    portfolio = config["portfolio"]
    capital = float(portfolio["initial_capital"])
    cap = float(portfolio["max_weight"])
    cost = float(portfolio["transaction_cost_rate"])

    def settings(strategies: tuple[str, ...]) -> RunConfig:
        return RunConfig(strategies=strategies, start=start, end=end, initial_capital=capital,
                         max_weight=cap, transaction_cost_rate=cost)

    try:
        runs = run_strategies(returns, settings(("equal_weight", "mean_variance", "minimum_variance")))
        vix = load_vix()
        for algo in ("ppo", "ddpg"):
            spec = services.models.get(f"research/{event}/{algo}")
            if spec is None:
                raise HTTPException(status_code=404, detail=f"Research model research/{event}/{algo} not found.")
            result = run_agent(spec, returns, vix, start=start, end=end, initial_capital=capital,
                               transaction_cost_rate=cost, max_weight=cap)
            runs.update(run_strategies(
                returns, settings((algo,)),
                agent_runner=lambda _strategy, _progress, _r=result, _s=spec: (_r, _s.to_json()),
            ))
    except (BacktestError, RLError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    checks: list[CheckRow] = []
    for strategy in STRATEGY_ORDER:
        stored = frame[(frame.event == event) & (frame.strategy == strategy) & (frame.section == "downturn")]
        if stored.empty or strategy not in runs:
            continue
        row = stored.iloc[0]
        for metric in VERIFY_METRICS:
            a, b = float(row[metric]), float(runs[strategy].summary[metric])
            checks.append(CheckRow(
                strategy=strategy, metric=metric, stored=a, recomputed=b,
                matches=abs(a - b) <= VERIFY_TOLERANCE * max(1.0, abs(a)),
            ))
    return VerifyResult(
        event=event, period_start=start.isoformat(), period_end=end.isoformat(),
        all_match=bool(checks) and all(c.matches for c in checks),
        tolerance=VERIFY_TOLERANCE, checks=checks,
    )
