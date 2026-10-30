"""Read-only access to the existing CSC-475 research artifacts (never writes)."""

import math
from datetime import date
from typing import Any

import pandas as pd
import yaml
from fastapi import APIRouter, Depends, HTTPException
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
