import csv
import io
from datetime import date, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, undefer

from app.api.portfolios import _report
from app.assets import normalize_symbol
from app.backtests.runner import STRATEGY_LABELS
from app.deps import Services, get_services, get_session
from app.models import BacktestRow, PortfolioRow
from app.rl.registry import check_compatibility
from app.schemas import PortfolioIn, utcnow

router = APIRouter(prefix="/backtests", tags=["backtests"])

SessionDep = Annotated[Session, Depends(get_session)]
ServicesDep = Annotated[Services, Depends(get_services)]
StrategyKey = Literal["equal_weight", "mean_variance", "minimum_variance", "ppo", "ddpg"]


class BacktestRequest(BaseModel):
    portfolio_id: int
    strategies: list[StrategyKey] = Field(min_length=1, max_length=5)
    rl_models: dict[Literal["ppo", "ddpg"], str] = Field(
        default_factory=dict, description="Model id for each PPO/DDPG strategy; never chosen implicitly."
    )
    start: date
    end: date
    lookback_days: int = Field(252, ge=20, le=1260, description="Trading days used by optimizers.")
    transaction_cost_rate: float = Field(0.001, ge=0, lt=0.05, description="Fraction of traded value.")
    rebalance: Literal["daily", "monthly"] | None = Field(
        None, description="Null uses each strategy's research default."
    )
    turnover_threshold: float | None = Field(
        None, ge=0, le=0.5, description="Skip rebalances smaller than this; null uses research defaults."
    )
    risk_free_rate_annual: float = Field(0.0, ge=0, le=0.25)
    initial_allocation: Literal["equal", "portfolio_weights"] = "equal"
    benchmark: str | None = Field(None, description="Optional ticker compared as buy and hold, e.g. SPY.")

    @model_validator(mode="after")
    def _checks(self) -> "BacktestRequest":
        if self.benchmark is not None and self.benchmark.strip():
            self.benchmark = normalize_symbol(self.benchmark)
        else:
            self.benchmark = None
        if self.start >= self.end:
            raise ValueError("start must be before end.")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategies must not repeat.")
        for key in ("ppo", "ddpg"):
            if key in self.strategies and not self.rl_models.get(key):
                raise ValueError(f"Choose a trained model for {key.upper()} (see GET /api/v1/models).")
        self.rl_models = {k: v for k, v in self.rl_models.items() if k in self.strategies}
        return self


class BacktestStatus(BaseModel):
    id: int
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: float
    error: str | None
    portfolio_id: int | None
    portfolio_name: str
    created_at: datetime
    finished_at: datetime | None
    config: dict[str, Any]
    summary: dict[str, Any] | None


class BacktestResults(BacktestStatus):
    series: dict[str, Any]
    benchmark: dict[str, Any] | None = None


def _status(row: BacktestRow, services: Services) -> dict[str, Any]:
    live = services.backtests.progress(row.id)
    done = row.status in ("succeeded", "failed")
    return dict(
        id=row.id,
        status=row.status,
        progress=1.0 if done else (live or 0.0),
        error=row.error,
        portfolio_id=row.portfolio_id,
        portfolio_name=row.portfolio_name,
        created_at=row.created_at,
        finished_at=row.finished_at,
        config=row.config,
        summary=row.summary,
    )


def _get_or_404(session: Session, backtest_id: int, with_result: bool = False) -> BacktestRow:
    query = select(BacktestRow).where(BacktestRow.id == backtest_id)
    if with_result:
        query = query.options(undefer(BacktestRow.result))
    row = session.scalars(query).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found.")
    return row


@router.post("", response_model=BacktestStatus, status_code=202)
def start_backtest(body: BacktestRequest, session: SessionDep, services: ServicesDep) -> dict[str, Any]:
    """Validate the request and the portfolio, then queue the run. Poll GET /backtests/{id}."""
    portfolio = session.get(PortfolioRow, body.portfolio_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail=f"Portfolio {body.portfolio_id} not found.")
    payload = PortfolioIn(
        name=portfolio.name,
        initial_capital=portfolio.initial_capital,
        max_weight=portfolio.max_weight,
        weight_mode=portfolio.weight_mode,  # type: ignore[arg-type]
        holdings=portfolio.holdings,  # type: ignore[arg-type]
    )
    report = _report(payload, services)
    errors = [i.message for i in report.issues if i.severity == "error"]
    if errors:
        raise HTTPException(status_code=422, detail="Portfolio is not valid: " + " ".join(errors))

    symbols = [h.symbol for h in payload.holdings]
    equal_start = body.initial_allocation == "equal" or len(
        {round(w, 9) for w in report.effective_weights.values()}
    ) == 1
    for strategy, model_id in body.rl_models.items():
        spec = services.models.get(model_id)
        if spec is None or spec.algorithm.lower() != strategy:
            raise HTTPException(status_code=422, detail=f"Model {model_id!r} is not an available {strategy.upper()} model.")
        verdict = check_compatibility(
            spec, assets=symbols, max_weight=portfolio.max_weight,
            transaction_cost_rate=body.transaction_cost_rate,
            first_decision=body.start, last_decision=body.end, equal_initial_weights=equal_start,
        )
        if not verdict.compatible:
            raise HTTPException(
                status_code=422, detail=f"{spec.label} cannot be used: " + " ".join(verdict.issues)
            )
    row = BacktestRow(
        portfolio_id=portfolio.id,
        portfolio_name=portfolio.name,
        status="queued",
        config={
            **body.model_dump(mode="json"),
            "symbols": symbols,
            "initial_capital": portfolio.initial_capital,
            "max_weight": portfolio.max_weight,
            "initial_weights": report.effective_weights if body.initial_allocation == "portfolio_weights" else None,
            "strategy_labels": {k: STRATEGY_LABELS[k] for k in body.strategies},
        },
        created_at=utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    services.backtests.submit(row.id)
    return _status(row, services)


@router.get("", response_model=list[BacktestStatus])
def list_backtests(session: SessionDep, services: ServicesDep, limit: int = 20) -> list[dict[str, Any]]:
    rows = session.scalars(select(BacktestRow).order_by(BacktestRow.id.desc()).limit(min(limit, 100))).all()
    return [_status(r, services) for r in rows]


@router.get("/{backtest_id}", response_model=BacktestStatus)
def get_backtest(backtest_id: int, session: SessionDep, services: ServicesDep) -> dict[str, Any]:
    return _status(_get_or_404(session, backtest_id), services)


@router.get("/{backtest_id}/results", response_model=BacktestResults)
def get_results(backtest_id: int, session: SessionDep, services: ServicesDep) -> dict[str, Any]:
    row = _get_or_404(session, backtest_id, with_result=True)
    if row.status != "succeeded" or row.result is None:
        raise HTTPException(status_code=409, detail=f"Backtest {backtest_id} has no results (status: {row.status}).")
    return {**_status(row, services), "series": row.result["series"], "benchmark": row.result.get("benchmark")}


@router.get("/{backtest_id}/export.csv")
def export_csv(backtest_id: int, session: SessionDep) -> Response:
    row = _get_or_404(session, backtest_id, with_result=True)
    if row.status != "succeeded" or row.result is None:
        raise HTTPException(status_code=409, detail=f"Backtest {backtest_id} has no results (status: {row.status}).")
    symbols: list[str] = row.config["symbols"]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        ["date", "strategy", "portfolio_value", "net_return", "drawdown", "turnover", "transaction_cost"]
        + [f"weight_{s}" for s in symbols]
    )
    for key, s in row.result["series"].items():
        for i, day in enumerate(s["dates"]):
            writer.writerow(
                [day, key, s["portfolio_value"][i], s["net_return"][i], s["drawdown"][i], s["turnover"][i],
                 s["transaction_cost"][i]]
                + [s["weights"][sym][i] for sym in symbols]
            )
    bench = row.result.get("benchmark")
    if bench:
        label = f"benchmark_{row.config['benchmark']}"
        for i, day in enumerate(bench["dates"]):
            writer.writerow(
                [day, label, bench["portfolio_value"][i], bench["net_return"][i], bench["drawdown"][i], "", ""]
                + [""] * len(symbols)
            )
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="backtest_{backtest_id}.csv"'},
    )


@router.delete("/{backtest_id}", status_code=204)
def delete_backtest(backtest_id: int, session: SessionDep) -> Response:
    row = _get_or_404(session, backtest_id)
    if row.status in ("queued", "running"):
        raise HTTPException(status_code=409, detail="A running backtest cannot be deleted.")
    session.delete(row)
    session.commit()
    return Response(status_code=204)
