"""Trained-model catalogue, compatibility checks and background training jobs."""

from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import Services, get_services, get_session
from app.models import PortfolioRow, TrainingJobRow
from app.rl.inference import RESEARCH_VIX_FILE, load_vix, RLError
from app.rl.registry import check_compatibility
from app.rl.training import MIN_TRAINING_DAYS, estimate_seconds
from app.schemas import utcnow

router = APIRouter(tags=["reinforcement-learning"])

SessionDep = Annotated[Session, Depends(get_session)]
ServicesDep = Annotated[Services, Depends(get_services)]

DEFAULT_TIMESTEPS = 100_000  # the research setting
EVAL_FREQ = 2520  # the research validation frequency


def _holdings_equal(portfolio: PortfolioRow) -> bool:
    return portfolio.weight_mode == "equal"


@router.get("/models")
def list_models(
    services: ServicesDep,
    session: SessionDep,
    portfolio_id: int | None = None,
    start: date | None = None,
    end: date | None = None,
    transaction_cost_rate: Annotated[float, Query(ge=0, lt=0.05)] = 0.001,
    initial_allocation: Literal["equal", "portfolio_weights"] = "equal",
) -> dict[str, Any]:
    """Trained models. With a portfolio and window, each gets a compatibility verdict."""
    check = portfolio_id is not None and start is not None and end is not None
    portfolio = session.get(PortfolioRow, portfolio_id) if check else None
    if check and portfolio is None:
        raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found.")
    models = []
    for spec in services.models.list():
        item = spec.to_json()
        item["compatibility"] = None
        if portfolio is not None and start is not None and end is not None:
            equal = initial_allocation == "equal" or _holdings_equal(portfolio)
            item["compatibility"] = check_compatibility(
                spec,
                assets=[h["symbol"] for h in portfolio.holdings],
                max_weight=portfolio.max_weight,
                transaction_cost_rate=transaction_cost_rate,
                first_decision=start,
                last_decision=end,
                equal_initial_weights=equal,
            ).to_json()
        models.append(item)
    try:
        vix = load_vix()
        vix_range = [vix.index.min().date().isoformat(), vix.index.max().date().isoformat()]
    except RLError:
        vix_range = None
    return {"models": models, "vix_range": vix_range, "vix_source": RESEARCH_VIX_FILE.name}


class TrainingRequest(BaseModel):
    portfolio_id: int
    algorithm: Literal["PPO", "DDPG"]
    training_start: date
    training_end: date
    validation_start: date
    validation_end: date
    total_timesteps: int = Field(DEFAULT_TIMESTEPS, ge=5_000, le=300_000)
    transaction_cost_rate: float = Field(0.001, ge=0, lt=0.05)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    confirm: bool = Field(False, description="Must be true: training is never started implicitly.")

    @model_validator(mode="after")
    def _checks(self) -> "TrainingRequest":
        if not self.confirm:
            raise ValueError("Training uses significant CPU time and starts only after explicit confirmation (confirm=true).")
        if not self.training_start < self.training_end < self.validation_start < self.validation_end:
            raise ValueError(
                "Dates must run in order: training start < training end < validation start < validation end."
            )
        return self


def _job_status(row: TrainingJobRow, services: Services) -> dict[str, Any]:
    config = row.config
    live = services.training.progress(row.id)
    return {
        "id": row.id,
        "algorithm": row.algorithm,
        "status": row.status,
        "progress": 1.0 if row.status == "succeeded" else (live or 0.0),
        "estimated_seconds": estimate_seconds(row.algorithm, config["total_timesteps"]),
        "error": row.error,
        "model_id": row.model_id,
        "config": config,
        "created_at": row.created_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
    }


@router.post("/training-jobs", status_code=202)
def start_training(body: TrainingRequest, session: SessionDep, services: ServicesDep) -> dict[str, Any]:
    portfolio = session.get(PortfolioRow, body.portfolio_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail=f"Portfolio {body.portfolio_id} not found.")
    symbols = [h["symbol"] for h in portfolio.holdings]
    if len(symbols) != 9:
        raise HTTPException(
            status_code=422,
            detail=f"The research environment supports exactly 9 assets; this portfolio has {len(symbols)}.",
        )
    if services.training.active():
        raise HTTPException(status_code=409, detail="Another training job is already queued or running.")
    row = TrainingJobRow(
        algorithm=body.algorithm,
        status="queued",
        config={
            **body.model_dump(mode="json", exclude={"confirm"}),
            "confirmed": True,
            "symbols": symbols,
            "max_weight": portfolio.max_weight,
            "initial_capital": portfolio.initial_capital,
            "eval_freq": min(EVAL_FREQ, body.total_timesteps // 2),
            "min_training_days": MIN_TRAINING_DAYS,
        },
        created_at=utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    services.training.submit(row.id)
    return _job_status(row, services)


@router.get("/training-jobs")
def list_training_jobs(session: SessionDep, services: ServicesDep, limit: int = 20) -> list[dict[str, Any]]:
    rows = session.scalars(select(TrainingJobRow).order_by(TrainingJobRow.id.desc()).limit(min(limit, 100)))
    return [_job_status(r, services) for r in rows]


def _get_job(session: Session, job_id: int) -> TrainingJobRow:
    row = session.get(TrainingJobRow, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found.")
    return row


@router.get("/training-jobs/{job_id}")
def get_training_job(job_id: int, session: SessionDep, services: ServicesDep) -> dict[str, Any]:
    return _job_status(_get_job(session, job_id), services)


@router.post("/training-jobs/{job_id}/cancel")
def cancel_training_job(job_id: int, session: SessionDep, services: ServicesDep) -> dict[str, Any]:
    row = _get_job(session, job_id)
    if row.status not in ("queued", "running"):
        raise HTTPException(status_code=409, detail=f"Job {job_id} already {row.status}.")
    services.training.cancel(job_id)
    session.refresh(row)
    return _job_status(row, services)
