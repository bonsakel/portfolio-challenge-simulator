from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import Services, get_services, get_session
from app.models import PortfolioRow
from app.schemas import PortfolioIn, PortfolioOut, utcnow
from app.validation import ValidationReport, validate_portfolio

router = APIRouter(prefix="/portfolios", tags=["portfolios"])

SessionDep = Annotated[Session, Depends(get_session)]
ServicesDep = Annotated[Services, Depends(get_services)]


def _report(payload: PortfolioIn, services: Services) -> ValidationReport:
    index = services.symbols
    lookup = (lambda s: index.get(s) is not None) if index.available else (lambda s: None)
    return validate_portfolio(
        symbols=[h.symbol for h in payload.holdings],
        weights=[h.weight for h in payload.holdings],
        weight_mode=payload.weight_mode,
        max_weight=payload.max_weight,
        initial_capital=payload.initial_capital,
        is_listed=lookup,
    )


def _out(row: PortfolioRow, services: Services) -> PortfolioOut:
    payload = PortfolioIn(
        name=row.name,
        initial_capital=row.initial_capital,
        max_weight=row.max_weight,
        weight_mode=row.weight_mode,  # type: ignore[arg-type]
        holdings=row.holdings,  # type: ignore[arg-type]
    )
    return PortfolioOut(
        **payload.model_dump(),
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        validation=_report(payload, services),
    )


def _get_or_404(session: Session, portfolio_id: int) -> PortfolioRow:
    row = session.get(PortfolioRow, portfolio_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found.")
    return row


@router.post("/validate", response_model=ValidationReport)
def validate(payload: PortfolioIn, services: ServicesDep) -> ValidationReport:
    return _report(payload, services)


@router.post("", response_model=PortfolioOut, status_code=201)
def create_portfolio(payload: PortfolioIn, session: SessionDep, services: ServicesDep) -> PortfolioOut:
    # Drafts may be saved with validation issues; they must pass validation before a backtest.
    now = utcnow()
    row = PortfolioRow(
        name=payload.name,
        initial_capital=payload.initial_capital,
        max_weight=payload.max_weight,
        weight_mode=payload.weight_mode,
        holdings=[h.model_dump() for h in payload.holdings],
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return _out(row, services)


@router.get("", response_model=list[PortfolioOut])
def list_portfolios(session: SessionDep, services: ServicesDep) -> list[PortfolioOut]:
    rows = session.scalars(select(PortfolioRow).order_by(PortfolioRow.updated_at.desc())).all()
    return [_out(r, services) for r in rows]


@router.get("/{portfolio_id}", response_model=PortfolioOut)
def get_portfolio(portfolio_id: int, session: SessionDep, services: ServicesDep) -> PortfolioOut:
    return _out(_get_or_404(session, portfolio_id), services)


@router.put("/{portfolio_id}", response_model=PortfolioOut)
def update_portfolio(
    portfolio_id: int, payload: PortfolioIn, session: SessionDep, services: ServicesDep
) -> PortfolioOut:
    row = _get_or_404(session, portfolio_id)
    row.name = payload.name
    row.initial_capital = payload.initial_capital
    row.max_weight = payload.max_weight
    row.weight_mode = payload.weight_mode
    row.holdings = [h.model_dump() for h in payload.holdings]
    row.updated_at = utcnow()
    session.commit()
    session.refresh(row)
    return _out(row, services)


@router.delete("/{portfolio_id}", status_code=204)
def delete_portfolio(portfolio_id: int, session: SessionDep) -> Response:
    session.delete(_get_or_404(session, portfolio_id))
    session.commit()
    return Response(status_code=204)
