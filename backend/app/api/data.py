from dataclasses import asdict
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator

from app.assets import SymbolError, normalize_symbol
from app.deps import Services, get_services

router = APIRouter(prefix="/data", tags=["data"])

ServicesDep = Annotated[Services, Depends(get_services)]


class SymbolIndexStatus(BaseModel):
    available: bool
    count: int
    retrieved_at: str | None
    sources: list[str]


class CachedSymbol(BaseModel):
    symbol: str
    provider: str
    price_field: str
    retrieved_at: str
    first_date: str
    last_date: str
    rows: int


class Failure(BaseModel):
    symbol: str
    kind: str
    message: str
    at: str


class ProviderStatus(BaseModel):
    name: str
    price_field: str
    attribution: str
    real_time: bool = False


class DataStatus(BaseModel):
    symbol_index: SymbolIndexStatus
    provider: ProviderStatus
    cached_symbols: list[CachedSymbol]
    cached_range: tuple[str, str] | None
    failures: list[Failure]
    disclaimer: str


class GapOut(BaseModel):
    after: str
    before: str
    calendar_days: int


class SymbolCoverageOut(BaseModel):
    symbol: str
    status: str
    source: str | None
    first_date: str | None
    last_date: str | None
    rows: int
    retrieved_at: str | None
    gaps: list[GapOut]
    message: str | None


class CoverageReportOut(BaseModel):
    symbols: list[SymbolCoverageOut]
    requested_start: str | None
    requested_end: str | None
    common_start: str | None
    common_end: str | None
    common_rows: int
    limiting_symbol: str | None
    ready: bool
    issues: list[str]


class CoverageRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=25)
    start: date | None = None
    end: date | None = None

    @field_validator("symbols")
    @classmethod
    def _normalize(cls, values: list[str]) -> list[str]:
        try:
            return [normalize_symbol(v) for v in values]
        except SymbolError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _ordered(self) -> "CoverageRequest":
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be before end.")
        return self


class ClearResult(BaseModel):
    cleared_symbols: int


@router.get("/status", response_model=DataStatus)
def data_status(services: ServicesDep) -> DataStatus:
    info = services.symbols.info()
    market = services.market_data
    entries = market.cache.entries()
    return DataStatus(
        symbol_index=SymbolIndexStatus(
            available=info.available,
            count=info.count,
            retrieved_at=info.retrieved_at,
            sources=list(info.sources),
        ),
        provider=ProviderStatus(
            name=market.provider.name,
            price_field=market.provider.price_field,
            attribution=market.provider.attribution,
        ),
        cached_symbols=[CachedSymbol(**asdict(e)) for e in entries],
        cached_range=(
            (min(e.first_date for e in entries), max(e.last_date for e in entries)) if entries else None
        ),
        failures=[Failure(**asdict(f)) for f in market.cache.failures()],
        disclaimer="Data may be delayed, incomplete, adjusted or revised and is not real-time. "
        "Listing in the symbol index does not guarantee price history is available.",
    )


def _coverage(services: Services, body: CoverageRequest, refresh: bool) -> CoverageReportOut:
    report, _ = services.market_data.coverage(body.symbols, body.start, body.end, refresh=refresh)
    return CoverageReportOut.model_validate(asdict(report))


@router.post("/coverage", response_model=CoverageReportOut)
def coverage(body: CoverageRequest, services: ServicesDep) -> CoverageReportOut:
    """Cache-first availability check; downloads any symbol not yet cached."""
    return _coverage(services, body, refresh=False)


@router.post("/refresh", response_model=CoverageReportOut)
def refresh(body: CoverageRequest, services: ServicesDep) -> CoverageReportOut:
    """Deliberately re-download the listed symbols, replacing their cached history."""
    return _coverage(services, body, refresh=True)


@router.delete("/cache", response_model=ClearResult)
def clear_cache(
    services: ServicesDep,
    confirm: Annotated[bool, Query()] = False,
) -> ClearResult:
    if not confirm:
        raise HTTPException(status_code=400, detail="Clearing the cache requires confirm=true.")
    return ClearResult(cleared_symbols=services.market_data.cache.clear())
