from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from app.assets import SymbolError, normalize_symbol
from app.deps import Services, get_services

router = APIRouter(prefix="/assets", tags=["assets"])


class Asset(BaseModel):
    symbol: str
    name: str
    exchange: str
    asset_type: str


class AssetDetail(Asset):
    # Price-history coverage is verified once the data provider lands (Milestone 3).
    data_availability: str = "not_checked"


class SearchResponse(BaseModel):
    query: str
    results: list[Asset]
    source: str


SOURCE_LABEL = "NASDAQ Trader symbol directory (currently listed US securities)"


@router.get("/search", response_model=SearchResponse)
def search_assets(
    services: Annotated[Services, Depends(get_services)],
    q: Annotated[str, Query(min_length=1, max_length=40)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> SearchResponse:
    if not services.symbols.available:
        raise HTTPException(
            status_code=503,
            detail="Symbol index not built. Run `python -m app.update_symbol_index` in backend/.",
        )
    results = services.symbols.search(q, limit)
    return SearchResponse(query=q, results=[Asset(**r) for r in results], source=SOURCE_LABEL)


@router.get("/{symbol}", response_model=AssetDetail)
def get_asset(
    services: Annotated[Services, Depends(get_services)],
    symbol: Annotated[str, Path(max_length=12)],
) -> AssetDetail:
    try:
        normalized = normalize_symbol(symbol)
    except SymbolError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not services.symbols.available:
        raise HTTPException(status_code=503, detail="Symbol index not built.")
    record = services.symbols.get(normalized)
    if record is None:
        raise HTTPException(status_code=404, detail=f"{normalized} is not in the supported symbol index.")
    return AssetDetail(**record)
