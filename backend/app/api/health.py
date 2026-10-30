from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    server_time_utc: datetime


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", version=__version__, server_time_utc=datetime.now(timezone.utc)
    )
