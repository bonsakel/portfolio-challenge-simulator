from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.assets import SymbolError, normalize_symbol
from app.validation import ValidationReport


class HoldingIn(BaseModel):
    symbol: str
    weight: float | None = Field(default=None, ge=-1, le=1)

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, value: str) -> str:
        try:
            return normalize_symbol(value)
        except SymbolError as exc:
            raise ValueError(str(exc)) from exc


class PortfolioIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    initial_capital: float = Field(gt=0, le=1e12)
    max_weight: float = Field(default=0.30, gt=0, le=1)
    weight_mode: Literal["equal", "custom"] = "equal"
    holdings: list[HoldingIn] = Field(max_length=50)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name must not be blank.")
        return value

    @model_validator(mode="after")
    def _equal_mode_has_no_weights(self) -> "PortfolioIn":
        if self.weight_mode == "equal":
            for holding in self.holdings:
                holding.weight = None
        return self


class PortfolioOut(PortfolioIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
    validation: ValidationReport


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
