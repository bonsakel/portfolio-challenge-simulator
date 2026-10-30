"""Local price cache: one CSV + provenance entry per symbol, plus a record of failures."""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.assets import normalize_symbol


@dataclass(frozen=True)
class CacheEntry:
    symbol: str
    provider: str
    price_field: str
    retrieved_at: str
    first_date: str
    last_date: str
    rows: int


@dataclass(frozen=True)
class FailureRecord:
    symbol: str
    kind: str  # "no_data" | "provider_error"
    message: str
    at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PriceCache:
    def __init__(self, directory: Path) -> None:
        self.dir = directory / "prices"
        self.index_path = self.dir / "index.json"

    def _path(self, symbol: str) -> Path:
        # normalize_symbol only admits [A-Z-], so the symbol can never escape self.dir.
        return self.dir / f"{normalize_symbol(symbol)}.csv"

    def _read_index(self) -> dict:
        if not self.index_path.is_file():
            return {"entries": {}, "failures": {}}
        return json.loads(self.index_path.read_text())

    def _write_index(self, data: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.index_path)

    def put(self, symbol: str, series: pd.Series, provider: str, price_field: str) -> CacheEntry:
        symbol = normalize_symbol(symbol)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(symbol).with_suffix(".tmp")
        series.rename("adj_close").to_frame().rename_axis("date").to_csv(tmp)
        tmp.replace(self._path(symbol))
        entry = CacheEntry(
            symbol=symbol,
            provider=provider,
            price_field=price_field,
            retrieved_at=_now(),
            first_date=series.index.min().date().isoformat(),
            last_date=series.index.max().date().isoformat(),
            rows=int(len(series)),
        )
        index = self._read_index()
        index["entries"][symbol] = asdict(entry)
        index["failures"].pop(symbol, None)
        self._write_index(index)
        return entry

    def get(self, symbol: str) -> tuple[pd.Series, CacheEntry] | None:
        symbol = normalize_symbol(symbol)
        raw = self._read_index()["entries"].get(symbol)
        path = self._path(symbol)
        if raw is None or not path.is_file():
            return None
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        series = frame["adj_close"].astype(float).rename(symbol)
        return series, CacheEntry(**raw)

    def record_failure(self, symbol: str, kind: str, message: str) -> FailureRecord:
        symbol = normalize_symbol(symbol)
        record = FailureRecord(symbol=symbol, kind=kind, message=message, at=_now())
        index = self._read_index()
        index["failures"][symbol] = asdict(record)
        self._write_index(index)
        return record

    def entries(self) -> list[CacheEntry]:
        return [CacheEntry(**e) for _, e in sorted(self._read_index()["entries"].items())]

    def failures(self) -> list[FailureRecord]:
        return [FailureRecord(**f) for _, f in sorted(self._read_index()["failures"].items())]

    def clear(self) -> int:
        count = len(self._read_index()["entries"])
        if self.dir.is_dir():
            shutil.rmtree(self.dir)
        return count
