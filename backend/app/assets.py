"""Locally maintained symbol index built from the NASDAQ Trader symbol directory.

Source: https://www.nasdaqtrader.com/dynamic/SymDir/ (nasdaqlisted.txt, otherlisted.txt).
It lists *currently listed* US securities only; delisted symbols are absent, so a
backtest over a custom universe is a selected-universe study, not a survivorship-free one.
Being in the index does not imply price history is available (checked in a later milestone).
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

SOURCES = {
    "nasdaqlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "otherlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
OTHER_EXCHANGES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}
COLUMNS = ["symbol", "name", "exchange", "asset_type"]

# Plain tickers, optionally with one share-class suffix (BRK-B). Preferreds, warrants,
# units and rights carry characters outside this pattern and are excluded.
SYMBOL_RE = re.compile(r"^[A-Z]{1,6}(-[A-Z]{1,2})?$")
_NON_COMMON_NAME_RE = re.compile(
    r"\b(?:warrants?|units?|preferred|depositary shares|subordinated|debentures?|notes?)\b|\brights?\s*$",
    re.IGNORECASE,
)


class SymbolError(ValueError):
    pass


def normalize_symbol(raw: str) -> str:
    """Upper-case, trim, map '.' class separators to '-', and validate."""
    symbol = raw.strip().upper().replace(".", "-")
    if not SYMBOL_RE.fullmatch(symbol):
        raise SymbolError(f"Invalid ticker symbol: {raw!r}")
    return symbol


def _clean(frame: pd.DataFrame, symbol_col: str, exchange: pd.Series) -> pd.DataFrame:
    frame = frame[frame["Test Issue"] == "N"]
    out = pd.DataFrame(
        {
            "symbol": frame[symbol_col].str.strip().str.upper().str.replace(".", "-", regex=False),
            "name": frame["Security Name"].str.strip(),
            "exchange": exchange.loc[frame.index],
            "asset_type": frame["ETF"].map({"Y": "ETF", "N": "Equity"}),
        }
    )
    return out


def parse_directory(nasdaq_text: str, other_text: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse both directory files into a clean index plus counts of what was dropped."""
    nasdaq = pd.read_csv(StringIO(nasdaq_text), sep="|", dtype=str, keep_default_na=False)
    nasdaq = nasdaq[~nasdaq["Symbol"].str.startswith("File Creation Time")]
    other = pd.read_csv(StringIO(other_text), sep="|", dtype=str, keep_default_na=False)
    other = other[~other["ACT Symbol"].str.startswith("File Creation Time")]

    combined = pd.concat(
        [
            _clean(nasdaq, "Symbol", pd.Series("NASDAQ", index=nasdaq.index)),
            _clean(other, "ACT Symbol", other["Exchange"].map(OTHER_EXCHANGES).fillna("Other")),
        ],
        ignore_index=True,
    )
    total = len(combined)
    combined = combined[combined["symbol"].map(lambda s: bool(SYMBOL_RE.fullmatch(s)))]
    bad_symbol = total - len(combined)
    non_common = combined["name"].str.contains(_NON_COMMON_NAME_RE)
    # ETFs legitimately mention "notes"/"units" rarely, so only filter equities.
    drop = non_common & (combined["asset_type"] == "Equity")
    combined = combined[~drop].drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    stats = {
        "rows_dropped_invalid_symbol": int(bad_symbol),
        "rows_dropped_non_common": int(drop.sum()),
    }
    return combined[COLUMNS], stats


@dataclass(frozen=True)
class IndexInfo:
    available: bool
    count: int = 0
    retrieved_at: str | None = None
    sources: tuple[str, ...] = ()


class SymbolIndex:
    """CSV-backed index, reloaded automatically when the file changes."""

    def __init__(self, directory: Path) -> None:
        self.csv_path = directory / "symbols.csv"
        self.meta_path = directory / "symbols_meta.json"
        self._frame: pd.DataFrame | None = None
        self._mtime: float | None = None

    def _load(self) -> pd.DataFrame | None:
        if not self.csv_path.is_file():
            self._frame = None
            return None
        mtime = self.csv_path.stat().st_mtime
        if self._frame is None or mtime != self._mtime:
            self._frame = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
            self._mtime = mtime
        return self._frame

    @property
    def available(self) -> bool:
        return self._load() is not None

    def info(self) -> IndexInfo:
        frame = self._load()
        if frame is None:
            return IndexInfo(available=False)
        meta = json.loads(self.meta_path.read_text()) if self.meta_path.is_file() else {}
        return IndexInfo(
            available=True,
            count=len(frame),
            retrieved_at=meta.get("retrieved_at"),
            sources=tuple(meta.get("sources", [])),
        )

    def get(self, symbol: str) -> dict[str, str] | None:
        frame = self._load()
        if frame is None:
            return None
        hit = frame[frame["symbol"] == symbol]
        return None if hit.empty else hit.iloc[0].to_dict()

    def search(self, query: str, limit: int = 20) -> list[dict[str, str]]:
        """Rank: exact symbol, symbol prefix, name starts-with, name word prefix, all tokens in name."""
        frame = self._load()
        if frame is None:
            raise FileNotFoundError(str(self.csv_path))
        q = query.strip().upper().replace(".", "-")
        if not q:
            return []
        tokens = q.split()
        symbol = frame["symbol"]
        name = frame["name"].str.upper()
        rank = pd.Series(9, index=frame.index)
        rank[name.map(lambda n: all(t in n for t in tokens))] = 4
        rank[name.map(lambda n: any(w.startswith(q) for w in re.split(r"[^A-Z0-9]+", n)))] = 3
        rank[name.str.startswith(q)] = 2
        rank[symbol.str.startswith(q)] = 1
        rank[symbol == q] = 0
        hits = frame[rank < 9].assign(_rank=rank[rank < 9])
        hits = hits.assign(_nlen=hits["name"].str.len())
        hits = hits.sort_values(["_rank", "_nlen", "symbol"]).head(limit)
        return hits[COLUMNS].to_dict("records")


def build_index(directory: Path, nasdaq_text: str, other_text: str) -> IndexInfo:
    """Write the index and its provenance metadata (retrieval time, sources, drop counts)."""
    frame, stats = parse_directory(nasdaq_text, other_text)
    directory.mkdir(parents=True, exist_ok=True)
    index = SymbolIndex(directory)
    frame.to_csv(index.csv_path, index=False)
    index.meta_path.write_text(
        json.dumps(
            {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sources": list(SOURCES.values()),
                "count": len(frame),
                **stats,
            },
            indent=2,
        )
    )
    return index.info()
