"""Price retrieval (cache-first), gap detection, and date alignment."""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.assets import normalize_symbol
from app.market_data.cache import CacheEntry, PriceCache
from app.market_data.provider import NoDataError, PriceProvider, ProviderError

# Longest ordinary closure is a holiday weekend (4 calendar days). Anything above this is
# reported as a gap; nothing is ever filled across it.
GAP_THRESHOLD_DAYS = 5
# Full history is downloaded once and sliced locally, so cache hits never depend on the range.
HISTORY_START = date(1990, 1, 1)
STALE_AFTER_DAYS = 7


@dataclass(frozen=True)
class Gap:
    after: str
    before: str
    calendar_days: int


@dataclass
class SymbolCoverage:
    symbol: str
    status: str  # "ok" | "unavailable" | "provider_error"
    source: str | None = None  # "cache" | "downloaded"
    first_date: str | None = None
    last_date: str | None = None
    rows: int = 0
    retrieved_at: str | None = None
    gaps: list[Gap] = field(default_factory=list)
    message: str | None = None


@dataclass
class CoverageReport:
    symbols: list[SymbolCoverage]
    requested_start: str | None
    requested_end: str | None
    common_start: str | None
    common_end: str | None
    common_rows: int
    limiting_symbol: str | None
    ready: bool
    issues: list[str]


def find_gaps(series: pd.Series, threshold_days: int = GAP_THRESHOLD_DAYS) -> list[Gap]:
    deltas = series.index.to_series().diff().dt.days
    gaps = []
    for idx in deltas[deltas > threshold_days].index:
        pos = series.index.get_loc(idx)
        gaps.append(
            Gap(
                after=series.index[pos - 1].date().isoformat(),
                before=idx.date().isoformat(),
                calendar_days=int(deltas[idx]),
            )
        )
    return gaps


def align_prices(
    series: dict[str, pd.Series], start: date | None = None, end: date | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Inner-join on dates where *every* asset has a price, within [start, end].

    No forward-fill: a date missing for any asset is dropped for all. Returns the aligned
    price frame and, per symbol, how many of its in-window observations were dropped.
    """
    window = {}
    for symbol, s in series.items():
        s = s.loc[(s.index >= pd.Timestamp(start)) if start else slice(None)]
        s = s.loc[(s.index <= pd.Timestamp(end)) if end else slice(None)]
        window[symbol] = s
    frame = pd.concat(window, axis=1, join="inner").sort_index()
    dropped = {sym: int(len(s) - len(frame)) for sym, s in window.items()}
    return frame, dropped


class MarketDataService:
    def __init__(self, provider: PriceProvider, cache: PriceCache) -> None:
        self.provider = provider
        self.cache = cache

    def load(self, symbol: str, refresh: bool = False) -> tuple[pd.Series, CacheEntry, str]:
        """Cache-first. `refresh=True` forces a deliberate re-download. Raises NoDataError /
        ProviderError, recording the failure so Data Status can show it."""
        symbol = normalize_symbol(symbol)
        if not refresh:
            hit = self.cache.get(symbol)
            if hit is not None:
                return hit[0], hit[1], "cache"
        try:
            series = self.provider.fetch_prices(symbol, HISTORY_START)
        except NoDataError as exc:
            self.cache.record_failure(symbol, "no_data", str(exc))
            raise
        except ProviderError as exc:
            self.cache.record_failure(symbol, "provider_error", str(exc))
            raise
        entry = self.cache.put(symbol, series, self.provider.name, self.provider.price_field)
        return series, entry, "downloaded"

    def coverage(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
        refresh: bool = False,
    ) -> tuple[CoverageReport, dict[str, pd.Series]]:
        loaded: dict[str, pd.Series] = {}
        rows: list[SymbolCoverage] = []
        for symbol in dict.fromkeys(normalize_symbol(raw) for raw in symbols):
            try:
                series, entry, source = self.load(symbol, refresh=refresh)
            except NoDataError as exc:
                rows.append(SymbolCoverage(symbol, "unavailable", message=str(exc)))
            except ProviderError as exc:
                rows.append(SymbolCoverage(symbol, "provider_error", message=str(exc)))
            else:
                loaded[symbol] = series
                rows.append(
                    SymbolCoverage(
                        symbol=symbol,
                        status="ok",
                        source=source,
                        first_date=entry.first_date,
                        last_date=entry.last_date,
                        rows=entry.rows,
                        retrieved_at=entry.retrieved_at,
                        gaps=find_gaps(series),
                    )
                )

        issues: list[str] = []
        for row in rows:
            if row.status != "ok":
                issues.append(f"{row.symbol}: {row.message}")
        common_start = common_end = limiting = None
        common_rows = 0
        if loaded:
            frame, _ = align_prices(loaded, start, end)
            common_rows = len(frame)
            if common_rows:
                common_start = frame.index[0].date().isoformat()
                common_end = frame.index[-1].date().isoformat()
                firsts = {s: v.index[0] for s, v in loaded.items()}
                limiting = max(firsts, key=lambda s: firsts[s])
                if start and pd.Timestamp(start) < frame.index[0] - pd.Timedelta(days=7):
                    issues.append(
                        f"Requested start {start} is before the common history, which begins "
                        f"{common_start} (limited by {limiting})."
                    )
                if end and pd.Timestamp(end) > frame.index[-1] + pd.Timedelta(days=7):
                    issues.append(
                        f"Requested end {end} is after the latest common data, {common_end}."
                    )
            else:
                issues.append("The selected assets have no overlapping trading dates in this window.")
            ok_rows = [r for r in rows if r.status == "ok"]
            windowed = {
                r.symbol: [
                    g
                    for g in r.gaps
                    if (start is None or g.after >= start.isoformat())
                    and (end is None or g.before <= end.isoformat())
                ]
                for r in ok_rows
            }
            # A gap present for every asset is almost certainly a market closure, not missing data.
            shared: set[tuple[str, str]] = set()
            if len(ok_rows) >= 2:
                shared = set.intersection(*({(g.after, g.before) for g in gs} for gs in windowed.values()))
            if shared:
                after, before = min(shared)
                issues.append(
                    f"{len(shared)} gap(s) longer than {GAP_THRESHOLD_DAYS} calendar days are shared by all "
                    f"assets (likely market closures), e.g. {after} → {before}; returns across them span the gap."
                )
            for row in ok_rows:
                own = [g for g in windowed[row.symbol] if (g.after, g.before) not in shared]
                if own:
                    first = own[0]
                    issues.append(
                        f"{row.symbol}: {len(own)} gap(s) longer than {GAP_THRESHOLD_DAYS} "
                        f"calendar days in the window (e.g. {first.after} → {first.before}); "
                        "dates missing for any asset are excluded for all."
                    )
            today = pd.Timestamp(date.today())
            for row in rows:
                if row.status == "ok" and row.last_date:
                    age = (today - pd.Timestamp(row.last_date)).days
                    if age > STALE_AFTER_DAYS:
                        issues.append(
                            f"{row.symbol}: latest price is {row.last_date} ({age} days old); "
                            "history may have ended (delisting, merger or symbol change)."
                        )
        ready = bool(loaded) and len(loaded) == len(rows) and common_rows >= 2
        report = CoverageReport(
            symbols=rows,
            requested_start=start.isoformat() if start else None,
            requested_end=end.isoformat() if end else None,
            common_start=common_start,
            common_end=common_end,
            common_rows=common_rows,
            limiting_symbol=limiting,
            ready=ready,
            issues=issues,
        )
        return report, loaded
