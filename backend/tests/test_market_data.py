from datetime import date

import pandas as pd
import pytest

from app.assets import SymbolError
from app.market_data.cache import PriceCache
from app.market_data.provider import NoDataError, PriceProvider, ProviderError, clean_series
from app.market_data.service import MarketDataService, align_prices, find_gaps


def make_series(start: str, periods: int, base: float = 100.0, freq: str = "B") -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.Series([base + i for i in range(periods)], index=idx, dtype=float)


class FakeProvider(PriceProvider):
    name = "fake"
    attribution = "test"

    def __init__(self, data: dict[str, pd.Series], errors: dict[str, Exception] | None = None):
        self.data = data
        self.errors = errors or {}
        self.calls: list[str] = []

    def fetch_prices(self, symbol, start):
        self.calls.append(symbol)
        if symbol in self.errors:
            raise self.errors[symbol]
        if symbol not in self.data:
            raise NoDataError(f"no data for {symbol}")
        return clean_series(self.data[symbol], symbol)


# ---- clean_series -------------------------------------------------------------------------

def test_clean_series_drops_nan_without_filling_and_normalizes_index():
    idx = pd.to_datetime(["2020-01-02 00:00", "2020-01-03 00:00", "2020-01-06 00:00", "2020-01-06 00:00"])
    raw = pd.Series([10.0, None, 12.0, 13.0], index=idx)
    out = clean_series(raw, "X")
    assert list(out.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-06"]
    assert list(out) == [10.0, 13.0]  # NaN row removed, duplicate keeps last, nothing interpolated


def test_clean_series_rejects_nonpositive_and_empty():
    with pytest.raises(NoDataError):
        clean_series(pd.Series([1.0, 0.0], index=pd.date_range("2020-01-01", periods=2)), "X")
    with pytest.raises(NoDataError):
        clean_series(pd.Series([None, None], index=pd.date_range("2020-01-01", periods=2)), "X")


def test_clean_series_strips_timezone():
    raw = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-02 05:00"], tz="UTC"))
    assert clean_series(raw, "X").index.tz is None


# ---- gaps and alignment -------------------------------------------------------------------

def test_find_gaps_flags_only_gaps_over_threshold():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-06", "2020-01-20", "2020-01-21"])
    gaps = find_gaps(pd.Series(1.0, index=idx))
    assert [(g.after, g.before, g.calendar_days) for g in gaps] == [("2020-01-06", "2020-01-20", 14)]


def test_find_gaps_ignores_long_weekend():
    idx = pd.to_datetime(["2020-04-09", "2020-04-13"])  # Good Friday weekend, 4 days
    assert find_gaps(pd.Series(1.0, index=idx)) == []


def test_align_prices_inner_join_no_forward_fill():
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]))
    b = pd.Series([10.0, 30.0, 40.0], index=pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"]))
    frame, dropped = align_prices({"A": a, "B": b})
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-03", "2020-01-06"]
    assert frame["A"].tolist() == [1.0, 3.0, 4.0]  # 2020-01-02 dropped for both, B not filled
    assert dropped == {"A": 1, "B": 0}


def test_align_prices_window_is_inclusive():
    s = make_series("2020-01-01", 10)
    frame, _ = align_prices({"A": s}, date(2020, 1, 3), date(2020, 1, 7))
    assert frame.index[0] == pd.Timestamp("2020-01-03") and frame.index[-1] == pd.Timestamp("2020-01-07")


# ---- cache --------------------------------------------------------------------------------

def test_cache_round_trip_and_provenance(tmp_path):
    cache = PriceCache(tmp_path)
    s = make_series("2020-01-01", 5)
    entry = cache.put("xlk", s, "fake", "Adj Close")
    assert entry.symbol == "XLK" and entry.rows == 5 and entry.provider == "fake"
    assert entry.first_date == "2020-01-01" and entry.retrieved_at
    got, meta = cache.get("XLK")
    pd.testing.assert_series_equal(got, s.rename("XLK"), check_freq=False, check_names=False)
    assert meta == entry


def test_cache_failure_recorded_then_cleared_on_success(tmp_path):
    cache = PriceCache(tmp_path)
    cache.record_failure("ABC", "no_data", "nothing")
    assert [f.symbol for f in cache.failures()] == ["ABC"]
    cache.put("ABC", make_series("2020-01-01", 3), "fake", "Adj Close")
    assert cache.failures() == []


def test_cache_clear_and_miss(tmp_path):
    cache = PriceCache(tmp_path)
    cache.put("A", make_series("2020-01-01", 3), "fake", "Adj Close")
    assert cache.clear() == 1
    assert cache.get("A") is None and cache.entries() == []


def test_cache_rejects_path_traversal_symbols(tmp_path):
    cache = PriceCache(tmp_path)
    for bad in ["../../etc/passwd", "a/b", "..", "A B"]:
        with pytest.raises(SymbolError):
            cache.get(bad)
    assert not list(tmp_path.parent.glob("etc"))


# ---- service ------------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path):
    provider = FakeProvider(
        {
            "AAA": make_series("2019-01-01", 700),
            "BBB": make_series("2020-06-01", 300),
        },
        errors={"ERR": ProviderError("rate limited")},
    )
    return MarketDataService(provider, PriceCache(tmp_path)), provider


def test_load_is_cache_first_and_refresh_is_deliberate(svc):
    service, provider = svc
    _, _, source = service.load("AAA")
    assert source == "downloaded" and provider.calls == ["AAA"]
    _, entry, source = service.load("aaa")
    assert source == "cache" and provider.calls == ["AAA"]
    service.load("AAA", refresh=True)
    assert provider.calls == ["AAA", "AAA"]


def test_failures_are_recorded_by_kind_and_reraised(svc):
    service, _ = svc
    with pytest.raises(NoDataError):
        service.load("NOPE")
    with pytest.raises(ProviderError):
        service.load("ERR")
    kinds = {f.symbol: f.kind for f in service.cache.failures()}
    assert kinds == {"NOPE": "no_data", "ERR": "provider_error"}


def test_coverage_reports_limiting_symbol_and_common_range(svc):
    service, _ = svc
    report, loaded = service.coverage(["AAA", "BBB"], date(2019, 1, 1), date(2021, 6, 30))
    assert set(loaded) == {"AAA", "BBB"}
    assert report.limiting_symbol == "BBB"
    assert report.common_start == "2020-06-01"
    assert report.ready is True
    assert any("before the common history" in i and "BBB" in i for i in report.issues)
    sources = {s.symbol: s.source for s in report.symbols}
    assert sources == {"AAA": "downloaded", "BBB": "downloaded"}


def test_coverage_with_unavailable_symbol_is_not_ready(svc):
    service, _ = svc
    report, _ = service.coverage(["AAA", "NOPE", "ERR"])
    status = {s.symbol: s.status for s in report.symbols}
    assert status == {"AAA": "ok", "NOPE": "unavailable", "ERR": "provider_error"}
    assert report.ready is False
    assert any("rate limited" in i for i in report.issues)


def test_coverage_no_overlap_is_not_ready(tmp_path):
    provider = FakeProvider(
        {"A": make_series("2010-01-01", 50), "B": make_series("2020-01-01", 50)}
    )
    service = MarketDataService(provider, PriceCache(tmp_path))
    report, _ = service.coverage(["A", "B"])
    assert report.ready is False and report.common_rows == 0
    assert any("no overlapping" in i for i in report.issues)


def test_coverage_flags_stale_history(tmp_path):
    provider = FakeProvider({"OLD": make_series("2010-01-01", 100)})
    service = MarketDataService(provider, PriceCache(tmp_path))
    report, _ = service.coverage(["OLD"])
    assert any("history may have ended" in i for i in report.issues)


def test_coverage_dedupes_symbols(svc):
    service, provider = svc
    report, _ = service.coverage(["AAA", "aaa", "AAA"])
    assert [s.symbol for s in report.symbols] == ["AAA"] and provider.calls == ["AAA"]


def test_gap_warning_only_for_gaps_inside_window(tmp_path):
    idx = pd.to_datetime(["2001-09-10", "2001-09-17", "2020-01-02", "2020-01-03"])
    provider = FakeProvider({"G": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)})
    service = MarketDataService(provider, PriceCache(tmp_path))
    outside, _ = service.coverage(["G"], date(2020, 1, 1), date(2020, 1, 31))
    assert not any("gap(s)" in i for i in outside.issues)
    assert len(outside.symbols[0].gaps) >= 1  # still reported in the per-symbol data
    inside, _ = service.coverage(["G"], date(2001, 1, 1), date(2001, 12, 31))
    assert any("2001-09-10 → 2001-09-17" in i for i in inside.issues)


def test_gap_shared_by_all_assets_is_one_market_closure_note(tmp_path):
    idx = pd.to_datetime(["2001-09-10", "2001-09-17", "2001-09-18", "2001-12-03", "2001-12-31"])
    data = {
        "SAA": pd.Series([1.0, 2, 3, 4, 5], index=idx),
        "SBB": pd.Series([1.0, 2, 3, 4, 5], index=idx),
    }
    service = MarketDataService(FakeProvider(data), PriceCache(tmp_path))
    report, _ = service.coverage(["SAA", "SBB"])
    gap_notes = [i for i in report.issues if "gap(s)" in i]
    assert len(gap_notes) == 1 and "shared by all assets" in gap_notes[0]
    assert not any(i.startswith("SAA:") or i.startswith("SBB:") for i in gap_notes)
