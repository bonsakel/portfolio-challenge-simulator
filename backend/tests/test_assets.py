import pytest

from app.assets import SymbolError, SymbolIndex, build_index, normalize_symbol
from directory_fixtures import NASDAQ, OTHER


@pytest.fixture
def index(tmp_path) -> SymbolIndex:
    info = build_index(tmp_path, NASDAQ, OTHER)
    assert info.count == 4
    return SymbolIndex(tmp_path)


def test_normalize_symbol():
    assert normalize_symbol(" brk.b ") == "BRK-B"
    for bad in ["", "  ", "AA PL", "123", "A;B", "TOOLONGSYM", "../etc"]:
        with pytest.raises(SymbolError):
            normalize_symbol(bad)


def test_build_excludes_test_issues_warrants_preferreds_and_footer(index):
    symbols = set(index._load()["symbol"])
    assert symbols == {"AAPL", "AAPX", "BRK-B", "XLK"}


def test_exchange_and_type_mapping(index):
    assert index.get("XLK") == {
        "symbol": "XLK",
        "name": "State Street Technology Select Sector SPDR ETF",
        "exchange": "NYSE Arca",
        "asset_type": "ETF",
    }
    assert index.get("AAPL")["exchange"] == "NASDAQ"
    assert index.get("NOPE") is None


def test_search_ranking_and_no_results(index):
    assert [r["symbol"] for r in index.search("aapl")] == ["AAPL"]
    assert [r["symbol"] for r in index.search("apple")][0] == "AAPL"
    assert [r["symbol"] for r in index.search("brk.b")] == ["BRK-B"]
    assert [r["symbol"] for r in index.search("technology select")] == ["XLK"]
    assert index.search("zzzzqq") == []
    assert len(index.search("a", limit=1)) == 1


def test_search_treats_regex_characters_literally(index):
    assert index.search(".*") == []
    assert index.search("(") == []


def test_info_has_provenance(index):
    info = index.info()
    assert info.available and info.retrieved_at and len(info.sources) == 2
