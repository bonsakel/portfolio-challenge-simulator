import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.market_data.provider import ProviderError
from test_market_data import FakeProvider, make_series


@pytest.fixture
def client(tmp_path):
    provider = FakeProvider(
        {"AAA": make_series("2019-01-01", 700), "BBB": make_series("2020-06-01", 300)},
        errors={"ERR": ProviderError("rate limited")},
    )
    app = create_app(Settings(data_dir=tmp_path), provider)
    test_client = TestClient(app)
    test_client.provider = provider  # type: ignore[attr-defined]
    return test_client


def test_status_before_any_download(client):
    body = client.get("/api/v1/data/status").json()
    assert body["provider"]["name"] == "fake" and body["provider"]["real_time"] is False
    assert body["cached_symbols"] == [] and body["cached_range"] is None and body["failures"] == []
    assert "not real-time" in body["disclaimer"]


def test_coverage_downloads_then_serves_from_cache(client):
    first = client.post("/api/v1/data/coverage", json={"symbols": ["aaa", "bbb"]}).json()
    assert first["ready"] and {s["source"] for s in first["symbols"]} == {"downloaded"}
    second = client.post("/api/v1/data/coverage", json={"symbols": ["AAA", "BBB"]}).json()
    assert {s["source"] for s in second["symbols"]} == {"cache"}
    assert client.provider.calls == ["AAA", "BBB"]
    status = client.get("/api/v1/data/status").json()
    assert [c["symbol"] for c in status["cached_symbols"]] == ["AAA", "BBB"]
    assert status["cached_range"][0] == "2019-01-01"
    assert status["cached_symbols"][0]["price_field"] == "Adj Close"


def test_refresh_redownloads(client):
    client.post("/api/v1/data/coverage", json={"symbols": ["AAA"]})
    body = client.post("/api/v1/data/refresh", json={"symbols": ["AAA"]}).json()
    assert body["symbols"][0]["source"] == "downloaded"
    assert client.provider.calls == ["AAA", "AAA"]


def test_failures_surface_in_coverage_and_status(client):
    body = client.post("/api/v1/data/coverage", json={"symbols": ["AAA", "NOPE", "ERR"]}).json()
    assert body["ready"] is False
    assert {s["symbol"]: s["status"] for s in body["symbols"]} == {
        "AAA": "ok",
        "NOPE": "unavailable",
        "ERR": "provider_error",
    }
    failures = {f["symbol"]: f["kind"] for f in client.get("/api/v1/data/status").json()["failures"]}
    assert failures == {"NOPE": "no_data", "ERR": "provider_error"}


def test_clear_cache_requires_confirmation(client):
    client.post("/api/v1/data/coverage", json={"symbols": ["AAA"]})
    assert client.delete("/api/v1/data/cache").status_code == 400
    assert len(client.get("/api/v1/data/status").json()["cached_symbols"]) == 1
    cleared = client.delete("/api/v1/data/cache", params={"confirm": "true"})
    assert cleared.status_code == 200 and cleared.json() == {"cleared_symbols": 1}
    assert client.get("/api/v1/data/status").json()["cached_symbols"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"symbols": []},
        {"symbols": ["bad symbol!"]},
        {"symbols": ["../x"]},
        {"symbols": ["A"] * 26},
        {"symbols": ["AAA"], "start": "2020-05-01", "end": "2020-01-01"},
        {"symbols": ["AAA"], "start": "not-a-date"},
    ],
)
def test_invalid_coverage_requests_rejected(client, body):
    assert client.post("/api/v1/data/coverage", json=body).status_code == 422


def test_coverage_with_date_window_explains_limits(client):
    body = client.post(
        "/api/v1/data/coverage",
        json={"symbols": ["AAA", "BBB"], "start": "2019-01-01", "end": "2021-06-30"},
    ).json()
    assert body["limiting_symbol"] == "BBB" and body["common_start"] == "2020-06-01"
    assert any("before the common history" in i for i in body["issues"])
