import pytest
from fastapi.testclient import TestClient

from app.assets import build_index
from app.config import Settings
from app.main import create_app
from directory_fixtures import NASDAQ, OTHER


def make_client(tmp_path, with_index: bool = True) -> TestClient:
    if with_index:
        build_index(tmp_path, NASDAQ, OTHER)
    return TestClient(create_app(Settings(data_dir=tmp_path)))


@pytest.fixture
def client(tmp_path) -> TestClient:
    return make_client(tmp_path)


def portfolio(**overrides):
    body = {
        "name": "Test",
        "initial_capital": 100000,
        "max_weight": 1.0,
        "weight_mode": "equal",
        "holdings": [{"symbol": "aapl"}, {"symbol": "xlk"}],
    }
    body.update(overrides)
    return body


def test_search_success_no_results_and_validation(client):
    ok = client.get("/api/v1/assets/search", params={"q": "apple"})
    assert ok.status_code == 200
    assert ok.json()["results"][0]["symbol"] == "AAPL"
    assert "NASDAQ Trader" in ok.json()["source"]
    assert client.get("/api/v1/assets/search", params={"q": "zzzzqq"}).json()["results"] == []
    assert client.get("/api/v1/assets/search", params={"q": ""}).status_code == 422
    assert client.get("/api/v1/assets/search", params={"q": "x", "limit": 500}).status_code == 422


def test_search_without_index_returns_503_with_instructions(tmp_path):
    response = make_client(tmp_path, with_index=False).get("/api/v1/assets/search", params={"q": "a"})
    assert response.status_code == 503
    assert "update_symbol_index" in response.json()["detail"]


def test_asset_detail(client):
    body = client.get("/api/v1/assets/brk.b").json()
    assert body["symbol"] == "BRK-B" and body["data_availability"] == "not_checked"
    assert client.get("/api/v1/assets/NOPE").status_code == 404
    assert client.get("/api/v1/assets/A$B").status_code == 422


def test_validate_endpoint_reports_without_saving(client):
    report = client.post("/api/v1/portfolios/validate", json=portfolio(max_weight=0.3)).json()
    assert report["valid"] is False
    assert any(i["code"] == "cap_infeasible" for i in report["issues"])
    assert client.get("/api/v1/portfolios").json() == []


def test_portfolio_crud_round_trip(client):
    created = client.post("/api/v1/portfolios", json=portfolio())
    assert created.status_code == 201
    body = created.json()
    assert [h["symbol"] for h in body["holdings"]] == ["AAPL", "XLK"]
    assert body["validation"]["valid"] is True
    assert body["validation"]["effective_weights"] == {"AAPL": 0.5, "XLK": 0.5}

    pid = body["id"]
    assert client.get(f"/api/v1/portfolios/{pid}").json()["name"] == "Test"
    updated = client.put(f"/api/v1/portfolios/{pid}", json=portfolio(name="Renamed", initial_capital=5000))
    assert updated.json()["name"] == "Renamed" and updated.json()["initial_capital"] == 5000
    assert len(client.get("/api/v1/portfolios").json()) == 1
    assert client.delete(f"/api/v1/portfolios/{pid}").status_code == 204
    assert client.get(f"/api/v1/portfolios/{pid}").status_code == 404
    assert client.delete(f"/api/v1/portfolios/{pid}").status_code == 404


def test_draft_with_issues_can_be_saved_and_reports_them(client):
    saved = client.post("/api/v1/portfolios", json=portfolio(max_weight=0.3)).json()
    assert saved["validation"]["valid"] is False


def test_custom_weights_persist_and_equal_mode_drops_weights(client):
    custom = portfolio(
        weight_mode="custom",
        holdings=[{"symbol": "AAPL", "weight": 0.6}, {"symbol": "XLK", "weight": 0.4}],
    )
    assert client.post("/api/v1/portfolios", json=custom).json()["holdings"][0]["weight"] == 0.6
    equal = portfolio(holdings=[{"symbol": "AAPL", "weight": 0.9}, {"symbol": "XLK"}])
    assert client.post("/api/v1/portfolios", json=equal).json()["holdings"][0]["weight"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "   "},
        {"initial_capital": 0},
        {"initial_capital": -5},
        {"max_weight": 0},
        {"max_weight": 1.5},
        {"holdings": [{"symbol": "bad symbol!"}]},
        {"holdings": [{"symbol": "AAPL", "weight": 7}]},
        {"weight_mode": "random"},
    ],
)
def test_malformed_portfolio_rejected_with_422(client, overrides):
    assert client.post("/api/v1/portfolios", json=portfolio(**overrides)).status_code == 422


def test_data_status_reports_index_provenance(client, tmp_path):
    status = client.get("/api/v1/data/status").json()["symbol_index"]
    assert status["available"] and status["count"] == 4 and status["retrieved_at"]
    empty = make_client(tmp_path / "empty", with_index=False).get("/api/v1/data/status").json()
    assert empty["symbol_index"]["available"] is False
