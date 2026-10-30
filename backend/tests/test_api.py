import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_cors_allows_local_frontend_only(client: TestClient) -> None:
    ok = client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    other = client.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_research_summary_matches_artifact(client: TestClient) -> None:
    import pandas as pd

    from app.config import get_settings

    csv = pd.read_csv(get_settings().results_dir / "tables" / "summary_metrics.csv")
    expected = csv[csv["section"] == "downturn"]
    response = client.get("/api/v1/research/summary")
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == len(expected) == 15
    gfc_mv = next(
        r for r in rows if r["event"] == "global_financial_crisis" and r["strategy"] == "minimum_variance"
    )
    source = expected[
        (expected.event == "global_financial_crisis") & (expected.strategy == "minimum_variance")
    ].iloc[0]
    assert gfc_mv["total_return"] == pytest.approx(source["total_return"])


def test_research_summary_missing_artifacts_returns_404(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "results_dir", tmp_path)
    assert client.get("/api/v1/research/summary").status_code == 404
