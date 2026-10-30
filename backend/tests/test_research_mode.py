"""Research Mode: a locked, read-only view computed from the stored experiment artifacts."""

import hashlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.research import DownturnRow, RecoveryRow, _findings
from app.config import PROJECT_ROOT, get_settings
from app.main import create_app

SUMMARY = PROJECT_ROOT / "results/tables/summary_metrics.csv"
needs_research = pytest.mark.skipif(
    not (SUMMARY.exists() and (PROJECT_ROOT / "results/models/covid_crash/ppo").exists()),
    reason="research artifacts absent",
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _down(event, strategy, ret, sharpe=0.0):
    return DownturnRow(
        event=event, strategy=strategy, period_start="a", period_end="b", trading_days=10,
        total_return=ret, annualized_return=0, annualized_volatility=0, sharpe_ratio=sharpe,
        maximum_drawdown=0, transaction_costs=0,
    )


def _rec(event, strategy, crossed, end):
    return RecoveryRow(
        event=event, strategy=strategy, trough_date="x", trough_value=1, recovery_return_from_trough=0,
        recovered_within_24_months=crossed, recovery_time_trading_days=None, recovery_date=None,
        ending_value=end,
    )


def test_findings_are_computed_from_the_rows_not_stored():
    down = [
        _down("covid_crash", "equal_weight", -0.3, -1),
        _down("covid_crash", "ppo", -0.1, -0.5),
        _down("2022_downturn", "equal_weight", -0.05),
        _down("2022_downturn", "ppo", -0.2),
    ]
    rec = [_rec("covid_crash", "ppo", True, 90_000), _rec("covid_crash", "equal_weight", True, 120_000)]
    texts = " ".join(f.text + " " + f.evidence for f in _findings(down, rec))
    assert "No strategy had the best downturn total return in every event" in texts
    assert "PPO or DDPG had the best downturn total return in covid crash" in texts
    assert "Every strategy lost money" in texts
    assert "PPO (90,000)" in texts and "2 of 2 strategies regained" in texts and "1 ended" in texts


def test_findings_do_not_claim_losses_when_a_strategy_gained():
    findings = _findings([_down("covid_crash", "ppo", 0.02), _down("covid_crash", "ddpg", -0.1)], [])
    assert any("1 of 2 strategy/event runs lost money" in f.text for f in findings)
    assert not any("Every strategy lost money" in f.text for f in findings)


@needs_research
def test_overview_matches_the_artifacts_and_config(client):
    body = client.get("/api/v1/research/overview").json()
    frame = pd.read_csv(SUMMARY)
    assert body["locked"] is True
    assert len(body["downturn"]) == len(frame[frame.section == "downturn"]) == 15
    assert len(body["recovery"]) == len(frame[frame.section == "recovery"]) == 15
    assert [e for e in body["events"]] == ["global_financial_crisis", "covid_crash", "2022_downturn"]
    s = body["settings"]
    assert s["assets"] == ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
    assert (s["initial_capital"], s["max_weight"], s["transaction_cost_rate"]) == (100000, 0.3, 0.001)
    assert s["mean_variance"]["turnover_threshold"] == 0.02 and s["observation_size"] == 191
    row = next(r for r in body["downturn"] if r["event"] == "covid_crash" and r["strategy"] == "ppo")
    src = frame[(frame.event == "covid_crash") & (frame.strategy == "PPO") & (frame.section == "downturn")].iloc[0]
    assert row["total_return"] == pytest.approx(src.total_return)
    assert body["findings"] and body["caveats"]


def test_overview_without_artifacts_is_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "results_dir", tmp_path)
    assert client.get("/api/v1/research/overview").status_code == 404


def test_unknown_event_is_404(client):
    assert client.post("/api/v1/research/verify/not_an_event").status_code == 404


@needs_research
def test_verify_recomputes_all_five_strategies_and_never_writes(client):
    digest = lambda: hashlib.sha256(SUMMARY.read_bytes()).hexdigest()  # noqa: E731
    before = digest()
    body = client.post("/api/v1/research/verify/covid_crash").json()
    assert body["all_match"] is True, [c for c in body["checks"] if not c["matches"]]
    assert {c["strategy"] for c in body["checks"]} == {
        "equal_weight", "mean_variance", "minimum_variance", "ppo", "ddpg"}
    assert len(body["checks"]) == 5 * 6
    assert digest() == before
