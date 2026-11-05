"""Research Mode: a locked, read-only view computed from the stored experiment artifacts."""

import hashlib
import json

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


@needs_research
def test_dashboard_returns_only_selected_event_and_strategy(client):
    response = client.get(
        "/api/v1/research/dashboard",
        params={"events": "covid_crash", "strategies": "ppo"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "original_single_seed"
    assert body["seed_count"] == 1
    assert len(body["events"]) == 1
    event = body["events"][0]
    assert event["event"] == "covid_crash"
    assert list(event["strategies"]) == ["ppo"]
    assert len(event["strategies"]["ppo"]["series"]["dates"]) == 529
    assert len(event["strategies"]["ppo"]["series"]["sample_returns"]) == 1
    assert set(event["strategies"]["ppo"]["series"]["weights"]) == {
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"
    }


def test_dashboard_rejects_unknown_selection(client):
    response = client.get(
        "/api/v1/research/dashboard",
        params={"events": "not-an-event", "strategies": "ppo"},
    )
    assert response.status_code == 422


def test_dashboard_uses_complete_seed_study_and_reports_seed_spread(
    client, tmp_path, monkeypatch
):
    results_dir = tmp_path
    table_dir = results_dir / "robustness" / "tables"
    table_dir.mkdir(parents=True)
    seeds = [42, 43, 44, 45, 46]
    (results_dir / "robustness" / "study.json").write_text(
        json.dumps({"complete": True, "seeds": seeds}), encoding="utf-8"
    )
    assets = [
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"
    ]
    for event in ("global_financial_crisis", "covid_crash", "2022_downturn"):
        for strategy in ("ppo", "ddpg"):
            for seed in seeds:
                path = table_dir / f"daily_{event}_{strategy}_seed_{seed}.csv"
                if event == "covid_crash" and strategy == "ppo":
                    frame = pd.DataFrame(
                        {
                            "Date": ["2020-02-19", "2020-03-23"],
                            "portfolio_value": [100_000, 50_000 + seed * 1_000],
                            "net_return": [0.0, -0.3],
                            "transaction_cost": [0.0, float(seed)],
                            "turnover": [0.0, 0.1],
                            **{
                                f"weight_{asset}": [1 / len(assets), 1 / len(assets)]
                                for asset in assets
                            },
                        }
                    )
                    frame.to_csv(path, index=False)
                else:
                    path.write_text("Date\n2020-02-19\n2020-03-23\n", encoding="utf-8")

    rows = []
    for seed in seeds:
        rows.extend(
            [
                {
                    "event": "covid_crash",
                    "strategy": "ppo",
                    "seed": seed,
                    "section": "downturn",
                    "total_return": -0.5 + (seed - 42) * 0.1,
                    "annualized_return": -0.9,
                    "annualized_volatility": 0.8,
                    "sharpe_ratio": -4.1,
                    "maximum_drawdown": -0.6,
                    "transaction_costs": 100 + seed,
                },
                {
                    "event": "covid_crash",
                    "strategy": "ppo",
                    "seed": seed,
                    "section": "recovery",
                    "recovery_return_from_trough": 0.2,
                    "recovery_time_trading_days": "not recovered within 24 months",
                    "recovery_ending_value": 90_000,
                    "recovered_within_24_months": False,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(table_dir / "metrics_by_seed.csv", index=False)
    monkeypatch.setattr(get_settings(), "results_dir", results_dir)

    response = client.get(
        "/api/v1/research/dashboard",
        params={"events": "covid_crash", "strategies": "ppo"},
    )
    assert response.status_code == 200
    strategy = response.json()["events"][0]["strategies"]["ppo"]
    assert response.json()["source"] == "five_seed_study"
    assert strategy["metrics"]["total_return"] == {
        "mean": pytest.approx(-0.3),
        "min": pytest.approx(-0.5),
        "max": pytest.approx(-0.1),
    }
    assert strategy["series"]["seed_count"] == 5
    assert strategy["series"]["portfolio_value"][1] == pytest.approx(94_000)
    assert strategy["series"]["value_min"][1] == pytest.approx(92_000)
    assert strategy["series"]["value_max"][1] == pytest.approx(96_000)
    assert strategy["recovery"]["recovered_count"] == 0
    assert strategy["recovery"]["sample_count"] == 5


def test_pipeline_reports_baseline_and_pending_robustness_run(client, tmp_path, monkeypatch):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "summary_metrics.csv").write_text("event,strategy\n", encoding="utf-8")
    monkeypatch.setattr(get_settings(), "results_dir", tmp_path)

    response = client.get("/api/v1/research/pipeline")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_started"
    assert body["baseline_available"] is True
    assert body["total_models"] == 30
    assert body["completed_models"] == 0
    assert body["steps"][-1]["status"] == "current"


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
