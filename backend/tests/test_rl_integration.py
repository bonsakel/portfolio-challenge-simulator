"""PPO/DDPG integration: compatibility gating, exact reproduction of the research results, and
background training. The reproduction and training tests run real Stable-Baselines3 code."""

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.main import create_app
from test_backtests_api import client, make_portfolio, random_walk, run, wait  # noqa: F401
from test_market_data import FakeProvider

ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
RESEARCH_PRICES = PROJECT_ROOT / "data/processed/daily_returns/adjusted_close.csv"
SUMMARY = PROJECT_ROOT / "results/tables/summary_metrics.csv"
needs_research = pytest.mark.skipif(
    not (RESEARCH_PRICES.is_file() and SUMMARY.is_file() and (PROJECT_ROOT / "results/models").is_dir()),
    reason="research artifacts are not present",
)


def make_client(tmp_path, data):
    c = TestClient(create_app(Settings(data_dir=tmp_path), FakeProvider(data)))
    return c


@pytest.fixture
def research_client(tmp_path):
    prices = pd.read_csv(RESEARCH_PRICES, index_col="Date", parse_dates=True)
    return make_client(tmp_path, {s: prices[s] for s in ETFS})


@pytest.fixture
def synthetic_client(tmp_path):
    # 9 random-walk ETFs on the VIX file's trading days.
    vix = pd.read_csv(PROJECT_ROOT / "data/processed/daily_returns/vix.csv", index_col="Date", parse_dates=True)
    days = vix.loc["2010-01-04":].index[:1500]
    data = {}
    for i, s in enumerate(ETFS):
        walk = random_walk(i, periods=len(days), start="2010-01-04")
        data[s] = pd.Series(walk.to_numpy(), index=days)
    return make_client(tmp_path, data)


def test_models_endpoint_lists_research_models_and_flags_a_different_universe(client):
    pid = make_portfolio(client)  # four assets
    body = client.get("/api/v1/models", params={"portfolio_id": pid, "start": "2020-02-19", "end": "2020-03-23"}).json()
    if not body["models"]:
        pytest.skip("research models are not present")
    assert len(body["models"]) >= 6 and body["vix_range"][0] == "2003-03-03"
    for model in body["models"]:
        verdict = model["compatibility"]
        assert verdict["compatible"] is False and "Different asset universe" in verdict["issues"][0]


def test_rl_strategy_requires_an_explicit_model(client):
    pid = make_portfolio(client)
    r = run(client, pid, strategies=["ppo"])
    assert r.status_code == 422 and "trained model for PPO" in r.text


def test_rl_strategy_with_incompatible_universe_is_rejected_before_running(client):
    pid = make_portfolio(client)
    r = run(client, pid, strategies=["ppo"], rl_models={"ppo": "research/covid_crash/ppo"})
    if r.status_code == 422 and "not an available" in r.text:
        pytest.skip("research models are not present")
    assert r.status_code == 422 and "Different asset universe" in r.text


def test_wrong_algorithm_for_model_is_rejected(research_client):
    pid = make_portfolio(research_client, symbols=ETFS)
    r = run(research_client, pid, strategies=["ppo"], rl_models={"ppo": "research/covid_crash/ddpg"},
            start="2020-02-19", end="2020-03-23")
    assert r.status_code == 422 and "not an available PPO" in r.text


@needs_research
def test_in_sample_window_is_blocked(research_client):
    pid = make_portfolio(research_client, symbols=ETFS)
    r = run(research_client, pid, strategies=["ppo"], rl_models={"ppo": "research/covid_crash/ppo"},
            start="2017-01-03", end="2017-12-29")
    assert r.status_code == 422 and "in-sample" in r.text


@needs_research
def test_different_cap_is_blocked(research_client):
    pid = make_portfolio(research_client, symbols=ETFS, max_weight=0.5)
    r = run(research_client, pid, strategies=["ddpg"], rl_models={"ddpg": "research/covid_crash/ddpg"},
            start="2020-02-19", end="2020-03-23")
    assert r.status_code == 422 and "position cap" in r.text


@needs_research
def test_rl_backtest_reproduces_the_research_results_exactly(research_client):
    summary = pd.read_csv(SUMMARY)
    pid = make_portfolio(research_client, symbols=ETFS, initial_capital=100_000)
    started = run(
        research_client, pid, strategies=["ppo", "ddpg", "equal_weight"],
        rl_models={"ppo": "research/covid_crash/ppo", "ddpg": "research/covid_crash/ddpg"},
        start="2020-02-19", end="2022-03-23",
    )
    assert started.status_code == 202, started.text
    body = wait(research_client, started.json()["id"], timeout=240)
    assert body["status"] == "succeeded", body["error"]
    for key in ("ppo", "ddpg"):
        got = body["summary"]["strategies"][key]
        assert got["model"]["id"] == f"research/covid_crash/{key}" and got["rebalance"] == "daily"
        assert got["trading_days"] == 529
    results = research_client.get(f"/api/v1/backtests/{started.json()['id']}/results").json()
    assert results["series"]["ppo"]["portfolio_value"][-1] == pytest.approx(93707.10800848585, rel=1e-9)
    assert results["series"]["ddpg"]["portfolio_value"][-1] == pytest.approx(120737.850484323, rel=1e-9)
    # Weights respect the cap and are fully invested on every day.
    for key in ("ppo", "ddpg"):
        weights = np.array([results["series"][key]["weights"][s] for s in ETFS])
        assert weights.max() <= 0.3 + 1e-9 and np.allclose(weights.sum(axis=0), 1.0)


def test_training_requires_explicit_confirmation(synthetic_client):
    pid = make_portfolio(synthetic_client, symbols=ETFS)
    body = dict(portfolio_id=pid, algorithm="PPO", training_start="2010-03-01", training_end="2011-12-30",
                validation_start="2012-01-03", validation_end="2012-06-29", total_timesteps=5000)
    assert synthetic_client.post("/api/v1/training-jobs", json=body).status_code == 422
    assert synthetic_client.post("/api/v1/training-jobs", json={**body, "confirm": False}).status_code == 422
    assert synthetic_client.get("/api/v1/training-jobs").json() == []


def test_training_validates_dates_and_universe(synthetic_client, client):
    pid = make_portfolio(synthetic_client, symbols=ETFS)
    base = dict(portfolio_id=pid, algorithm="PPO", total_timesteps=5000, confirm=True,
                training_start="2010-03-01", training_end="2011-12-30",
                validation_start="2012-01-03", validation_end="2012-06-29")
    out_of_order = {**base, "validation_start": "2011-06-01"}
    assert synthetic_client.post("/api/v1/training-jobs", json=out_of_order).status_code == 422
    assert synthetic_client.post("/api/v1/training-jobs", json={**base, "total_timesteps": 10}).status_code == 422
    four = make_portfolio(client)
    r = client.post("/api/v1/training-jobs", json={**base, "portfolio_id": four})
    assert r.status_code == 422 and "exactly 9 assets" in r.json()["detail"]


def test_training_job_that_cannot_run_fails_with_a_clear_message(synthetic_client):
    pid = make_portfolio(synthetic_client, symbols=ETFS)
    r = synthetic_client.post("/api/v1/training-jobs", json=dict(
        portfolio_id=pid, algorithm="PPO", total_timesteps=5000, confirm=True,
        training_start="2010-03-01", training_end="2010-06-30",  # far fewer than 252 trading days
        validation_start="2010-07-01", validation_end="2010-12-31"))
    assert r.status_code == 202
    job = wait_job(synthetic_client, r.json()["id"])
    assert job["status"] == "failed" and "at least 252" in job["error"]


def wait_job(client, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/training-jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.3)
    raise AssertionError("training did not finish")


def test_train_then_backtest_a_custom_model_end_to_end(synthetic_client, tmp_path):
    pid = make_portfolio(synthetic_client, symbols=ETFS)
    body = dict(portfolio_id=pid, algorithm="PPO", total_timesteps=5000, confirm=True,
                training_start="2010-03-01", training_end="2011-12-30",
                validation_start="2012-01-03", validation_end="2012-06-29")
    r = synthetic_client.post("/api/v1/training-jobs", json=body)
    assert r.status_code == 202 and r.json()["estimated_seconds"] > 0
    # A second job cannot start while one is active.
    assert synthetic_client.post("/api/v1/training-jobs", json=body).status_code == 409
    job = wait_job(synthetic_client, r.json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert job["progress"] == 1.0 and job["model_id"] == f"user/job-{job['id']}"

    import json
    meta = json.loads((tmp_path / "models" / f"job-{job['id']}" / "metadata.json").read_text())
    assert meta["seed"] == 42 and meta["training_period"] == ["2010-03-01", "2011-12-30"]
    assert meta["environment"]["universe"] == ETFS and meta["environment"]["observation_size"] == 191
    assert meta["environment"]["max_weight"] == 0.3 and meta["normalization"]["statistics_fitted_on"] == "training_only"
    assert "stable-baselines3" in meta["package_versions"]

    models = synthetic_client.get("/api/v1/models", params={
        "portfolio_id": pid, "start": "2012-09-03", "end": "2012-12-31"}).json()["models"]
    mine = next(m for m in models if m["id"] == job["model_id"])
    assert mine["source"] == "user" and mine["compatibility"]["compatible"] is True

    # Validation data is not a test: a window inside it is refused.
    bad = run(synthetic_client, pid, strategies=["ppo"], rl_models={"ppo": job["model_id"]},
              start="2012-02-01", end="2012-05-01")
    assert bad.status_code == 422 and "in-sample" in bad.text

    ok = run(synthetic_client, pid, strategies=["ppo", "equal_weight"], rl_models={"ppo": job["model_id"]},
             start="2012-09-03", end="2012-12-31")
    assert ok.status_code == 202, ok.text
    result = wait(synthetic_client, ok.json()["id"], timeout=120)
    assert result["status"] == "succeeded", result["error"]
    assert result["summary"]["strategies"]["ppo"]["model"]["id"] == job["model_id"]


def test_running_training_job_can_be_cancelled_and_leaves_no_model(synthetic_client, tmp_path):
    pid = make_portfolio(synthetic_client, symbols=ETFS)
    r = synthetic_client.post("/api/v1/training-jobs", json=dict(
        portfolio_id=pid, algorithm="DDPG", total_timesteps=300_000, confirm=True,
        training_start="2010-03-01", training_end="2011-12-30",
        validation_start="2012-01-03", validation_end="2012-06-29"))
    job_id = r.json()["id"]
    deadline = time.time() + 60
    while time.time() < deadline and synthetic_client.get(f"/api/v1/training-jobs/{job_id}").json()["status"] != "running":
        time.sleep(0.2)
    time.sleep(3)  # let the worker process start
    assert synthetic_client.post(f"/api/v1/training-jobs/{job_id}/cancel").status_code == 200
    job = wait_job(synthetic_client, job_id, timeout=60)
    assert job["status"] == "cancelled" and job["model_id"] is None
    assert not (tmp_path / "models" / f"job-{job_id}").exists()
    assert synthetic_client.post(f"/api/v1/training-jobs/{job_id}/cancel").status_code == 409
    assert synthetic_client.post("/api/v1/training-jobs/999/cancel").status_code == 404
