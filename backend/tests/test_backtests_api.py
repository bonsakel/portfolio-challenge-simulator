import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backtests.jobs import mark_interrupted
from app.config import Settings
from app.main import create_app
from app.models import BacktestRow
from app.schemas import utcnow
from sqlalchemy.orm import Session
from test_market_data import FakeProvider

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def random_walk(seed: int, periods: int = 600, start="2019-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, periods)), index=idx)


@pytest.fixture
def client(tmp_path):
    provider = FakeProvider({**{s: random_walk(i) for i, s in enumerate(SYMBOLS)}, "SPY": random_walk(9)})
    c = TestClient(create_app(Settings(data_dir=tmp_path), provider))
    c.provider = provider  # type: ignore[attr-defined]
    return c


def make_portfolio(client, symbols=SYMBOLS, **kw):
    body = dict(
        name="Test",
        initial_capital=100_000,
        max_weight=0.3,
        weight_mode="equal",
        holdings=[{"symbol": s, "weight": None} for s in symbols],
    )
    body.update(kw)
    r = client.post("/api/v1/portfolios", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def run(client, portfolio_id, **kw):
    body = dict(
        portfolio_id=portfolio_id,
        strategies=["equal_weight", "minimum_variance"],
        start="2020-06-01",
        end="2020-12-31",
    )
    body.update(kw)
    r = client.post("/api/v1/backtests", json=body)
    return r


def wait(client, backtest_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/backtests/{backtest_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("backtest did not finish")


def test_full_flow_results_are_consistent(client):
    pid = make_portfolio(client)
    started = run(client, pid)
    assert started.status_code == 202 and started.json()["status"] in ("queued", "running", "succeeded")
    body = wait(client, started.json()["id"])
    assert body["status"] == "succeeded" and body["error"] is None and body["progress"] == 1.0

    summary = body["summary"]
    assert set(summary["strategies"]) == {"equal_weight", "minimum_variance"}
    assert {s["source"] for s in summary["data"]["symbols"]} == {"downloaded"}
    assert summary["data"]["price_field"] == "Adj Close"
    assert "not investment advice" in summary["disclaimer"]
    assert summary["strategies"]["equal_weight"]["rebalance"] == "monthly"
    assert summary["strategies"]["minimum_variance"]["turnover_threshold"] == 0.02

    results = client.get(f"/api/v1/backtests/{started.json()['id']}/results").json()
    for key, s in results["series"].items():
        m = summary["strategies"][key]
        assert len(s["dates"]) == m["trading_days"]
        assert s["portfolio_value"][-1] == pytest.approx(m["ending_value"])
        assert sum(s["transaction_cost"]) == pytest.approx(m["transaction_costs"])
        assert sum(s["turnover"]) == pytest.approx(m["total_turnover"])
        weights = pd.DataFrame(s["weights"])
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-7) and (weights.max() <= 0.3 + 1e-7).all()
        growth = np.prod(1 + np.array(s["net_return"]))
        assert growth - 1 == pytest.approx(m["total_return"])
        assert s["portfolio_value"][0] == pytest.approx(100_000 * (1 + s["net_return"][0]))
    # Both strategies were compared over identical dates.
    dates = [s["dates"] for s in results["series"].values()]
    assert dates[0] == dates[1]


def test_second_run_uses_the_cache(client):
    pid = make_portfolio(client)
    wait(client, run(client, pid, strategies=["equal_weight"]).json()["id"])
    second = wait(client, run(client, pid, strategies=["equal_weight"]).json()["id"])
    assert {s["source"] for s in second["summary"]["data"]["symbols"]} == {"cache"}
    assert client.provider.calls == SYMBOLS


def test_export_csv(client):
    pid = make_portfolio(client)
    bid = run(client, pid).json()["id"]
    wait(client, bid)
    r = client.get(f"/api/v1/backtests/{bid}/export.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0].split(",")[:7] == [
        "date", "strategy", "portfolio_value", "net_return", "drawdown", "turnover", "transaction_cost"
    ]
    assert lines[0].endswith("weight_AAA,weight_BBB,weight_CCC,weight_DDD")
    results = client.get(f"/api/v1/backtests/{bid}/results").json()
    assert len(lines) - 1 == sum(len(s["dates"]) for s in results["series"].values())


def test_analytics_and_drawdowns_are_consistent(client):
    bid = run(client, make_portfolio(client)).json()["id"]
    body = wait(client, bid)
    results = client.get(f"/api/v1/backtests/{bid}/results").json()
    assert results["benchmark"] is None and body["summary"]["benchmark"] is None
    for key, s in results["series"].items():
        m = body["summary"]["strategies"][key]
        assert min(s["drawdown"]) == pytest.approx(m["maximum_drawdown"])
        assert max(s["drawdown"]) <= 0
        assert m["downside"]["var_95_daily"] <= 0
        assert m["downside"]["cvar_95_daily"] <= m["downside"]["var_95_daily"] + 1e-12
        assert m["recovery"]["max_drawdown"] == pytest.approx(m["maximum_drawdown"])


def test_benchmark_is_buy_and_hold_over_the_same_dates(client):
    pid = make_portfolio(client)
    bid = run(client, pid, benchmark=" spy ").json()["id"]
    body = wait(client, bid)
    assert body["status"] == "succeeded", body["error"]
    results = client.get(f"/api/v1/backtests/{bid}/results").json()
    bench, summary = results["benchmark"], body["summary"]["benchmark"]
    assert summary["symbol"] == "SPY" and summary["initial_value"] == 100_000
    assert bench["dates"] == next(iter(results["series"].values()))["dates"]
    # Independent check against the raw prices: base date close -> last date close.
    prices = random_walk(9)
    base = pd.Timestamp(body["summary"]["base_date"])
    last = pd.Timestamp(bench["dates"][-1])
    assert bench["portfolio_value"][-1] == pytest.approx(100_000 * prices[last] / prices[base])
    assert summary["total_return"] == pytest.approx(prices[last] / prices[base] - 1)
    assert summary["maximum_drawdown"] == pytest.approx(min(bench["drawdown"]))
    csv = client.get(f"/api/v1/backtests/{bid}/export.csv").text
    assert "benchmark_SPY" in csv


def test_unavailable_benchmark_fails_clearly(client):
    body = wait(client, run(client, make_portfolio(client), benchmark="NOPE").json()["id"])
    assert body["status"] == "failed" and "Benchmark NOPE" in body["error"]


def test_benchmark_with_too_short_history_fails(client):
    client.provider.data["LATE"] = random_walk(5, periods=100, start="2020-09-01")
    body = wait(client, run(client, make_portfolio(client), benchmark="LATE").json()["id"])
    assert body["status"] == "failed" and "no price on" in body["error"]


def test_invalid_benchmark_symbol_is_a_validation_error(client):
    r = run(client, make_portfolio(client), benchmark="not a ticker!")
    assert r.status_code == 422


def test_invalid_portfolio_is_rejected_before_running(client):
    pid = make_portfolio(client, symbols=SYMBOLS[:3])  # 3 * 30% < 100%
    r = run(client, pid)
    assert r.status_code == 422 and "not valid" in r.json()["detail"]
    assert client.get("/api/v1/backtests").json() == []


def test_request_validation(client):
    pid = make_portfolio(client)
    assert run(client, 9999).status_code == 404
    assert run(client, pid, start="2020-12-31", end="2020-06-01").status_code == 422
    assert run(client, pid, strategies=["magic"]).status_code == 422
    assert run(client, pid, strategies=["equal_weight", "equal_weight"]).status_code == 422
    assert run(client, pid, transaction_cost_rate=0.5).status_code == 422
    assert run(client, pid, strategies=[]).status_code == 422


def test_unavailable_symbol_fails_clearly_and_has_no_results(client):
    pid = make_portfolio(client, symbols=SYMBOLS + ["ZZZZ"], max_weight=0.3)
    body = wait(client, run(client, pid, strategies=["equal_weight"]).json()["id"])
    assert body["status"] == "failed" and "ZZZZ" in body["error"] and "unavailable" in body["error"]
    assert client.get(f"/api/v1/backtests/{body['id']}/results").status_code == 409
    assert client.get(f"/api/v1/backtests/{body['id']}/export.csv").status_code == 409


def test_insufficient_warmup_fails_with_actionable_message(client):
    pid = make_portfolio(client)
    body = wait(client, run(client, pid, strategies=["minimum_variance"], start="2019-03-01").json()["id"])
    assert body["status"] == "failed" and "252 daily returns" in body["error"] and "on or after" in body["error"]


def test_portfolio_weights_as_initial_allocation(client):
    holdings = [{"symbol": s, "weight": w} for s, w in zip(SYMBOLS, [0.3, 0.3, 0.2, 0.2])]
    pid = make_portfolio(client, weight_mode="custom", holdings=holdings)
    body = wait(client, run(client, pid, strategies=["equal_weight"], initial_allocation="portfolio_weights").json()["id"])
    assert body["status"] == "succeeded"
    assert body["config"]["initial_weights"] == pytest.approx(dict(zip(SYMBOLS, [0.3, 0.3, 0.2, 0.2])))
    # Moving from 30/30/20/20 to 25% each costs fee > 0 on the first day.
    series = client.get(f"/api/v1/backtests/{body['id']}/results").json()["series"]["equal_weight"]
    assert series["transaction_cost"][0] > 0


def test_get_list_delete_and_404s(client):
    pid = make_portfolio(client)
    bid = run(client, pid, strategies=["equal_weight"]).json()["id"]
    wait(client, bid)
    assert [b["id"] for b in client.get("/api/v1/backtests").json()] == [bid]
    assert client.delete(f"/api/v1/backtests/{bid}").status_code == 204
    assert client.get(f"/api/v1/backtests/{bid}").status_code == 404
    assert client.get("/api/v1/backtests/12345/results").status_code == 404


def test_unfinished_jobs_are_marked_failed_on_restart(tmp_path):
    app = create_app(Settings(data_dir=tmp_path), FakeProvider({}))
    engine = app.state.services.engine
    with Session(engine) as session:
        session.add(BacktestRow(portfolio_name="x", status="running", config={}, created_at=utcnow()))
        session.commit()
    mark_interrupted(engine)
    with Session(engine) as session:
        row = session.query(BacktestRow).one()
        assert row.status == "failed" and "restarted" in row.error
