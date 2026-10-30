import numpy as np
import pandas as pd
import pytest

from app.backtests.analytics import (
    downside_metrics,
    drawdown_series,
    performance_metrics,
    recovery_analysis,
    value_from_returns,
)

DATES = ["d1", "d2", "d3", "d4", "d5"]


def test_drawdown_series_includes_initial_capital_as_peak():
    assert drawdown_series([90.0, 95.0], 100.0) == pytest.approx([-0.10, -0.05])
    dd = drawdown_series([110, 99, 88, 100, 121], 100)
    assert dd == pytest.approx([0, -0.1, -0.2, 100 / 110 - 1, 0])


def test_recovered_drawdown():
    r = recovery_analysis(DATES, [110, 99, 88, 100, 121], 100, "d0")
    assert r["status"] == "recovered"
    assert (r["peak_date"], r["trough_date"], r["recovery_date"]) == ("d1", "d3", "d5")
    assert r["peak_value"] == 110 and r["trough_value"] == 88
    assert r["recovery_trading_days"] == 2 and r["trading_days_peak_to_trough"] == 2
    assert r["max_drawdown"] == pytest.approx(-0.2)
    assert r["recovery_return_from_trough"] == pytest.approx(121 / 88 - 1)


def test_not_recovered_drawdown():
    r = recovery_analysis(DATES[:4], [110, 99, 88, 100], 100, "d0")
    assert r["status"] == "not_recovered"
    assert r["recovery_date"] is None and r["recovery_trading_days"] is None
    assert r["recovery_return_from_trough"] == pytest.approx(100 / 88 - 1)


def test_no_drawdown():
    assert recovery_analysis(["d1", "d2"], [101, 102], 100, "d0") == {"status": "no_drawdown", "max_drawdown": 0.0}


def test_loss_on_first_day_counts_and_peak_is_the_start():
    r = recovery_analysis(["d1", "d2", "d3"], [90, 95, 101], 100, "d0")
    assert r["peak_date"] == "d0" and r["trough_date"] == "d1" and r["recovery_date"] == "d3"
    assert r["recovery_trading_days"] == 2


def test_downside_metrics_hand_calculated():
    returns = [-0.02, 0.01, 0.03, -0.01] * 10  # 40 observations
    d = downside_metrics(returns, 0.0)
    r = np.array(returns)
    dev = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert d["sortino_ratio"] == pytest.approx(r.mean() / dev * np.sqrt(252))
    var = np.quantile(r, 0.05)
    assert d["var_95_daily"] == pytest.approx(var)
    assert d["cvar_95_daily"] == pytest.approx(r[r <= var].mean()) and d["cvar_95_daily"] <= var + 1e-12
    assert d["worst_day"] == -0.02


def test_downside_metrics_degenerate_cases():
    assert downside_metrics([0.01] * 30)["sortino_ratio"] is None  # no downside at all
    short = downside_metrics([-0.01, 0.02, 0.01])
    assert short["var_95_daily"] is None and short["cvar_95_daily"] is None  # too few observations


def test_performance_metrics_match_the_research_implementation():
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import period_metrics
    from src.strategies.traditional import equal_weight_policy

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (300, 3)), index=idx, columns=["A", "B", "C"])
    result = run_backtest(
        returns, equal_weight_policy, start=idx[0], end=idx[-1], initial_value=100_000,
        initial_weights=None, transaction_cost_rate=0.001, max_weight=0.5, rebalance="monthly",
        turnover_threshold=0.0,
    )
    research = period_metrics(
        result, start=str(idx[0].date()), end=str(idx[-1].date()), risk_free_rate_daily=0.0001
    )
    mine = performance_metrics(result.daily["net_return"].tolist(), 100_000, 0.0001)
    for key in ("total_return", "annualized_return", "annualized_volatility", "sharpe_ratio",
                "maximum_drawdown", "ending_value", "trading_days"):
        assert mine[key] == pytest.approx(research[key], rel=1e-9), key
    assert value_from_returns(result.daily["net_return"].tolist(), 100_000)[-1] == pytest.approx(
        result.daily["portfolio_value"].iloc[-1]
    )


def test_performance_metrics_reject_invalid_returns():
    with pytest.raises(ValueError):
        performance_metrics([0.01, -1.0], 100)
    with pytest.raises(ValueError):
        performance_metrics([], 100)
