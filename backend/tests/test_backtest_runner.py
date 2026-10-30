from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtests.runner import (
    BacktestError,
    RunConfig,
    check_window,
    prices_to_returns,
    run_strategies,
)
from app.config import PROJECT_ROOT


def make_returns(n=300, seed=0, assets=("A", "B", "C", "D")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (n, len(assets))), index=idx, columns=list(assets))


def cfg(returns, **kw):
    base = dict(
        strategies=("equal_weight",),
        start=returns.index[0].date(),
        end=returns.index[-1].date(),
        initial_capital=100_000.0,
        max_weight=0.5,
    )
    base.update(kw)
    return RunConfig(**base)


def test_prices_to_returns_does_not_fill_and_rejects_bad_prices():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-06"])
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=idx)
    r = prices_to_returns(prices)
    assert r["A"].tolist() == pytest.approx([0.10, -0.10])
    with pytest.raises(BacktestError):
        prices_to_returns(prices.iloc[:1])
    with pytest.raises(BacktestError):
        prices_to_returns(prices.assign(A=[100.0, 0.0, 99.0]))


def test_hand_calculated_equal_weight_accounting():
    # Initial equal weights, daily rebalance to equal weights, 10 bp fee.
    idx = pd.bdate_range("2021-01-04", periods=3)
    returns = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=idx)
    config = cfg(returns, max_weight=1.0, rebalance="daily", transaction_cost_rate=0.001)
    run = run_strategies(returns, config)["equal_weight"]

    # Day 1: start 50/50 -> no trade, fee 0; value = 100000 * (1 + 0.05).
    assert run.series["portfolio_value"][0] == pytest.approx(105_000.0)
    assert run.series["transaction_cost"][0] == 0.0
    # Day 1 drift: A = 0.5*1.1/1.05, B = 0.5/1.05. Day 2 rebalance back to 50/50.
    a_end = 0.5 * 1.1 / 1.05
    turnover = abs(0.5 - a_end) + abs(0.5 - (1 - a_end))
    fee = 0.001 * 105_000.0 * turnover
    assert run.series["turnover"][1] == pytest.approx(turnover)
    assert run.series["transaction_cost"][1] == pytest.approx(fee)
    assert run.series["portfolio_value"][1] == pytest.approx(105_000.0 - fee)
    assert run.summary["transaction_costs"] == pytest.approx(fee)
    assert run.summary["total_turnover"] == pytest.approx(turnover)
    assert run.summary["ending_value"] == pytest.approx(105_000.0 - fee)


def test_weights_sum_to_one_and_respect_cap_for_all_strategies():
    returns = make_returns()
    config = cfg(
        returns,
        strategies=("equal_weight", "mean_variance", "minimum_variance"),
        start=returns.index[260].date(),
        max_weight=0.4,
        lookback_days=252,
    )
    runs = run_strategies(returns, config)
    for run in runs.values():
        w = pd.DataFrame(run.series["weights"])
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-7)
        assert (w.min() >= -1e-8).all() and (w.max() <= 0.4 + 1e-7).all()
    assert runs["equal_weight"].rebalance == "monthly"
    assert runs["minimum_variance"].turnover_threshold == 0.02


def test_optimizer_decisions_ignore_the_decision_days_own_return():
    returns = make_returns()
    start = returns.index[260]
    config = cfg(returns, strategies=("minimum_variance",), start=start.date(), max_weight=0.5)
    base = run_strategies(returns, config)["minimum_variance"]

    # A huge shock *on* the first decision day must not change that day's weights.
    shocked = returns.copy()
    shocked.loc[start] = [0.5, -0.5, 0.3, -0.3]
    again = run_strategies(shocked, config)["minimum_variance"]
    for asset in returns.columns:
        assert again.series["weights"][asset][0] == pytest.approx(base.series["weights"][asset][0])
    # ...but it does change that day's outcome.
    assert again.series["portfolio_value"][0] != pytest.approx(base.series["portfolio_value"][0])


def test_insufficient_warmup_is_a_clear_error_with_a_suggested_start():
    returns = make_returns(300)
    config = cfg(returns, strategies=("minimum_variance",), start=returns.index[100].date())
    with pytest.raises(BacktestError, match="252 daily returns.*only 100.*on or after"):
        check_window(returns, config)
    # Equal weight needs no warm-up.
    check_window(returns, cfg(returns, start=returns.index[100].date()))


def test_window_errors():
    returns = make_returns(50)
    with pytest.raises(BacktestError, match="before the end"):
        check_window(returns, cfg(returns, start=date(2020, 3, 1), end=date(2020, 2, 1)))
    with pytest.raises(BacktestError, match="Fewer than 2"):
        check_window(returns, cfg(returns, start=date(2030, 1, 1), end=date(2030, 2, 1)))


def test_unknown_strategy_and_infeasible_cap_are_rejected():
    returns = make_returns(60)
    with pytest.raises(BacktestError, match="Unknown strategy"):
        run_strategies(returns, cfg(returns, strategies=("magic",)))
    with pytest.raises(BacktestError, match="failed"):
        run_strategies(returns, cfg(returns, max_weight=0.2))  # 4 assets * 0.2 < 1


def test_progress_is_monotonic_and_completes():
    returns = make_returns(80)
    seen: list[float] = []
    run_strategies(returns, cfg(returns, strategies=("equal_weight",), rebalance="daily"), seen.append)
    assert seen[-1] == 1.0 and seen == sorted(seen)


RESEARCH_RETURNS = PROJECT_ROOT / "data/processed/daily_returns/daily_returns.csv"
RESEARCH_SUMMARY = PROJECT_ROOT / "results/tables/summary_metrics.csv"


@pytest.mark.skipif(
    not (RESEARCH_RETURNS.exists() and RESEARCH_SUMMARY.exists()), reason="research artifacts absent"
)
@pytest.mark.parametrize(
    "event,start,end,strategies",
    [
        ("covid_crash", "2020-02-19", "2020-03-23", ("equal_weight", "minimum_variance", "mean_variance")),
        ("global_financial_crisis", "2007-10-09", "2009-03-09", ("equal_weight", "minimum_variance")),
    ],
)
def test_reproduces_the_research_results_exactly(event, start, end, strategies):
    returns = pd.read_csv(RESEARCH_RETURNS, index_col=0, parse_dates=True)
    summary = pd.read_csv(RESEARCH_SUMMARY)
    config = RunConfig(
        strategies=strategies,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        initial_capital=100_000.0,
        max_weight=0.30,
    )
    runs = run_strategies(returns, config)
    for strategy in strategies:
        row = summary[
            (summary.event == event) & (summary.strategy == strategy) & (summary.section == "downturn")
        ].iloc[0]
        got = runs[strategy].summary
        assert got["total_return"] == pytest.approx(row.total_return, rel=1e-9)
        assert got["transaction_costs"] == pytest.approx(row.transaction_costs, rel=1e-9)
        assert got["maximum_drawdown"] == pytest.approx(row.maximum_drawdown, rel=1e-9)
        assert got["sharpe_ratio"] == pytest.approx(row.sharpe_ratio, rel=1e-9)
