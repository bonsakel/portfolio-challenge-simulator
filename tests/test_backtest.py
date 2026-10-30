from math import isclose, log

import pandas as pd
import pytest

from src.backtest.engine import equal_weights, run_backtest


def test_backtest_uses_only_prior_returns_and_applies_cost_once() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame(
        {"A": [0.10, 0.0], "B": [0.0, 0.0]},
        index=dates,
    )
    observations: list[tuple[pd.Timestamp, list[pd.Timestamp]]] = []

    def policy(
        decision_date: pd.Timestamp,
        history: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        observations.append((decision_date, history.index.tolist()))
        return pd.Series({"A": 1.0, "B": 0.0})

    result = run_backtest(
        returns,
        policy,
        start=dates[0],
        end=dates[-1],
        initial_value=1_000.0,
        transaction_cost_rate=0.001,
        max_weight=1.0,
    )

    assert observations == [(dates[0], []), (dates[1], [dates[0]])]
    assert isclose(result.daily.iloc[0]["turnover"], 1.0)
    assert isclose(result.daily.iloc[0]["traded_value"], 1_000.0)
    assert isclose(result.daily.iloc[0]["transaction_cost"], 1.0)
    assert isclose(result.daily.iloc[0]["portfolio_value"], 999.0 * 1.10)
    assert isclose(result.daily.iloc[0]["net_return"], 0.0989)
    assert isclose(result.daily.iloc[0]["log_reward"], log(1.0989))
    assert isclose(result.total_transaction_costs, 1.0)
    assert isclose(result.final_value, result.daily.iloc[-1]["portfolio_value"])


def test_monthly_rebalance_runs_policy_only_on_first_available_date_each_month() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    calls: list[pd.Timestamp] = []

    def policy(
        decision_date: pd.Timestamp,
        history: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        calls.append(decision_date)
        return pd.Series({"A": 0.5, "B": 0.5})

    result = run_backtest(
        returns,
        policy,
        start=dates[0],
        end=dates[-1],
        max_weight=0.5,
        rebalance="monthly",
    )

    assert calls == [dates[0], dates[2]]
    assert result.daily["rebalance"].tolist() == [False, False, False]


def test_turnover_below_threshold_does_not_trade() -> None:
    dates = pd.to_datetime(["2024-01-02"])
    returns = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=dates)

    result = run_backtest(
        returns,
        lambda *_: pd.Series({"A": 0.51, "B": 0.49}),
        start=dates[0],
        end=dates[0],
        max_weight=0.6,
        turnover_threshold=0.03,
    )

    assert result.daily.iloc[0]["turnover"] == 0.0
    assert result.daily.iloc[0]["transaction_cost"] == 0.0
    assert not result.daily.iloc[0]["rebalance"]


def test_turnover_threshold_does_not_leave_drifted_weights_over_the_cap() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame({"A": [0.01, 0.0], "B": [0.0, 0.0]}, index=dates)
    target = pd.Series({"A": 0.6, "B": 0.4})

    result = run_backtest(
        returns,
        lambda *_: target,
        start=dates[0],
        end=dates[-1],
        initial_weights=target,
        max_weight=0.6,
        turnover_threshold=0.02,
    )

    assert result.daily.iloc[0]["ending_weight_A"] > 0.6
    assert result.daily.iloc[1]["rebalance"]
    assert result.daily.iloc[1]["weight_A"] == pytest.approx(0.6)


def test_invalid_weights_are_rejected() -> None:
    dates = pd.to_datetime(["2024-01-02"])
    returns = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=dates)

    with pytest.raises(ValueError, match="sum to 1"):
        run_backtest(
            returns,
            lambda *_: pd.Series({"A": 0.2, "B": 0.2}),
            start=dates[0],
            end=dates[0],
            max_weight=1.0,
        )


def test_equal_weights_rejects_duplicate_or_empty_assets() -> None:
    assert equal_weights(["A", "B"]).to_dict() == {"A": 0.5, "B": 0.5}
    with pytest.raises(ValueError):
        equal_weights([])
    with pytest.raises(ValueError):
        equal_weights(["A", "A"])
