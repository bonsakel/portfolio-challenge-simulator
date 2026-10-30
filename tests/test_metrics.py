from math import sqrt

import pandas as pd
import pytest

from src.backtest.engine import BacktestResult
from src.backtest.metrics import period_metrics, recovery_metrics


@pytest.fixture
def history() -> BacktestResult:
    dates = pd.bdate_range("2024-01-02", periods=5)
    daily = pd.DataFrame(
        {
            "portfolio_value": [90.0, 80.0, 85.0, 100.0, 110.0],
            "net_return": [-0.10, -1.0 / 9.0, 0.0625, 100.0 / 85.0 - 1.0, 0.10],
            "transaction_cost": [1.0, 0.5, 0.0, 0.2, 0.0],
        },
        index=dates,
    )
    return BacktestResult(
        daily=daily,
        initial_value=100.0,
        final_value=110.0,
        total_transaction_costs=1.7,
    )


def test_period_metrics_include_initial_value_in_drawdown_and_net_costs(
    history: BacktestResult,
) -> None:
    metrics = period_metrics(
        history,
        start="2024-01-02",
        end="2024-01-03",
        risk_free_rate_daily=0.0,
    )

    assert metrics["trading_days"] == 2
    assert metrics["total_return"] == pytest.approx(-0.2)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.2)
    assert metrics["transaction_costs"] == pytest.approx(1.5)
    assert metrics["ending_value"] == pytest.approx(80.0)
    assert metrics["annualized_volatility"] == pytest.approx(
        pd.Series([-0.1, -1 / 9]).std(ddof=1) * sqrt(252)
    )


def test_recovery_metrics_use_starting_capital_reference_peak(
    history: BacktestResult,
) -> None:
    metrics = recovery_metrics(
        history,
        downturn_start="2024-01-02",
        downturn_end="2024-01-03",
        recovery_start="2024-01-04",
        recovery_end="2024-01-08",
    )

    assert metrics["downturn_trough_date"] == "2024-01-03"
    assert metrics["downturn_trough_value"] == 80.0
    assert metrics["reference_peak_value"] == 100.0
    assert metrics["recovered_within_24_months"]
    assert metrics["recovery_date"] == "2024-01-05"
    assert metrics["recovery_time_trading_days"] == 2
    assert metrics["recovery_return_from_trough"] == pytest.approx(110 / 80 - 1)


def test_recovery_metrics_report_not_recovered_within_window(
    history: BacktestResult,
) -> None:
    unrecovered_daily = history.daily.copy()
    unrecovered_daily["portfolio_value"] = [90.0, 80.0, 85.0, 90.0, 95.0]
    unrecovered = BacktestResult(
        daily=unrecovered_daily,
        initial_value=100.0,
        final_value=95.0,
        total_transaction_costs=history.total_transaction_costs,
    )

    metrics = recovery_metrics(
        unrecovered,
        downturn_start="2024-01-02",
        downturn_end="2024-01-03",
        recovery_start="2024-01-04",
        recovery_end="2024-01-08",
    )

    assert not metrics["recovered_within_24_months"]
    assert metrics["recovery_time_trading_days"] == "not recovered within 24 months"
    assert metrics["recovery_date"] is None


def test_period_metrics_reject_missing_period(history: BacktestResult) -> None:
    with pytest.raises(ValueError, match="No backtest observations"):
        period_metrics(history, start="2025-01-01", end="2025-01-31")
