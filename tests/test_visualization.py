from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult
from src.visualization.plots import plot_all_drawdowns, plot_event_values


def test_comparison_plots_are_written(
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    daily = pd.DataFrame(
        {
            "portfolio_value": [98.0, 101.0, 99.0],
            "net_return": [-0.02, 101 / 98 - 1, 99 / 101 - 1],
            "transaction_cost": [1.0, 0.0, 0.0],
        },
        index=dates,
    )
    result = BacktestResult(daily, 100.0, 99.0, 1.0)
    event_path = tmp_path / "event.png"
    drawdown_path = tmp_path / "drawdowns.png"

    plot_event_values({"equal_weight": result}, event_name="test", output_path=event_path)
    plot_all_drawdowns(
        {"test": {"equal_weight": result}},
        output_path=drawdown_path,
    )

    assert event_path.is_file() and event_path.stat().st_size > 0
    assert drawdown_path.is_file() and drawdown_path.stat().st_size > 0
