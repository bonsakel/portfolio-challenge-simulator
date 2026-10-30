from datetime import date
from math import isclose

import pandas as pd

from src.data.download import configured_date_range, load_config, prepare_aligned_data


def test_configured_date_range_includes_rolling_window_warmup() -> None:
    config = load_config()

    start, end = configured_date_range(config, warmup_trading_days=252)

    assert start < date(2004, 4, 1)
    assert end == date(2024, 10, 13)


def test_prepare_aligned_data_does_not_turn_missing_prices_into_multiday_returns() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    prices = {
        "ETF_A": pd.Series([100.0, None, 110.0, 121.0], index=dates, name="ETF_A"),
        "ETF_B": pd.Series([100.0, 110.0, 121.0, 133.1], index=dates, name="ETF_B"),
    }
    vix = pd.Series([20.0, 21.0, 22.0, 23.0], index=dates, name="VIX")

    combined, aligned, returns = prepare_aligned_data(prices, vix, ["ETF_A", "ETF_B"])

    assert len(combined) == 4
    assert aligned.index.tolist() == [dates[0], dates[2], dates[3]]
    assert returns.index.tolist() == [dates[3]]
    assert isclose(returns.loc[dates[3], "ETF_A"], 0.1)
    assert isclose(returns.loc[dates[3], "ETF_B"], 0.1)


def test_vix_only_dates_do_not_drop_the_next_etf_daily_return() -> None:
    etf_dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    vix_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = {
        "ETF_A": pd.Series([100.0, 110.0], index=etf_dates, name="ETF_A"),
        "ETF_B": pd.Series([100.0, 105.0], index=etf_dates, name="ETF_B"),
    }
    vix = pd.Series([20.0, 21.0, 22.0], index=vix_dates, name="VIX")

    _, aligned, returns = prepare_aligned_data(prices, vix, ["ETF_A", "ETF_B"])

    assert aligned.index.tolist() == etf_dates.tolist()
    assert returns.index.tolist() == [etf_dates[1]]
    assert isclose(returns.loc[etf_dates[1], "ETF_A"], 0.1)
