"""Reproducible performance and recovery statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252


def _period_rows(
    result: BacktestResult, start: str, end: str
) -> pd.DataFrame:
    rows = result.daily.loc[start:end]
    if rows.empty:
        raise ValueError(f"No backtest observations in period {start} through {end}")
    return rows


def period_metrics(
    result: BacktestResult,
    *,
    start: str,
    end: str,
    risk_free_rate_daily: float = 0.0,
) -> dict[str, Any]:
    """Calculate daily return statistics over an inclusive date interval."""
    if not np.isfinite(risk_free_rate_daily):
        raise ValueError("Daily risk-free rate must be finite")
    rows = _period_rows(result, start, end)
    daily_returns = rows["net_return"].astype(float)
    if (daily_returns <= -1.0).any():
        raise ValueError("Net portfolio returns must be greater than -100%")
    growth = float(np.prod(1.0 + daily_returns.to_numpy()))
    if growth <= 0 or not np.isfinite(growth):
        raise ArithmeticError("Period portfolio growth must be finite and positive")
    count = len(rows)
    annualized_return = growth ** (TRADING_DAYS_PER_YEAR / count) - 1.0
    volatility = (
        float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if count > 1
        else 0.0
    )
    excess = daily_returns - risk_free_rate_daily
    excess_volatility = float(excess.std(ddof=1)) if count > 1 else 0.0
    sharpe = (
        float(excess.mean() / excess_volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess_volatility > 0
        else float("nan")
    )
    wealth_path = np.concatenate(([1.0], np.cumprod(1.0 + daily_returns.to_numpy())))
    running_peak = np.maximum.accumulate(wealth_path)
    drawdowns = wealth_path / running_peak - 1.0
    return {
        "period_start": rows.index.min().date().isoformat(),
        "period_end": rows.index.max().date().isoformat(),
        "trading_days": count,
        "total_return": growth - 1.0,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdowns.min()),
        "transaction_costs": float(rows["transaction_cost"].sum()),
        "ending_value": float(rows["portfolio_value"].iloc[-1]),
    }


def recovery_metrics(
    result: BacktestResult,
    *,
    downturn_start: str,
    downturn_end: str,
    recovery_start: str,
    recovery_end: str,
) -> dict[str, Any]:
    """Measure recovery from the downturn trough to the prior-peak reference.

    The reference peak is the portfolio value at the start of the downturn
    backtest (the initial capital). The drawdown trough is the lowest daily
    closing portfolio value during the downturn. Recovery return measures
    value change from that trough through the final available recovery date.
    """
    downturn = _period_rows(result, downturn_start, downturn_end)
    recovery = _period_rows(result, recovery_start, recovery_end)
    trough_date = downturn["portfolio_value"].idxmin()
    trough_value = float(downturn.loc[trough_date, "portfolio_value"])
    ending_recovery_value = float(recovery["portfolio_value"].iloc[-1])
    reference_peak = float(result.initial_value)
    recovery_path = result.daily.loc[recovery_start:recovery_end, "portfolio_value"]
    recovered = recovery_path[recovery_path >= reference_peak]
    recovery_period_growth = float(
        np.prod(1.0 + recovery["net_return"].to_numpy(dtype=float))
    )
    if recovered.empty:
        recovery_time: int | str = "not recovered within 24 months"
        recovery_date: str | None = None
    else:
        recovery_date = recovered.index[0].date().isoformat()
        recovery_time = int(
            result.daily.index.get_loc(recovered.index[0])
            - result.daily.index.get_loc(trough_date)
        )
    return {
        "downturn_trough_date": trough_date.date().isoformat(),
        "downturn_trough_value": trough_value,
        "recovery_period_start": recovery.index.min().date().isoformat(),
        "recovery_period_end": recovery.index.max().date().isoformat(),
        "reference_peak_value": reference_peak,
        "recovery_return_from_trough": ending_recovery_value / trough_value - 1.0,
        "recovery_period_return": recovery_period_growth - 1.0,
        "recovered_within_24_months": not recovered.empty,
        "recovery_time_trading_days": recovery_time,
        "recovery_date": recovery_date,
        "recovery_ending_value": ending_recovery_value,
    }
