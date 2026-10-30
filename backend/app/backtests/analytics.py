"""Pure analytics on a backtest's daily results: drawdowns, recovery, downside risk, benchmark.

Conventions (also shown in the UI):
* Annualization uses 252 trading days, the same as the research metrics.
* The wealth path starts at the initial capital on the trading day before the first return, so a
  loss on the first day counts as a drawdown.
* VaR / CVaR are historical (empirical) on daily net returns at 95%: VaR is the 5th percentile
  (linear interpolation) and CVaR the mean of returns at or below it. Both are reported as returns,
  so losses are negative. They are not computed for fewer than MIN_TAIL_OBSERVATIONS days.
* Sortino uses the risk-free rate as the target: mean excess return divided by the downside
  deviation sqrt(mean(min(r - rf, 0)^2)) over all days, annualized by sqrt(252).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TRADING_DAYS_PER_YEAR = 252
VAR_LEVEL = 0.95
MIN_TAIL_OBSERVATIONS = 20


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def wealth_path(values: list[float], initial: float) -> np.ndarray:
    return np.concatenate(([float(initial)], np.asarray(values, dtype=float)))


def drawdown_series(values: list[float], initial: float) -> list[float]:
    """Drawdown from the running peak (including the initial capital) at each date; <= 0."""
    path = wealth_path(values, initial)
    return (path / np.maximum.accumulate(path) - 1.0)[1:].tolist()


def value_from_returns(returns: list[float], initial: float) -> list[float]:
    return (float(initial) * np.cumprod(1.0 + np.asarray(returns, dtype=float))).tolist()


def downside_metrics(net_returns: list[float], risk_free_daily: float = 0.0) -> dict[str, Any]:
    r = np.asarray(net_returns, dtype=float)
    excess = r - risk_free_daily
    downside_dev = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))) if len(r) else 0.0
    sortino = (
        float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_dev > 0 else None
    )
    var = cvar = None
    if len(r) >= MIN_TAIL_OBSERVATIONS:
        var = float(np.quantile(r, 1.0 - VAR_LEVEL))
        cvar = float(r[r <= var].mean())
    return {
        "sortino_ratio": sortino,
        "var_95_daily": var,
        "cvar_95_daily": cvar,
        "worst_day": float(r.min()) if len(r) else None,
    }


def recovery_analysis(dates: list[str], values: list[float], initial: float, base_date: str) -> dict[str, Any]:
    """Recovery from the maximum drawdown trough back to the peak that preceded it.

    Reference definition: the peak is the highest value reached before the trough (the initial
    capital counts). Recovery time is the number of trading days from the trough until the value
    first reaches that peak again; if it never does within the window the status is
    "not_recovered". Recovery return is the return from the trough to the end of the window.
    """
    path = wealth_path(values, initial)
    all_dates = [base_date, *dates]
    peaks = np.maximum.accumulate(path)
    drawdowns = path / peaks - 1.0
    trough = int(np.argmin(drawdowns))
    if drawdowns[trough] >= 0:
        return {"status": "no_drawdown", "max_drawdown": 0.0}
    peak_index = int(np.argmax(path[: trough + 1]))
    peak_value = float(path[peak_index])
    after = np.nonzero(path[trough + 1 :] >= peak_value)[0]
    out: dict[str, Any] = {
        "status": "recovered" if len(after) else "not_recovered",
        "max_drawdown": float(drawdowns[trough]),
        "peak_date": all_dates[peak_index],
        "peak_value": peak_value,
        "trough_date": all_dates[trough],
        "trough_value": float(path[trough]),
        "trading_days_peak_to_trough": trough - peak_index,
        "recovery_return_from_trough": float(path[-1] / path[trough] - 1.0),
        "recovery_date": None,
        "recovery_trading_days": None,
    }
    if len(after):
        first = trough + 1 + int(after[0])
        out["recovery_date"] = all_dates[first]
        out["recovery_trading_days"] = first - trough
    return out


def performance_metrics(net_returns: list[float], initial: float, risk_free_daily: float = 0.0) -> dict[str, Any]:
    """Same definitions as the research `period_metrics`, for series without an engine result."""
    r = np.asarray(net_returns, dtype=float)
    if len(r) == 0 or (r <= -1.0).any() or not np.isfinite(r).all():
        raise ValueError("Returns must be finite and greater than -100%.")
    growth = float(np.prod(1.0 + r))
    count = len(r)
    excess = r - risk_free_daily
    excess_vol = float(excess.std(ddof=1)) if count > 1 else 0.0
    path = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    return {
        "trading_days": count,
        "total_return": growth - 1.0,
        "annualized_return": growth ** (TRADING_DAYS_PER_YEAR / count) - 1.0,
        "annualized_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if count > 1 else 0.0,
        "sharpe_ratio": _finite(excess.mean() / excess_vol * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess_vol > 0
        else None,
        "maximum_drawdown": float((path / np.maximum.accumulate(path) - 1.0).min()),
        "ending_value": float(initial) * growth,
    }
