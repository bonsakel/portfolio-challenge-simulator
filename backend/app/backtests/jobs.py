"""Background execution and persistence of backtests.

A single worker thread runs jobs one at a time; the API returns immediately and the UI polls
for status and progress. Progress is held in memory only (it is meaningless after a restart),
so jobs left unfinished by a previous process are marked failed on startup.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from app.backtests.analytics import (
    downside_metrics,
    drawdown_series,
    performance_metrics,
    recovery_analysis,
    value_from_returns,
)
from app.backtests.runner import (
    RL_STRATEGIES,
    BacktestError,
    RunConfig,
    prices_to_returns,
    run_strategies,
)
from app.market_data.service import MarketDataService, align_prices
from app.models import BacktestRow
from app.rl.inference import RLError, load_vix, run_agent
from app.rl.registry import ModelRegistry, check_compatibility
from app.schemas import utcnow

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Historical simulation on a selected universe of currently listed securities (not "
    "survivorship-free). Simplified cost model: a proportional fee on traded value only; no "
    "spreads, market impact, taxes or liquidity limits. Past results do not predict future "
    "performance and this is not investment advice."
)


def config_from_json(raw: dict) -> RunConfig:
    return RunConfig(
        strategies=tuple(raw["strategies"]),
        start=date.fromisoformat(raw["start"]),
        end=date.fromisoformat(raw["end"]),
        initial_capital=raw["initial_capital"],
        max_weight=raw["max_weight"],
        lookback_days=raw["lookback_days"],
        transaction_cost_rate=raw["transaction_cost_rate"],
        rebalance=raw["rebalance"],
        turnover_threshold=raw["turnover_threshold"],
        risk_free_rate_annual=raw["risk_free_rate_annual"],
        initial_weights=raw["initial_weights"],
    )


def mark_interrupted(engine: Engine) -> None:
    with Session(engine) as session:
        session.execute(
            update(BacktestRow)
            .where(BacktestRow.status.in_(("queued", "running")))
            .values(status="failed", error="Interrupted: the backend restarted before this run finished.",
                    finished_at=utcnow())
        )
        session.commit()


class JobManager:
    def __init__(
        self, engine: Engine, market_data: MarketDataService, registry: ModelRegistry | None = None
    ) -> None:
        self._engine = engine
        self._market_data = market_data
        self._registry = registry
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest")
        self._progress: dict[int, float] = {}
        self._lock = threading.Lock()

    def submit(self, backtest_id: int) -> None:
        self._set_progress(backtest_id, 0.0)
        self._executor.submit(self._run, backtest_id)

    def progress(self, backtest_id: int) -> float | None:
        with self._lock:
            return self._progress.get(backtest_id)

    def _set_progress(self, backtest_id: int, value: float) -> None:
        with self._lock:
            self._progress[backtest_id] = value

    def _run(self, backtest_id: int) -> None:
        with Session(self._engine) as session:
            row = session.get(BacktestRow, backtest_id)
            if row is None:
                return
            row.status = "running"
            session.commit()
            try:
                summary, result = self._execute(row, backtest_id)
            except BacktestError as exc:
                row.status, row.error = "failed", str(exc)
            except Exception as exc:  # surfaced to the user, never swallowed
                log.exception("Backtest %s crashed", backtest_id)
                row.status, row.error = "failed", f"Unexpected error: {type(exc).__name__}: {exc}"
            else:
                row.status, row.summary, row.result = "succeeded", summary, result
            row.finished_at = utcnow()
            session.commit()
        self._set_progress(backtest_id, 1.0)

    def _execute(self, row: BacktestRow, backtest_id: int) -> tuple[dict, dict]:
        config = config_from_json(row.config)
        symbols: list[str] = row.config["symbols"]

        report, loaded = self._market_data.coverage(symbols, None, config.end)
        bad = [s for s in report.symbols if s.status != "ok"]
        if bad:
            detail = "; ".join(f"{s.symbol}: {s.message or s.status}" for s in bad)
            raise BacktestError(f"Price history is unavailable, so the backtest cannot run. {detail}")

        prices, dropped = align_prices({s: loaded[s] for s in symbols}, None, config.end)
        returns = prices_to_returns(prices)[symbols]
        agent_runner = self._agent_runner(row, config, symbols, returns)
        runs = run_strategies(
            returns, config, lambda p: self._set_progress(backtest_id, min(p, 0.99)), agent_runner
        )

        first = next(iter(runs.values())).series["dates"]
        first_position = prices.index.get_loc(pd.Timestamp(first[0]))
        base_date = prices.index[first_position - 1].date().isoformat()
        for run in runs.values():
            run.series["drawdown"] = drawdown_series(run.series["portfolio_value"], config.initial_capital)
            run.summary["downside"] = downside_metrics(run.series["net_return"], config.risk_free_rate_daily)
            run.summary["recovery"] = recovery_analysis(
                run.series["dates"], run.series["portfolio_value"], config.initial_capital, base_date
            )
        benchmark_summary, benchmark_series = self._benchmark(row, config, first, base_date)
        data = {
            "symbols": [
                {"symbol": s.symbol, "source": s.source, "first_date": s.first_date,
                 "last_date": s.last_date, "rows": s.rows}
                for s in report.symbols
            ],
            "common_start": prices.index[0].date().isoformat(),
            "common_end": prices.index[-1].date().isoformat(),
            "common_price_rows": int(len(prices)),
            "dates_dropped_per_symbol": dropped,
            "notes": list(report.issues),
            "price_field": self._market_data.provider.price_field,
            "provider": self._market_data.provider.name,
        }
        period = {
            "requested_start": row.config["start"],
            "requested_end": row.config["end"],
            "first_return_date": first[0],
            "last_return_date": first[-1],
            "trading_days": len(first),
        }
        summary = {
            "data": data,
            "period": period,
            "base_date": base_date,
            "benchmark": benchmark_summary,
            "disclaimer": DISCLAIMER,
            "strategies": {
                k: {"label": r.label, "rebalance": r.rebalance,
                    "turnover_threshold": r.turnover_threshold, "model": r.model, **r.summary}
                for k, r in runs.items()
            },
        }
        result = {"series": {k: r.series for k, r in runs.items()}, "benchmark": benchmark_series}
        return summary, result

    def _agent_runner(self, row: BacktestRow, config: RunConfig, symbols: list[str], returns):
        """Build the callable that evaluates PPO/DDPG, or None when no agent was requested.

        Each agent must name a model; the compatibility check runs again here (not only in the
        API) so an incompatible or in-sample model can never be used by accident.
        """
        wanted = [s for s in config.strategies if s in RL_STRATEGIES]
        if not wanted:
            return None
        if self._registry is None:
            raise BacktestError("PPO/DDPG are unavailable: no model registry is configured.")
        chosen: dict[str, str] = row.config.get("rl_models") or {}
        weights = config.initial_weights
        equal_start = weights is None or (
            len(set(round(v, 9) for v in weights.values())) == 1
        )
        try:
            vix = load_vix()
        except RLError as exc:
            raise BacktestError(str(exc)) from exc

        def run(strategy: str, progress):
            model_id = chosen.get(strategy)
            if not model_id:
                raise BacktestError(f"Select a trained model for {strategy.upper()}.")
            spec = self._registry.get(model_id)
            if spec is None or spec.algorithm.lower() != strategy:
                raise BacktestError(f"Model {model_id!r} was not found for {strategy.upper()}.")
            window = returns.loc[pd.Timestamp(config.start):pd.Timestamp(config.end)].index
            if len(window) == 0:
                raise BacktestError("No trading days fall inside the requested window.")
            verdict = check_compatibility(
                spec,
                assets=symbols,
                max_weight=config.max_weight,
                transaction_cost_rate=config.transaction_cost_rate,
                first_decision=window[0].date(),
                last_decision=window[-1].date(),
                equal_initial_weights=equal_start,
            )
            if not verdict.compatible:
                raise BacktestError(f"{spec.label} cannot be used here: " + " ".join(verdict.issues))
            try:
                result = run_agent(
                    spec, returns, vix, start=config.start, end=config.end,
                    initial_capital=config.initial_capital,
                    transaction_cost_rate=config.transaction_cost_rate,
                    max_weight=config.max_weight, progress=progress,
                )
            except RLError as exc:
                raise BacktestError(str(exc)) from exc
            return result, {**spec.to_json(), "warnings": verdict.warnings}

        return run

    def _benchmark(
        self, row: BacktestRow, config: RunConfig, dates: list[str], base_date: str
    ) -> tuple[dict | None, dict | None]:
        """Buy-and-hold of one symbol over the same dates, starting from the same capital."""
        symbol = row.config.get("benchmark")
        if not symbol:
            return None, None
        report, loaded = self._market_data.coverage([symbol], None, config.end)
        status = report.symbols[0]
        if status.status != "ok":
            raise BacktestError(f"Benchmark {symbol} is unavailable: {status.message or status.status}")
        series = loaded[symbol]
        wanted = pd.DatetimeIndex([pd.Timestamp(base_date), *[pd.Timestamp(d) for d in dates]])
        missing = wanted.difference(series.index)
        if len(missing):
            raise BacktestError(
                f"Benchmark {symbol} has no price on {len(missing)} of the {len(wanted)} dates used "
                f"(first missing: {missing[0].date()}); choose a benchmark with a longer history."
            )
        prices = series.loc[wanted].astype(float)
        returns = prices.pct_change().iloc[1:].tolist()
        values = value_from_returns(returns, config.initial_capital)
        summary = {
            "symbol": symbol,
            "source": status.source,
            "label": f"{symbol} (buy and hold)",
            **performance_metrics(returns, config.initial_capital, config.risk_free_rate_daily),
            "initial_value": config.initial_capital,
            "downside": downside_metrics(returns, config.risk_free_rate_daily),
            "recovery": recovery_analysis(dates, values, config.initial_capital, base_date),
        }
        return summary, {
            "dates": dates,
            "portfolio_value": values,
            "net_return": returns,
            "drawdown": drawdown_series(values, config.initial_capital),
        }
