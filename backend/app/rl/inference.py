"""Inference with a frozen, validation-selected PPO or DDPG model.

This mirrors `src.experiments.run_experiment.evaluate_agent` (which stays unchanged): the same
`PortfolioEnv`, frozen `VecNormalize` statistics fitted on the training period only, and
deterministic predictions. No learning happens here. Stable-Baselines3 is imported lazily so the
rest of the backend works without loading it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT
from app.rl.registry import ModelSpec

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestResult  # noqa: E402

RESEARCH_VIX_FILE = PROJECT_ROOT / "data" / "processed" / "daily_returns" / "vix.csv"
RETURN_LOOKBACK = 20


class RLError(ValueError):
    """A model or data problem the user can act on."""


def load_vix(path: Path = RESEARCH_VIX_FILE) -> pd.Series:
    """VIX closes from the research dataset (CBOE history). Coverage is limited to that file."""
    if not path.is_file():
        raise RLError(
            f"The VIX file {path.name} is not available, and PPO/DDPG need it as an input. "
            "Run the research data download first."
        )
    vix = pd.read_csv(path, index_col="Date", parse_dates=True)["VIX"]
    if vix.empty or vix.index.has_duplicates or not vix.index.is_monotonic_increasing:
        raise RLError("The VIX file is empty or not in chronological order.")
    return vix


def prepare_inputs(
    returns: pd.DataFrame, vix: pd.Series, start: date, end: date
) -> tuple[pd.DataFrame, pd.Series]:
    """Trim returns to the span the environment can use and check that VIX covers it.

    The environment needs 21 prior return rows and a VIX value on every return date. Nothing is
    filled or interpolated: a gap is reported.
    """
    trimmed = returns.loc[:pd.Timestamp(end)]
    trimmed = trimmed.loc[trimmed.index >= vix.index.min()]
    in_window = trimmed.index[trimmed.index >= pd.Timestamp(start)]
    if len(in_window) == 0:
        raise RLError(f"No common return dates on or after {start} and not later than {end}.")
    if in_window[-1] > vix.index.max():
        raise RLError(
            f"VIX data ends on {vix.index.max().date()}, so PPO/DDPG cannot be evaluated through "
            f"{in_window[-1].date()}. Choose an end date on or before {vix.index.max().date()}."
        )
    first_pos = trimmed.index.get_loc(in_window[0])
    if first_pos < RETURN_LOOKBACK + 1:
        raise RLError(
            f"PPO/DDPG need {RETURN_LOOKBACK + 1} trading days of history (and VIX) before the first "
            f"decision on {in_window[0].date()}, but only {first_pos} are available."
        )
    trimmed = trimmed.iloc[first_pos - (RETURN_LOOKBACK + 1):]  # only the warm-up rows the agent needs
    missing = trimmed.index.difference(vix.index)
    if len(missing):
        raise RLError(
            f"VIX has no value on {len(missing)} of the return dates (first: {missing[0].date()}), "
            "so the agent's inputs would be incomplete."
        )
    return trimmed, vix


def run_agent(
    spec: ModelSpec,
    returns: pd.DataFrame,
    vix: pd.Series,
    *,
    start: date,
    end: date,
    initial_capital: float,
    transaction_cost_rate: float,
    max_weight: float,
    progress: Callable[[float], None] | None = None,
) -> BacktestResult:
    from stable_baselines3 import DDPG, PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.environment.portfolio_env import PortfolioEnv

    returns, vix = prepare_inputs(returns, vix, start, end)
    env = PortfolioEnv(
        returns,
        vix,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        episode_length=None,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
        max_weight=max_weight,
    )
    expected = int(returns.index.searchsorted(pd.Timestamp(end), side="right")) - int(
        returns.index.searchsorted(pd.Timestamp(start))
    )
    vector_env = DummyVecEnv([lambda: Monitor(env)])
    normalized = VecNormalize.load(str(spec.normalization_file), vector_env)
    normalized.training = False
    normalized.norm_reward = False
    model_class = PPO if spec.algorithm == "PPO" else DDPG
    try:
        model = model_class.load(str(spec.model_file), env=normalized, device="cpu")
    except ValueError as exc:  # SB3 rejects mismatched observation/action spaces
        normalized.close()
        raise RLError(f"{spec.label} is not compatible with this environment: {exc}") from exc

    records: list[dict] = []
    observation = normalized.reset()
    try:
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, dones, infos = normalized.step(action)
            info = infos[0]
            row = {
                "Date": pd.Timestamp(info["date"]),
                "portfolio_value": float(info["portfolio_value"]),
                "gross_return": float(info["gross_return"]),
                "net_return": float(info["net_return"]),
                "turnover": float(info["turnover"]),
                "traded_value": float(info["traded_value"]),
                "transaction_cost": float(info["transaction_cost"]),
                "rebalance": float(info["turnover"]) > 1e-12,
            }
            row.update({f"weight_{a}": float(info["target_weights"][a]) for a in env.assets})
            records.append(row)
            if progress and expected:
                progress(min(len(records) / expected, 1.0))
            if bool(dones[0]):
                break
    finally:
        normalized.close()

    daily = pd.DataFrame(records).set_index("Date").sort_index()
    if not np.isfinite(daily["portfolio_value"].to_numpy()).all() or len(daily) != expected:
        raise RLError(f"{spec.label} produced {len(daily)} steps, expected {expected}.")
    return BacktestResult(
        daily=daily,
        initial_value=float(initial_capital),
        final_value=float(daily["portfolio_value"].iloc[-1]),
        total_transaction_costs=float(daily["transaction_cost"].sum()),
    )
