import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import load_config
from src.experiments.run_experiment import STRATEGIES, run_experiment


def test_full_smoke_experiment_writes_daily_results_metrics_and_plots(
    tmp_path: Path,
) -> None:
    config = load_config()
    dates = pd.bdate_range("2000-01-03", periods=80)
    assets = config["assets"]
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    vix = pd.Series(20.0 + rng.normal(0.0, 1.0, len(dates)), index=dates, name="VIX")
    config["events"] = {
        "smoke": {
            "training": [dates[30].date().isoformat(), dates[45].date().isoformat()],
            "validation": [dates[46].date().isoformat(), dates[50].date().isoformat()],
            "downturn_test": [dates[51].date().isoformat(), dates[55].date().isoformat()],
            "recovery_observation": [
                dates[56].date().isoformat(),
                dates[60].date().isoformat(),
            ],
        }
    }
    config["reinforcement_learning"]["episode_length_trading_days"] = 4
    config["reinforcement_learning"]["validation"]["eval_freq"] = 4
    config["reinforcement_learning"]["training"]["ppo"].update(
        {"total_timesteps": 8, "n_steps": 4, "batch_size": 4}
    )
    config["reinforcement_learning"]["training"]["ddpg"].update(
        {
            "total_timesteps": 8,
            "buffer_size": 32,
            "learning_starts": 2,
            "batch_size": 4,
        }
    )
    config["traditional_strategies"]["mean_variance"]["lookback_trading_days"] = 10
    config["traditional_strategies"]["minimum_variance"]["lookback_trading_days"] = 10
    config["traditional_strategies"]["mean_variance"]["risk_free_rate_daily"] = 0.0
    (tmp_path / "configs").mkdir()
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "configs/experiment.yaml",
        tmp_path / "configs/experiment.yaml",
    )

    summary = run_experiment(
        config=config,
        returns=returns,
        vix=vix,
        root=tmp_path,
        total_timesteps=8,
        eval_freq=4,
    )

    assert len(summary) == len(STRATEGIES) * 2
    assert set(summary["strategy"]) == set(STRATEGIES)
    assert set(summary["section"]) == {"downturn", "recovery"}
    assert (tmp_path / "results/tables/summary_metrics.csv").is_file()
    assert (tmp_path / "results/tables/recovery_metrics.csv").is_file()
    assert (tmp_path / "results/figures/portfolio_values_smoke.png").is_file()
    assert (tmp_path / "results/figures/drawdowns_all_events.png").is_file()
    assert (tmp_path / "reports/experiment_summary.md").is_file()
    metadata = json.loads((tmp_path / "results/run_metadata.json").read_text())
    assert metadata["strategies"] == list(STRATEGIES)
    for strategy in STRATEGIES:
        path = tmp_path / f"results/tables/daily_smoke_{strategy.lower()}.csv"
        assert pd.read_csv(path).shape[0] == 10
