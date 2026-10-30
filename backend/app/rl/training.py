"""Background training of PPO/DDPG models in a separate process.

Training never runs inside a request handler or the API process: the manager prepares the inputs,
starts `app.rl.train_worker` as a subprocess and watches it. That makes cancellation reliable (the
process is terminated) and keeps heavy numerical work away from the server. One job runs at a
time. Progress is read from the validation log that the research trainer writes every
`eval_freq` steps, so it reflects real training steps rather than a timer.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from app.backtests.runner import prices_to_returns
from app.config import PROJECT_ROOT
from app.market_data.service import MarketDataService, align_prices
from app.models import TrainingJobRow
from app.rl.inference import RLError, load_vix, prepare_inputs
from app.schemas import utcnow

log = logging.getLogger(__name__)

BACKEND_DIR = PROJECT_ROOT / "backend"
MIN_TRAINING_DAYS = 252  # the environment samples 252-day episodes
# Measured on the development machine (CPU); a rough guide shown as an estimate, never as progress.
APPROX_STEPS_PER_SECOND = {"PPO": 1500.0, "DDPG": 300.0}


def estimate_seconds(algorithm: str, timesteps: int) -> int:
    return int(timesteps / APPROX_STEPS_PER_SECOND[algorithm])


def mark_interrupted_training(engine: Engine) -> None:
    with Session(engine) as session:
        session.execute(
            update(TrainingJobRow)
            .where(TrainingJobRow.status.in_(("queued", "running")))
            .values(status="failed", finished_at=utcnow(),
                    error="Interrupted: the backend restarted before training finished.")
        )
        session.commit()


class TrainingManager:
    def __init__(self, engine: Engine, market_data: MarketDataService, data_dir: Path) -> None:
        self._engine = engine
        self._market_data = market_data
        self._models_dir = data_dir / "models"
        self._jobs_dir = data_dir / "training"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
        self._processes: dict[int, subprocess.Popen] = {}
        self._cancelled: set[int] = set()
        self._progress: dict[int, float] = {}
        self._lock = threading.Lock()

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def submit(self, job_id: int) -> None:
        self._executor.submit(self._run, job_id)

    def progress(self, job_id: int) -> float | None:
        with self._lock:
            return self._progress.get(job_id)

    def active(self) -> bool:
        with Session(self._engine) as session:
            return session.query(TrainingJobRow).filter(
                TrainingJobRow.status.in_(("queued", "running"))
            ).count() > 0

    def cancel(self, job_id: int) -> bool:
        """Stop a queued or running job. Returns False if it had already finished."""
        with Session(self._engine) as session:
            row = session.get(TrainingJobRow, job_id)
            if row is None or row.status not in ("queued", "running"):
                return False
            with self._lock:
                self._cancelled.add(job_id)
                process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                process.terminate()
            elif row.status == "queued":
                row.status, row.finished_at, row.error = "cancelled", utcnow(), "Cancelled before it started."
                session.commit()
            return True

    def _set(self, job_id: int, **values) -> None:
        with Session(self._engine) as session:
            session.execute(update(TrainingJobRow).where(TrainingJobRow.id == job_id).values(**values))
            session.commit()

    def _run(self, job_id: int) -> None:
        with Session(self._engine) as session:
            row = session.get(TrainingJobRow, job_id)
            if row is None or row.status != "queued":
                return
            config, algorithm = dict(row.config), row.algorithm
        self._set(job_id, status="running", started_at=utcnow())
        job_dir = self._jobs_dir / str(job_id)
        output_dir = self._models_dir / f"job-{job_id}"
        try:
            spec = self._prepare(job_id, algorithm, config, job_dir, output_dir)
            outcome = self._execute(job_id, job_dir, output_dir, spec["total_timesteps"])
        except RLError as exc:
            outcome = ("failed", str(exc))
        except Exception as exc:  # shown to the user, never swallowed
            log.exception("Training job %s crashed", job_id)
            outcome = ("failed", f"Unexpected error: {type(exc).__name__}: {exc}")
        status, message = outcome
        if status != "succeeded":
            shutil.rmtree(output_dir, ignore_errors=True)
        self._set(
            job_id, status=status, error=message if status != "succeeded" else None,
            model_id=f"user/job-{job_id}" if status == "succeeded" else None, finished_at=utcnow(),
        )
        with self._lock:
            self._progress[job_id] = 1.0 if status == "succeeded" else self._progress.get(job_id, 0.0)
            self._processes.pop(job_id, None)

    def _prepare(self, job_id: int, algorithm: str, config: dict, job_dir: Path, output_dir: Path) -> dict:
        symbols: list[str] = config["symbols"]
        train_start, train_end = config["training_start"], config["training_end"]
        val_start, val_end = config["validation_start"], config["validation_end"]
        report, loaded = self._market_data.coverage(symbols, None, _date(val_end))
        bad = [s for s in report.symbols if s.status != "ok"]
        if bad:
            raise RLError("Price history is unavailable: " + "; ".join(
                f"{s.symbol}: {s.message or s.status}" for s in bad))
        prices, _ = align_prices({s: loaded[s] for s in symbols}, None, _date(val_end))
        returns = prices_to_returns(prices)[symbols]
        returns, vix = prepare_inputs(returns, load_vix(), _date(train_start), _date(val_end))
        train_days = int(((returns.index >= train_start) & (returns.index <= train_end)).sum())
        val_days = int(((returns.index >= val_start) & (returns.index <= val_end)).sum())
        if train_days < MIN_TRAINING_DAYS:
            raise RLError(f"The training period has {train_days} trading days; at least {MIN_TRAINING_DAYS} are needed.")
        if val_days < 20:
            raise RLError(f"The validation period has {val_days} trading days; at least 20 are needed.")
        vix = vix.reindex(returns.index)

        job_dir.mkdir(parents=True, exist_ok=True)
        returns.to_csv(job_dir / "returns.csv", index_label="Date")
        vix.rename("VIX").to_csv(job_dir / "vix.csv", index_label="Date")
        spec = {
            "algorithm": algorithm,
            "universe": symbols,
            "max_weight": config["max_weight"],
            "transaction_cost_rate": config["transaction_cost_rate"],
            "initial_capital": config["initial_capital"],
            "seed": config["seed"],
            "training_period": [train_start, train_end],
            "validation_period": [val_start, val_end],
            "total_timesteps": config["total_timesteps"],
            "eval_freq": config["eval_freq"],
            "output_dir": str(output_dir),
            "price_field": self._market_data.provider.price_field,
        }
        (job_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return spec

    def _execute(self, job_id: int, job_dir: Path, output_dir: Path, total: int) -> tuple[str, str | None]:
        log_file = (job_dir / "worker.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "app.rl.train_worker", str(job_dir)],
            cwd=BACKEND_DIR, stdout=log_file, stderr=subprocess.STDOUT,
        )
        with self._lock:
            self._processes[job_id] = process
            if job_id in self._cancelled:
                process.terminate()
        try:
            while process.poll() is None:
                self._update_progress(job_id, output_dir, total)
                time.sleep(0.5)
        finally:
            log_file.close()
        if job_id in self._cancelled:
            return "cancelled", "Cancelled by the user. No model was saved."
        if process.returncode != 0:
            tail = (job_dir / "worker.log").read_text(encoding="utf-8").strip().splitlines()[-4:]
            return "failed", "Training failed: " + " | ".join(tail)
        if not (output_dir / "metadata.json").is_file():
            return "failed", "Training finished without saving model metadata."
        return "succeeded", None

    def _update_progress(self, job_id: int, output_dir: Path, total: int) -> None:
        try:
            steps = int(np.load(output_dir / "evaluation" / "evaluations.npz")["timesteps"][-1])
        except (OSError, ValueError, KeyError, IndexError):
            return
        with self._lock:
            self._progress[job_id] = min(steps / total, 0.99)


def _date(value: str):
    from datetime import date

    return date.fromisoformat(value)
