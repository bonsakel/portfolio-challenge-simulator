"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  apiGet,
  apiSend,
  type ModelsResponse,
  type Portfolio,
  type TrainingJob,
} from "@/lib/api";
import { STRATEGIES } from "@/lib/backtest";
import {
  buildTrainingRequest,
  defaultTrainingForm,
  estimateSeconds,
  formatDuration,
  type TrainingForm,
} from "@/lib/training";

const inputClass = "w-full rounded border border-line bg-bg px-3 py-2 text-sm tnum";
const ACTIVE = ["queued", "running"];

export function StrategyLab({ pollMs = 1500 }: { pollMs?: number }) {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [form, setForm] = useState<TrainingForm>(() => defaultTrainingForm());
  const [formError, setFormError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [understood, setUnderstood] = useState(false);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [m, j] = await Promise.all([apiGet<ModelsResponse>("/models"), apiGet<TrainingJob[]>("/training-jobs?limit=10")]);
      setModels(m);
      setJobs(j);
    } catch (e) {
      setLoadError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
    apiGet<Portfolio[]>("/portfolios").then(setPortfolios).catch(() => setPortfolios([]));
  }, [refresh]);

  const active = jobs.find((j) => ACTIVE.includes(j.status)) ?? null;
  const activeId = active?.id ?? null;
  useEffect(() => {
    if (activeId === null) return;
    let stop = false;
    const tick = async () => {
      try {
        const job = await apiGet<TrainingJob>(`/training-jobs/${activeId}`);
        if (stop) return;
        setJobs((all) => all.map((j) => (j.id === job.id ? job : j)));
        if (ACTIVE.includes(job.status)) timer.current = setTimeout(tick, pollMs);
        else void refresh();
      } catch {
        if (!stop) timer.current = setTimeout(tick, pollMs * 3);
      }
    };
    timer.current = setTimeout(tick, pollMs);
    return () => {
      stop = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [activeId, pollMs, refresh]);

  const nineAssets = portfolios.filter((p) => p.holdings.length === 9);
  const selected = portfolios.find((p) => p.id === form.portfolioId) ?? null;
  const steps = Number(form.timesteps);
  const estimate = Number.isFinite(steps) && steps > 0 ? formatDuration(estimateSeconds(form.algorithm, steps)) : null;

  function update(patch: Partial<TrainingForm>) {
    setForm((f) => ({ ...f, ...patch }));
    setFormError(null);
    setConfirming(false);
    setUnderstood(false);
  }

  function review() {
    const built = buildTrainingRequest(form, false);
    if ("error" in built) return setFormError(built.error);
    if (selected && !selected.validation.valid) return setFormError("The selected portfolio has validation errors.");
    setFormError(null);
    setConfirming(true);
  }

  async function startTraining() {
    const built = buildTrainingRequest(form, true);
    if ("error" in built) return setFormError(built.error);
    setBusy(true);
    try {
      const job = await apiSend<TrainingJob>("POST", "/training-jobs", built.request);
      setJobs((all) => [job, ...all]);
      setConfirming(false);
      setUnderstood(false);
    } catch (e) {
      setFormError((e as Error).message);
      setConfirming(false);
    } finally {
      setBusy(false);
    }
  }

  async function cancel(job: TrainingJob) {
    try {
      const next = await apiSend<TrainingJob>("POST", `/training-jobs/${job.id}/cancel`);
      setJobs((all) => all.map((j) => (j.id === next.id ? next : j)));
    } catch (e) {
      setFormError((e as Error).message);
    }
  }

  if (loadError)
    return (
      <p role="alert" className="text-sm text-loss">
        Could not load models: {loadError}
      </p>
    );

  return (
    <div className="flex flex-col gap-6">
      <section aria-labelledby="strategies-heading" className="rounded border border-line bg-panel p-4">
        <h2 id="strategies-heading" className="font-medium mb-3">
          Strategies
        </h2>
        <ul className="flex flex-col gap-3 text-sm">
          {STRATEGIES.map((s) => (
            <li key={s.key}>
              <span className="font-medium">{s.name}</span>
              <span className="block text-muted max-w-3xl">{s.blurb}</span>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-xs text-muted max-w-3xl">
          All strategies are evaluated under the same rules: long-only, fully invested, per-asset cap, a proportional fee
          on traded value, decisions that use only earlier data, and identical dates. PPO and DDPG are frozen trained
          models; the research trained them with a 252-day random-window environment, observation and reward
          normalization fitted on the training period only, and selected the checkpoint with the best validation reward.
        </p>
      </section>

      <section aria-labelledby="models-heading" className="rounded border border-line bg-panel p-4">
        <h2 id="models-heading" className="font-medium mb-1">
          Trained models
        </h2>
        <p className="text-xs text-muted mb-3 max-w-3xl">
          Each model is tied to the asset universe, cap, cost rate, input layout (191 inputs, 9 outputs) and dates it was
          trained on. Research models were trained on the nine sector ETFs; they cannot be applied to other assets.
        </p>
        {models === null ? (
          <p className="text-sm text-muted">Loading models…</p>
        ) : models.models.length === 0 ? (
          <p className="text-sm text-muted">No trained models found. Train one below.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">Trained PPO and DDPG models</caption>
              <thead>
                <tr className="border-b border-line text-left text-xs text-muted">
                  <th className="py-1 pr-3">Model</th>
                  <th className="pr-3">Universe</th>
                  <th className="pr-3">Training period</th>
                  <th className="pr-3">Validation period</th>
                  <th className="pr-3 text-right">Seed</th>
                  <th className="text-right">Steps</th>
                </tr>
              </thead>
              <tbody>
                {models.models.map((m) => (
                  <tr key={m.id} className="border-b border-line/50 align-top">
                    <td className="py-1.5 pr-3">
                      {m.label}
                      <div className="text-xs text-muted">{m.id}</div>
                    </td>
                    <td className="pr-3 text-xs">{m.universe.join(", ")}</td>
                    <td className="pr-3 tnum">{m.training_period.join(" to ")}</td>
                    <td className="pr-3 tnum">{m.validation_period.join(" to ")}</td>
                    <td className="pr-3 text-right tnum">{m.seed ?? "—"}</td>
                    <td className="text-right tnum">{m.timesteps?.toLocaleString("en-US") ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {models?.vix_range && (
          <p className="mt-3 text-xs text-muted">
            The agents use the VIX as an input. It comes from the research dataset ({models.vix_source}), covering{" "}
            {models.vix_range[0]} to {models.vix_range[1]}, so PPO/DDPG cannot be evaluated or trained outside that span.
          </p>
        )}
      </section>

      <section aria-labelledby="train-heading" className="rounded border border-line bg-panel p-4">
        <h2 id="train-heading" className="font-medium mb-1">
          Train a new model
        </h2>
        <p className="text-xs text-muted mb-3 max-w-3xl">
          Uses the research training code and settings unchanged. The portfolio must have exactly 9 assets (the
          environment’s fixed layout). Training uses CPU heavily, runs in a separate process, one job at a time, and
          starts only after you confirm. Keep the training and validation windows before any period you later want to
          test on.
        </p>
        <div className="grid gap-4 sm:grid-cols-3">
          <div>
            <label htmlFor="tr-portfolio" className="block text-sm mb-1">
              Portfolio (9 assets)
            </label>
            <select
              id="tr-portfolio"
              className={inputClass}
              value={form.portfolioId ?? ""}
              onChange={(e) => update({ portfolioId: e.target.value === "" ? null : Number(e.target.value) })}
            >
              <option value="">Select a portfolio…</option>
              {nineAssets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            {portfolios.length > 0 && nineAssets.length === 0 && (
              <p className="mt-1 text-xs text-muted">No saved portfolio has exactly 9 assets.</p>
            )}
          </div>
          <div>
            <label htmlFor="tr-algo" className="block text-sm mb-1">
              Algorithm
            </label>
            <select
              id="tr-algo"
              className={inputClass}
              value={form.algorithm}
              onChange={(e) => update({ algorithm: e.target.value as TrainingForm["algorithm"] })}
            >
              <option value="PPO">PPO</option>
              <option value="DDPG">DDPG</option>
            </select>
          </div>
          <div>
            <label htmlFor="tr-steps" className="block text-sm mb-1">
              Training steps
            </label>
            <input id="tr-steps" inputMode="numeric" className={inputClass} value={form.timesteps} onChange={(e) => update({ timesteps: e.target.value })} />
            <p className="mt-1 text-xs text-muted">Research: 100,000. Estimated time {estimate ?? "—"} (rough, CPU).</p>
          </div>
          {(
            [
              ["tr-ts", "Training start", "trainStart"],
              ["tr-te", "Training end", "trainEnd"],
              ["tr-vs", "Validation start", "valStart"],
              ["tr-ve", "Validation end", "valEnd"],
            ] as const
          ).map(([id, label, key]) => (
            <div key={id}>
              <label htmlFor={id} className="block text-sm mb-1">
                {label}
              </label>
              <input id={id} type="date" className={inputClass} value={form[key]} onChange={(e) => update({ [key]: e.target.value })} />
            </div>
          ))}
          <div>
            <label htmlFor="tr-cost" className="block text-sm mb-1">
              Transaction cost (% of traded value)
            </label>
            <input id="tr-cost" inputMode="decimal" className={inputClass} value={form.costPct} onChange={(e) => update({ costPct: e.target.value })} />
          </div>
          <div>
            <label htmlFor="tr-seed" className="block text-sm mb-1">
              Random seed
            </label>
            <input id="tr-seed" inputMode="numeric" className={inputClass} value={form.seed} onChange={(e) => update({ seed: e.target.value })} />
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-4">
          <button
            type="button"
            onClick={review}
            disabled={active !== null || busy}
            className="rounded bg-accent px-4 py-2 text-sm font-medium text-bg disabled:opacity-50"
          >
            Review training run…
          </button>
          {active && <span className="text-xs text-muted">A training job is already in progress.</span>}
          {formError && (
            <p role="alert" className="text-sm text-loss">
              {formError}
            </p>
          )}
        </div>

        {confirming && (
          <div role="alertdialog" aria-labelledby="confirm-heading" className="mt-4 rounded border border-accent/60 bg-bg p-4 text-sm">
            <h3 id="confirm-heading" className="font-medium">
              Confirm training
            </h3>
            <ul className="mt-2 list-disc pl-5 text-muted flex flex-col gap-1">
              <li>
                <span className="text-ink">{form.algorithm}</span> on {selected?.holdings.map((h) => h.symbol).join(", ")}, {Number(form.timesteps).toLocaleString("en-US")} steps, seed {form.seed}.
              </li>
              <li>
                Train {form.trainStart} to {form.trainEnd}; validate {form.valStart} to {form.valEnd} (used only to pick the best checkpoint).
              </li>
              <li>Estimated time {estimate} on CPU; the estimate is rough. You can cancel at any time and nothing is saved if you do.</li>
              <li>The model is only valid for these assets, this cap ({selected ? `${(selected.max_weight * 100).toFixed(0)}%` : ""}) and cost rate.</li>
            </ul>
            <label className="mt-3 flex items-center gap-2">
              <input type="checkbox" checked={understood} onChange={(e) => setUnderstood(e.target.checked)} />I understand this will use significant CPU time.
            </label>
            <div className="mt-3 flex gap-3">
              <button
                type="button"
                onClick={startTraining}
                disabled={!understood || busy}
                className="rounded bg-accent px-4 py-2 text-sm font-medium text-bg disabled:opacity-50"
              >
                Start training
              </button>
              <button type="button" onClick={() => setConfirming(false)} className="rounded border border-line px-4 py-2 text-sm hover:bg-line">
                Go back
              </button>
            </div>
          </div>
        )}

        {active && <JobProgress job={active} onCancel={() => cancel(active)} />}
      </section>

      {jobs.length > 0 && (
        <section aria-labelledby="jobs-heading" className="rounded border border-line bg-panel p-4">
          <h2 id="jobs-heading" className="font-medium mb-2">
            Training jobs
          </h2>
          <ul className="text-sm flex flex-col gap-1">
            {jobs.map((j) => (
              <li key={j.id}>
                #{j.id} {j.algorithm}, {j.config.training_start} to {j.config.validation_end},{" "}
                {j.config.total_timesteps.toLocaleString("en-US")} steps —{" "}
                <span className={j.status === "failed" ? "text-loss" : j.status === "succeeded" ? "text-gain" : "text-muted"}>
                  {j.status}
                </span>
                {j.model_id && <span className="text-muted"> ({j.model_id})</span>}
                {j.error && <span className="block text-xs text-loss">{j.error}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function JobProgress({ job, onCancel }: { job: TrainingJob; onCancel: () => void }) {
  const pct = Math.round(job.progress * 100);
  return (
    <div role="status" className="mt-4 text-sm">
      <div className="mb-1">
        Job #{job.id} ({job.algorithm}) {job.status === "queued" ? "queued" : `training — ${pct}% of steps validated`}. Estimated total{" "}
        {formatDuration(job.estimated_seconds)}.
      </div>
      <div className="h-2 w-full max-w-md rounded bg-line">
        <div className="h-2 rounded bg-accent" style={{ width: `${pct}%` }} />
      </div>
      <button type="button" onClick={onCancel} className="mt-3 rounded border border-line px-3 py-1.5 text-sm hover:bg-line">
        Cancel training
      </button>
    </div>
  );
}
