"use client";

import { useEffect, useState } from "react";
import { apiGet, type ModelInfo, type ModelsResponse } from "@/lib/api";
import { RL_KEYS, type BacktestForm } from "@/lib/backtest";
import { parseNumber } from "@/lib/portfolio";

type Props = {
  form: BacktestForm;
  onChange: (rlModels: BacktestForm["rlModels"]) => void;
};

/** For each selected RL strategy: which trained models may be used here, and why others may not. */
export function ModelPicker({ form, onChange }: Props) {
  const wanted = RL_KEYS.filter((k) => form.strategies.includes(k));
  const [response, setResponse] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cost = parseNumber(form.costPct);
  const ready = wanted.length > 0 && form.portfolioId !== null && form.start < form.end && cost !== null;

  useEffect(() => {
    if (!ready) return;
    const controller = new AbortController();
    const params = new URLSearchParams({
      portfolio_id: String(form.portfolioId),
      start: form.start,
      end: form.end,
      transaction_cost_rate: String((cost ?? 0) / 100),
      initial_allocation: form.allocation,
    });
    apiGet<ModelsResponse>(`/models?${params}`, controller.signal)
      .then((r) => {
        setResponse(r);
        setError(null);
      })
      .catch((e: Error) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [ready, form.portfolioId, form.start, form.end, form.allocation, cost]);

  // Keep each selection valid: only a model that passed the compatibility check may stay selected.
  const compatibleIds = response
    ? response.models.filter((m) => m.compatibility?.compatible).map((m) => m.id).join("|")
    : "";
  useEffect(() => {
    if (!response) return;
    const next: BacktestForm["rlModels"] = { ...form.rlModels };
    let changed = false;
    for (const key of RL_KEYS) {
      const options = response.models.filter((m) => m.algorithm.toLowerCase() === key && m.compatibility?.compatible);
      const current = next[key];
      if (current && !options.some((m) => m.id === current)) {
        delete next[key];
        changed = true;
      }
      if (!next[key] && options.length > 0 && form.strategies.includes(key)) {
        next[key] = options[0].id;
        changed = true;
      }
    }
    if (changed) onChange(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compatibleIds, form.strategies.join("|")]);

  if (wanted.length === 0) return null;
  if (!ready)
    return (
      <p className="mt-4 text-xs text-muted">Choose a portfolio, dates and cost to see which trained models are compatible.</p>
    );
  if (error)
    return (
      <p role="alert" className="mt-4 text-sm text-loss">
        Could not check models: {error}
      </p>
    );
  if (!response) return <p className="mt-4 text-xs text-muted">Checking trained models…</p>;

  return (
    <section aria-labelledby="rl-heading" className="mt-4 rounded border border-line bg-bg p-3 text-sm">
      <h3 id="rl-heading" className="font-medium">
        Reinforcement-learning models
      </h3>
      <p className="mt-1 text-xs text-muted max-w-3xl">
        A model is used only if it was trained on exactly this asset list and order, the same position cap and cost
        rate, and its training and validation dates do not overlap your window. Nothing is substituted silently.
        Results come from a frozen model; no training happens in a backtest.
        {response.vix_range && ` The VIX input is available ${response.vix_range[0]} to ${response.vix_range[1]}.`}
      </p>
      {wanted.map((key) => {
        const all = response.models.filter((m) => m.algorithm.toLowerCase() === key);
        const ok = all.filter((m) => m.compatibility?.compatible);
        const blocked = all.filter((m) => !m.compatibility?.compatible);
        const chosen = all.find((m) => m.id === form.rlModels[key]);
        return (
          <div key={key} className="mt-3">
            <label htmlFor={`rl-${key}`} className="block mb-1">
              {key.toUpperCase()} model
            </label>
            {ok.length > 0 ? (
              <select
                id={`rl-${key}`}
                className="w-full max-w-xl rounded border border-line bg-bg px-3 py-2 text-sm"
                value={form.rlModels[key] ?? ""}
                onChange={(e) => onChange({ ...form.rlModels, [key]: e.target.value })}
              >
                {ok.map((m) => (
                  <option key={m.id} value={m.id}>
                    {describe(m)}
                  </option>
                ))}
              </select>
            ) : (
              <p role="alert" className="text-loss">
                No compatible {key.toUpperCase()} model for this portfolio and window. Change the portfolio or dates, or{" "}
                <a className="underline" href="/strategy-lab">
                  train a new model in the Strategy Lab
                </a>
                .
              </p>
            )}
            {chosen?.compatibility?.warnings.map((w) => (
              <p key={w} className="mt-1 text-xs text-muted">
                Note: {w}
              </p>
            ))}
            {blocked.length > 0 && (
              <details className="mt-1 text-xs text-muted">
                <summary className="cursor-pointer">{blocked.length} model(s) not usable here</summary>
                <ul className="mt-1 list-disc pl-5">
                  {blocked.map((m) => (
                    <li key={m.id}>
                      {m.label}: {m.compatibility?.issues.join(" ")}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        );
      })}
    </section>
  );
}

function describe(m: ModelInfo): string {
  return `${m.label} — trained ${m.training_period[0]} to ${m.training_period[1]}, validated to ${m.validation_period[1]}`;
}
