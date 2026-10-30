"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { BacktestResults } from "@/components/BacktestResults";
import { ResultCharts } from "@/components/ResultCharts";
import { CoverageCheck } from "@/components/CoverageCheck";
import { ModelPicker } from "@/components/ModelPicker";
import {
  apiGet,
  apiSend,
  type BacktestRun,
  type Portfolio,
  type ResearchSummary,
  type StrategyKey,
} from "@/lib/api";
import { buildRequest, defaultForm, OPTIMIZER_KEYS, STRATEGIES, type BacktestForm } from "@/lib/backtest";

const inputClass = "w-full rounded border border-line bg-bg px-3 py-2 text-sm tnum";

export function BacktestRunner({ pollMs = 600 }: { pollMs?: number }) {
  const [form, setForm] = useState<BacktestForm>(() => defaultForm());
  const [portfolios, setPortfolios] = useState<Portfolio[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [events, setEvents] = useState<{ event: string; start: string; end: string }[]>([]);
  const [recent, setRecent] = useState<BacktestRun[]>([]);
  const [run, setRun] = useState<BacktestRun | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadRecent = useCallback(async () => {
    try {
      setRecent(await apiGet<BacktestRun[]>("/backtests?limit=10"));
    } catch {
      /* the portfolio load below reports connectivity problems */
    }
  }, []);

  useEffect(() => {
    apiGet<Portfolio[]>("/portfolios")
      .then(setPortfolios)
      .catch((e: Error) => setLoadError(e.message));
    apiGet<ResearchSummary>("/research/summary")
      .then((r) => {
        const seen = new Map<string, { event: string; start: string; end: string }>();
        for (const row of r.rows)
          if (!seen.has(row.event)) seen.set(row.event, { event: row.event, start: row.period_start, end: row.period_end });
        setEvents([...seen.values()]);
      })
      .catch(() => setEvents([]));
    void loadRecent();
  }, [loadRecent]);

  // Poll the active run until it finishes.
  const activeId = run && (run.status === "queued" || run.status === "running") ? run.id : null;
  useEffect(() => {
    if (activeId === null) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await apiGet<BacktestRun>(`/backtests/${activeId}`);
        if (cancelled) return;
        setRun(next);
        if (next.status === "queued" || next.status === "running") timer.current = setTimeout(tick, pollMs);
        else void loadRecent();
      } catch (e) {
        if (!cancelled) {
          setFormError(`Lost contact with the backend while the run was in progress: ${(e as Error).message}`);
          timer.current = setTimeout(tick, pollMs * 3);
        }
      }
    };
    timer.current = setTimeout(tick, pollMs);
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [activeId, pollMs, loadRecent]);

  const selected = portfolios?.find((p) => p.id === form.portfolioId) ?? null;
  const optimizerSelected = form.strategies.some((s) => OPTIMIZER_KEYS.includes(s));
  const busy = submitting || activeId !== null;

  function update(patch: Partial<BacktestForm>) {
    setForm((f) => ({ ...f, ...patch }));
    setFormError(null);
  }

  function toggle(key: StrategyKey) {
    update({
      strategies: form.strategies.includes(key)
        ? form.strategies.filter((s) => s !== key)
        : [...form.strategies, key],
    });
  }

  async function start() {
    const built = buildRequest(form);
    if ("error" in built) {
      setFormError(built.error);
      return;
    }
    if (selected && !selected.validation.valid) {
      setFormError("The selected portfolio has validation errors. Fix it in the Portfolio Builder first.");
      return;
    }
    setSubmitting(true);
    setFormError(null);
    setRun(null);
    try {
      setRun(await apiSend<BacktestRun>("POST", "/backtests", built.request));
    } catch (e) {
      setFormError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  if (loadError)
    return (
      <p role="alert" className="text-sm text-loss">
        Could not load saved portfolios: {loadError}
      </p>
    );
  if (portfolios === null) return <p className="text-sm text-muted">Loading portfolios…</p>;
  if (portfolios.length === 0)
    return (
      <p className="text-sm text-muted">
        No saved portfolios yet. Create one in the <a className="text-accent hover:underline" href="/portfolio-builder">Portfolio Builder</a>, then come back to run a backtest.
      </p>
    );

  return (
    <div className="flex flex-col gap-6">
      <section aria-labelledby="setup-heading" className="rounded border border-line bg-panel p-4">
        <h2 id="setup-heading" className="font-medium mb-3">
          Configure backtest
        </h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="bt-portfolio" className="block text-sm mb-1">
              Portfolio
            </label>
            <select
              id="bt-portfolio"
              className={inputClass}
              value={form.portfolioId ?? ""}
              onChange={(e) => update({ portfolioId: e.target.value === "" ? null : Number(e.target.value) })}
            >
              <option value="">Select a saved portfolio…</option>
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.holdings.length} assets){p.validation.valid ? "" : " — invalid"}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="bt-event" className="block text-sm mb-1">
              Date preset (research downturn)
            </label>
            <select
              id="bt-event"
              className={inputClass}
              value=""
              onChange={(e) => {
                const ev = events.find((x) => x.event === e.target.value);
                if (ev) update({ start: ev.start, end: ev.end });
              }}
            >
              <option value="">Custom dates</option>
              {events.map((ev) => (
                <option key={ev.event} value={ev.event}>
                  {ev.event.replace(/_/g, " ")} ({ev.start} to {ev.end})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="bt-start" className="block text-sm mb-1">
              Start date
            </label>
            <input id="bt-start" type="date" className={inputClass} value={form.start} onChange={(e) => update({ start: e.target.value })} />
          </div>
          <div>
            <label htmlFor="bt-end" className="block text-sm mb-1">
              End date
            </label>
            <input id="bt-end" type="date" className={inputClass} value={form.end} onChange={(e) => update({ end: e.target.value })} />
          </div>
        </div>

        <fieldset className="mt-4">
          <legend className="text-sm mb-1">Strategies</legend>
          <div className="flex flex-col gap-2">
            {STRATEGIES.map((s) => (
              <label key={s.key} className="flex items-start gap-2 text-sm">
                <input
                  type="checkbox"
                  className="mt-1"
                  checked={form.strategies.includes(s.key)}
                  onChange={() => toggle(s.key)}
                />
                <span>
                  {s.name}
                  <span className="block text-xs text-muted max-w-3xl">{s.blurb}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
        <ModelPicker form={form} onChange={(rlModels) => setForm((f) => ({ ...f, rlModels }))} />

        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <div>
            <label htmlFor="bt-cost" className="block text-sm mb-1">
              Transaction cost (% of traded value)
            </label>
            <input id="bt-cost" inputMode="decimal" className={inputClass} value={form.costPct} onChange={(e) => update({ costPct: e.target.value })} />
          </div>
          <div>
            <label htmlFor="bt-rebalance" className="block text-sm mb-1">
              Rebalancing
            </label>
            <select
              id="bt-rebalance"
              className={inputClass}
              value={form.rebalance}
              onChange={(e) => update({ rebalance: e.target.value as BacktestForm["rebalance"] })}
            >
              <option value="">Research default per strategy</option>
              <option value="daily">Daily</option>
              <option value="monthly">Monthly (first trading day)</option>
            </select>
          </div>
          <div>
            <label htmlFor="bt-threshold" className="block text-sm mb-1">
              Trade threshold (percentage points)
            </label>
            <input
              id="bt-threshold"
              inputMode="decimal"
              placeholder="Research default"
              className={inputClass}
              value={form.thresholdPct}
              onChange={(e) => update({ thresholdPct: e.target.value })}
            />
          </div>
          {optimizerSelected && (
            <div>
              <label htmlFor="bt-lookback" className="block text-sm mb-1">
                Optimizer lookback (trading days)
              </label>
              <input id="bt-lookback" inputMode="numeric" className={inputClass} value={form.lookback} onChange={(e) => update({ lookback: e.target.value })} />
            </div>
          )}
          <div>
            <label htmlFor="bt-rf" className="block text-sm mb-1">
              Risk-free rate (% per year)
            </label>
            <input id="bt-rf" inputMode="decimal" className={inputClass} value={form.riskFreePct} onChange={(e) => update({ riskFreePct: e.target.value })} />
          </div>
          <div>
            <label htmlFor="bt-allocation" className="block text-sm mb-1">
              Starting allocation
            </label>
            <select
              id="bt-allocation"
              className={inputClass}
              value={form.allocation}
              onChange={(e) => update({ allocation: e.target.value as BacktestForm["allocation"] })}
            >
              <option value="equal">Equal weight (research convention)</option>
              <option value="portfolio_weights">Portfolio’s own weights</option>
            </select>
          </div>
          <div>
            <label htmlFor="bt-benchmark" className="block text-sm mb-1">
              Benchmark ticker (optional)
            </label>
            <input
              id="bt-benchmark"
              className={inputClass}
              placeholder="SPY"
              value={form.benchmark}
              onChange={(e) => update({ benchmark: e.target.value })}
            />
            <p className="mt-1 text-xs text-muted">Buy and hold, same dates and starting capital. Leave empty for none.</p>
          </div>
        </div>
      </section>

      {selected && (
        <section aria-labelledby="summary-heading" className="rounded border border-line bg-panel p-4 text-sm">
          <h2 id="summary-heading" className="font-medium mb-2">
            Before you run
          </h2>
          <ul className="list-disc pl-5 flex flex-col gap-1 text-muted">
            <li>
              <span className="text-ink">{selected.holdings.map((h) => h.symbol).join(", ")}</span>: long-only, fully
              invested, max {(selected.max_weight * 100).toFixed(0)}% per asset, starting capital{" "}
              {selected.initial_capital.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 })}.
            </li>
            <li>
              Decisions use only returns before each day; the fee is {form.costPct}% of the dollar value traded (sum of
              absolute weight changes), charged once before that day’s return. No spreads, impact, taxes or liquidity.
            </li>
            <li>
              Prices are adjusted close (dividends reflected), with dates missing for any asset dropped for all; nothing
              is forward-filled. Missing history is downloaded on first use (internet required) or read from the cache.
            </li>
            <li>
              This is a selected-universe test of today’s listed securities, not a survivorship-free study. It is not
              investment advice.
            </li>
          </ul>
          <div className="mt-3">
            <CoverageCheck symbols={selected.holdings.map((h) => h.symbol)} />
          </div>
        </section>
      )}

      <div className="flex flex-wrap items-center gap-4">
        <button
          type="button"
          onClick={start}
          disabled={busy}
          className="rounded bg-accent px-4 py-2 text-sm font-medium text-bg disabled:opacity-50"
        >
          {busy ? "Running…" : run?.status === "failed" ? "Retry backtest" : "Run backtest"}
        </button>
        {formError && (
          <p role="alert" className="text-sm text-loss">
            {formError}
          </p>
        )}
      </div>

      {run && (run.status === "queued" || run.status === "running") && (
        <div role="status" className="text-sm">
          <div className="mb-1">
            {run.status === "queued" ? "Queued…" : "Running"} ({Math.round(run.progress * 100)}%). Missing price history
            is downloaded first, which can take a moment.
          </div>
          <div className="h-2 w-full max-w-md rounded bg-line">
            <div className="h-2 rounded bg-accent" style={{ width: `${Math.round(run.progress * 100)}%` }} />
          </div>
        </div>
      )}
      {run?.status === "failed" && (
        <p role="alert" className="rounded border border-loss/50 bg-panel p-3 text-sm text-loss">
          Backtest #{run.id} failed: {run.error}
        </p>
      )}
      {run?.status === "succeeded" && <BacktestResults run={run} />}
      {run?.status === "succeeded" && <ResultCharts run={run} groups={["performance", "trading"]} />}

      {recent.length > 0 && (
        <section aria-labelledby="recent-heading" className="rounded border border-line bg-panel p-4">
          <h2 id="recent-heading" className="font-medium mb-2">
            Recent backtests
          </h2>
          <ul className="text-sm flex flex-col gap-1">
            {recent.map((r) => (
              <li key={r.id} className="flex items-center justify-between gap-2">
                <span>
                  #{r.id} {r.portfolio_name} — {r.config.start} to {r.config.end} ({r.config.strategies.length} strategies){" "}
                  <span className={r.status === "failed" ? "text-loss" : r.status === "succeeded" ? "text-gain" : "text-muted"}>
                    {r.status}
                  </span>
                </span>
                <button
                  type="button"
                  className="rounded border border-line px-2 py-1 text-xs hover:bg-line"
                  aria-label={`Open backtest ${r.id}`}
                  onClick={() => setRun(r)}
                >
                  Open
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
