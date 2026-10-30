"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  apiGet,
  apiSend,
  type DownturnRow,
  type ResearchOverview,
  type ResearchRecoveryRow,
  type ResearchVerification,
} from "@/lib/api";
import { label, money, pct, pctPlain } from "@/lib/format";

export const strategyName = (key: string) =>
  key === "ppo" || key === "ddpg" ? key.toUpperCase() : label(key);

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: ResearchOverview };

const range = (r: [string, string]) => `${r[0]} → ${r[1]}`;

function Verify({ event }: { event: string }) {
  const [state, setState] = useState<
    { kind: "idle" } | { kind: "running" } | { kind: "error"; message: string } | { kind: "done"; result: ResearchVerification }
  >({ kind: "idle" });

  async function run() {
    setState({ kind: "running" });
    try {
      setState({ kind: "done", result: await apiSend<ResearchVerification>("POST", `/research/verify/${event}`) });
    } catch (e) {
      setState({ kind: "error", message: (e as Error).message });
    }
  }

  return (
    <div className="flex flex-col gap-1 text-xs">
      <button
        type="button"
        onClick={run}
        disabled={state.kind === "running"}
        className="self-start rounded border border-line px-2 py-1 hover:border-accent disabled:opacity-50"
      >
        {state.kind === "running" ? "Recomputing…" : `Recompute ${label(event)}`}
      </button>
      {state.kind === "error" && (
        <span role="alert" className="text-loss">
          {state.message}
        </span>
      )}
      {state.kind === "done" && (
        <span role={state.result.all_match ? "status" : "alert"} className={state.result.all_match ? "text-gain" : "text-loss"}>
          {state.result.all_match
            ? `All ${state.result.checks.length} metrics match the stored results (tolerance ${state.result.tolerance}).`
            : `${state.result.checks.filter((c) => !c.matches).length} of ${state.result.checks.length} metrics differ: ` +
              state.result.checks
                .filter((c) => !c.matches)
                .map((c) => `${strategyName(c.strategy)} ${c.metric}`)
                .join(", ")}
        </span>
      )}
    </div>
  );
}

function DownturnTable({ rows }: { rows: DownturnRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tnum">
        <caption className="sr-only">Downturn-period results by event and strategy</caption>
        <thead className="text-left text-muted border-b border-line">
          <tr>
            <th className="py-2 pr-3">Strategy</th>
            <th className="pr-3 text-right">Total return</th>
            <th className="pr-3 text-right">Annualized</th>
            <th className="pr-3 text-right">Volatility</th>
            <th className="pr-3 text-right">Sharpe</th>
            <th className="pr-3 text-right">Max drawdown</th>
            <th className="text-right">Costs</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.strategy} className="border-b border-line/50">
              <td className="py-1.5 pr-3">{strategyName(r.strategy)}</td>
              <td className={`pr-3 text-right ${r.total_return < 0 ? "text-loss" : "text-gain"}`}>{pct(r.total_return)}</td>
              <td className="pr-3 text-right">{pct(r.annualized_return)}</td>
              <td className="pr-3 text-right">{pctPlain(r.annualized_volatility)}</td>
              <td className="pr-3 text-right">{r.sharpe_ratio.toFixed(3)}</td>
              <td className="pr-3 text-right">{pct(r.maximum_drawdown)}</td>
              <td className="text-right">{money(r.transaction_costs)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RecoveryTable({ rows }: { rows: ResearchRecoveryRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tnum">
        <caption className="sr-only">Recovery-period results by event and strategy</caption>
        <thead className="text-left text-muted border-b border-line">
          <tr>
            <th className="py-2 pr-3">Strategy</th>
            <th className="pr-3 text-right">Trough value</th>
            <th className="pr-3 text-right">Recovery return</th>
            <th className="pr-3 text-right">Ending value</th>
            <th>Regained $100,000?</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.strategy} className="border-b border-line/50">
              <td className="py-1.5 pr-3">{strategyName(r.strategy)}</td>
              <td className="pr-3 text-right">{money(r.trough_value)}</td>
              <td className="pr-3 text-right">{pct(r.recovery_return_from_trough)}</td>
              <td className="pr-3 text-right">{money(r.ending_value)}</td>
              <td>
                {r.recovered_within_24_months
                  ? `Yes, on ${r.recovery_date} (${r.recovery_time_trading_days} trading days after the trough)`
                  : "Not recovered within 24 months"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ResearchMode() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    apiGet<ResearchOverview>("/research/overview", controller.signal)
      .then((data) => setState({ kind: "ready", data }))
      .catch((e: Error) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: e.message });
      });
    return () => controller.abort();
  }, []);

  if (state.kind === "loading") return <p className="text-muted">Loading research results…</p>;
  if (state.kind === "error")
    return (
      <p role="alert" className="text-loss">
        Could not load research results: {state.message}
      </p>
    );

  const { data } = state;
  const s = data.settings;
  const settings: [string, string][] = [
    ["Assets", s.assets.join(", ")],
    ["Starting capital", money(s.initial_capital)],
    ["Transaction cost", `${pctPlain(s.transaction_cost_rate, 2)} of traded value (sum of absolute weight changes × pre-trade value)`],
    ["Constraints", `long-only, fully invested, ${pctPlain(s.max_weight, 0)} maximum per asset`],
    ["Prices / VIX", `${label(s.price_field)} (yfinance) / ${s.volatility_source} VIX as a model input, not an asset`],
    ["Risk-free rate", `${pctPlain(s.risk_free_rate_daily, 0)} daily; ${s.annualization_trading_days} trading days per year`],
    ["Initial weights", label(s.initial_weights)],
    ["Equal Weight", `rebalance ${s.equal_weight.rebalance}`],
    [
      "Mean Variance",
      `${s.mean_variance.lookback_trading_days}-day lookback, ${s.mean_variance.rebalance}, ${pctPlain(s.mean_variance.turnover_threshold, 0)} turnover threshold, maximum Sharpe`,
    ],
    [
      "Minimum Variance",
      `${s.minimum_variance.lookback_trading_days}-day lookback, ${s.minimum_variance.rebalance}, ${pctPlain(s.minimum_variance.turnover_threshold, 0)} turnover threshold`,
    ],
    ["PPO / DDPG", `${s.observation_size}-value observation, one seed (${s.random_seed}) per model, frozen before the test window`],
  ];

  return (
    <div className="flex flex-col gap-6">
      <section aria-labelledby="locked-heading" className="rounded border border-line bg-panel p-4">
        <h2 id="locked-heading" className="font-medium mb-1">
          Locked configuration
        </h2>
        <p className="text-xs text-muted mb-3 max-w-3xl">
          Read-only. These settings come from <code>configs/experiment.yaml</code> and the results from{" "}
          <code>{data.source}</code>. Custom portfolios and backtests are stored separately and can never change them;
          try your own ideas in the <Link href="/backtests" className="text-accent underline">Backtests</Link> page,
          where results are labeled exploratory.
        </p>
        <dl className="grid grid-cols-1 sm:grid-cols-[10rem_1fr] gap-x-4 gap-y-1.5 text-sm">
          {settings.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted">{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section aria-labelledby="findings-heading" className="rounded border border-line bg-panel p-4">
        <h2 id="findings-heading" className="font-medium mb-3">
          What the stored results show
        </h2>
        <ul className="flex flex-col gap-2 text-sm">
          {data.findings.map((f) => (
            <li key={f.text}>
              {f.text}
              <span className="block text-xs text-muted">{f.evidence}</span>
            </li>
          ))}
        </ul>
        <ul className="mt-4 list-disc pl-5 text-xs text-muted flex flex-col gap-1 max-w-3xl">
          {data.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </section>

      {Object.entries(data.events).map(([event, windows]) => (
        <section
          key={event}
          aria-labelledby={`event-${event}`}
          className="rounded border border-line bg-panel p-4 flex flex-col gap-4"
        >
          <div>
            <h2 id={`event-${event}`} className="font-medium">
              {label(event)}
            </h2>
            <p className="text-xs text-muted mt-1">
              Training {range(windows.training)} · validation {range(windows.validation)} · downturn test{" "}
              {range(windows.downturn_test)} · recovery observation {range(windows.recovery_observation)}
            </p>
          </div>
          <div>
            <h3 className="text-sm text-muted mb-1">During the downturn</h3>
            <DownturnTable rows={data.downturn.filter((r) => r.event === event)} />
          </div>
          <div>
            <h3 className="text-sm text-muted mb-1">During the 24-month recovery observation</h3>
            <RecoveryTable rows={data.recovery.filter((r) => r.event === event)} />
          </div>
          <Verify event={event} />
        </section>
      ))}
    </div>
  );
}
