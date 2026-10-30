"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ResultCharts } from "@/components/ResultCharts";
import { apiGet, type BacktestRun, type DownsideMetrics, type RecoveryAnalysis } from "@/lib/api";
import { money, pct, pctPlain } from "@/lib/format";

const num = (v: number | null | undefined, digits = 2) => (v === null || v === undefined ? "n/a" : v.toFixed(digits));

type Row = {
  name: string;
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number | null;
  maximum_drawdown: number;
  downside: DownsideMetrics;
  recovery: RecoveryAnalysis;
  transaction_costs: number | null;
  total_turnover: number | null;
};

export function metricRows(run: BacktestRun): Row[] {
  const s = run.summary;
  if (!s) return [];
  const rows: Row[] = Object.values(s.strategies).map((m) => ({
    name: m.label,
    ...m,
  }));
  if (s.benchmark) rows.push({ name: s.benchmark.label, ...s.benchmark, transaction_costs: null, total_turnover: null });
  return rows;
}

function recoveryText(r: RecoveryAnalysis): { status: string; days: string; ret: string } {
  if (r.status === "no_drawdown") return { status: "No drawdown", days: "n/a", ret: "n/a" };
  const ret = r.recovery_return_from_trough === undefined ? "n/a" : pct(r.recovery_return_from_trough);
  if (r.status === "recovered")
    return { status: `Recovered ${r.recovery_date}`, days: String(r.recovery_trading_days), ret };
  return { status: "Not recovered in window", days: "n/a", ret };
}

export function RiskAnalyticsView() {
  const [runs, setRuns] = useState<BacktestRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  useEffect(() => {
    apiGet<BacktestRun[]>("/backtests?limit=50")
      .then((all) => {
        const done = all.filter((r) => r.status === "succeeded");
        setRuns(done);
        setSelectedId(done[0]?.id ?? null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <p className="text-sm text-loss">Could not load backtests: {error}</p>;
  if (!runs) return <p className="text-sm text-muted">Loading…</p>;
  if (runs.length === 0)
    return (
      <p className="rounded border border-line bg-panel p-4 text-sm text-muted">
        No completed backtests yet. Run one on the <Link className="text-accent underline" href="/backtests">Backtests</Link>{" "}
        page and its risk analytics will appear here.
      </p>
    );

  const run = runs.find((r) => r.id === selectedId) ?? runs[0];
  const s = run.summary!;
  const rows = metricRows(run);

  return (
    <div className="flex flex-col gap-6">
      <div className="max-w-md">
        <label htmlFor="risk-run" className="block text-sm mb-1">
          Backtest
        </label>
        <select
          id="risk-run"
          className="w-full rounded border border-line bg-bg px-3 py-2 text-sm"
          value={run.id}
          onChange={(e) => setSelectedId(Number(e.target.value))}
        >
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              #{r.id} {r.portfolio_name}, {r.config.start} to {r.config.end}
            </option>
          ))}
        </select>
      </div>

      <section aria-labelledby="risk-table" className="rounded border border-line bg-panel p-4">
        <h2 id="risk-table" className="font-medium">
          Risk and return metrics
        </h2>
        <p className="mt-1 text-xs text-muted">
          {s.period.first_return_date} to {s.period.last_return_date}, {s.period.trading_days} trading days, all series
          on identical dates and starting from {money(Object.values(s.strategies)[0].initial_value)}. Daily returns net of
          transaction costs; 252 trading days per year.
          {s.benchmark ? ` ${s.benchmark.symbol} is a buy-and-hold benchmark with no trading costs.` : " No benchmark was selected."}
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-muted border-b border-line">
              <tr>
                <th className="py-1 pr-3">Series</th>
                <th className="pr-3 text-right">Total return</th>
                <th className="pr-3 text-right">Annualized return</th>
                <th className="pr-3 text-right">Volatility</th>
                <th className="pr-3 text-right">Sharpe</th>
                <th className="pr-3 text-right">Sortino</th>
                <th className="pr-3 text-right">Max drawdown</th>
                <th className="pr-3 text-right">VaR 95% (1 day)</th>
                <th className="pr-3 text-right">CVaR 95% (1 day)</th>
                <th className="pr-3 text-right">Worst day</th>
                <th className="pr-3 text-right">Costs</th>
                <th className="text-right">Turnover (sum)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.name} className="border-b border-line/50">
                  <td className="py-1.5 pr-3">{m.name}</td>
                  <td className={`pr-3 text-right tnum ${m.total_return < 0 ? "text-loss" : "text-gain"}`}>{pct(m.total_return)}</td>
                  <td className="pr-3 text-right tnum">{pct(m.annualized_return)}</td>
                  <td className="pr-3 text-right tnum">{pctPlain(m.annualized_volatility)}</td>
                  <td className="pr-3 text-right tnum">{num(m.sharpe_ratio)}</td>
                  <td className="pr-3 text-right tnum">{num(m.downside.sortino_ratio)}</td>
                  <td className="pr-3 text-right tnum">{pct(m.maximum_drawdown)}</td>
                  <td className="pr-3 text-right tnum">{m.downside.var_95_daily === null ? "n/a" : pct(m.downside.var_95_daily, 2)}</td>
                  <td className="pr-3 text-right tnum">{m.downside.cvar_95_daily === null ? "n/a" : pct(m.downside.cvar_95_daily, 2)}</td>
                  <td className="pr-3 text-right tnum">{m.downside.worst_day === null ? "n/a" : pct(m.downside.worst_day, 2)}</td>
                  <td className="pr-3 text-right tnum">{m.transaction_costs === null ? "–" : money(m.transaction_costs)}</td>
                  <td className="text-right tnum">{m.total_turnover === null ? "–" : `${(m.total_turnover * 100).toFixed(0)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="recovery-table" className="rounded border border-line bg-panel p-4">
        <h2 id="recovery-table" className="font-medium">
          Drawdown and recovery
        </h2>
        <p className="mt-1 text-xs text-muted">
          Reference: the maximum-drawdown trough of each series within the tested window, and the highest value reached
          before it (the starting capital counts). Recovery time is the trading days from the trough until that peak is
          reached again. Recovery return is the gain from the trough to the end of the window. A series that has not
          recovered by the end of the window is reported as such, not extrapolated.
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-muted border-b border-line">
              <tr>
                <th className="py-1 pr-3">Series</th>
                <th className="pr-3">Peak</th>
                <th className="pr-3">Trough</th>
                <th className="pr-3 text-right">Depth</th>
                <th className="pr-3">Status</th>
                <th className="pr-3 text-right">Recovery time (trading days)</th>
                <th className="text-right">Recovery return from trough</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => {
                const t = recoveryText(m.recovery);
                return (
                  <tr key={m.name} className="border-b border-line/50">
                    <td className="py-1.5 pr-3">{m.name}</td>
                    <td className="pr-3 tnum">{m.recovery.peak_date ?? "–"}</td>
                    <td className="pr-3 tnum">{m.recovery.trough_date ?? "–"}</td>
                    <td className="pr-3 text-right tnum">{pct(m.recovery.max_drawdown)}</td>
                    <td className="pr-3">{t.status}</td>
                    <td className="pr-3 text-right tnum">{t.days}</td>
                    <td className="text-right tnum">{t.ret}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <ResultCharts run={run} groups={["risk", "performance"]} />

      <section aria-labelledby="definitions" className="rounded border border-line bg-panel p-4 text-sm">
        <h2 id="definitions" className="font-medium mb-2">
          Definitions
        </h2>
        <ul className="list-disc pl-5 flex flex-col gap-1 text-muted">
          <li>Sharpe: mean daily excess return over the risk-free rate divided by its standard deviation, times √252.</li>
          <li>
            Sortino: mean daily excess return divided by the downside deviation, √(mean of min(excess, 0)² over all days),
            times √252; the risk-free rate is the target.
          </li>
          <li>
            VaR 95% (1 day): the 5th percentile of daily net returns in this window (historical, not a forecast).
            CVaR 95%: the average of the daily returns at or below it. Negative numbers are losses; both need at least
            20 days.
          </li>
          <li>Maximum drawdown: the largest fall from a running peak, measured on daily closing values.</li>
          <li>Turnover: the sum over days of absolute weight changes traded (100% = the whole portfolio traded once).</li>
        </ul>
        <p className="mt-3 text-xs text-muted">{s.disclaimer}</p>
      </section>
    </div>
  );
}
