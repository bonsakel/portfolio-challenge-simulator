"use client";

import { useEffect, useState } from "react";
import { ChartCard } from "@/components/ChartCard";
import { apiGet, type BacktestResultData, type BacktestRun } from "@/lib/api";
import {
  allocationChart,
  comparisonChart,
  costChart,
  drawdownChart,
  recoveryChart,
  riskReturnChart,
  turnoverChart,
  valueChart,
  type ChartSpec,
} from "@/lib/charts";

export type ChartGroup = "performance" | "risk" | "trading";

// The result series are fetched once per run and reused by every chart on the page.
export function useRunResults(runId: number) {
  const [data, setData] = useState<BacktestResultData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    apiGet<BacktestResultData>(`/backtests/${runId}/results`)
      .then((d) => alive && setData(d))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [runId]);
  return { data, error };
}

export function ResultCharts({ run, groups }: { run: BacktestRun; groups: ChartGroup[] }) {
  const { data, error } = useRunResults(run.id);
  const keys = Object.keys(run.summary?.strategies ?? {});
  const [allocKey, setAllocKey] = useState<string>(keys[0] ?? "");

  if (error) return <p className="text-sm text-loss">Charts could not be loaded: {error}</p>;
  if (!data || !run.summary) return <p className="text-sm text-muted">Loading…</p>;
  if (!data.series || Object.keys(data.series).length === 0)
    return <p className="text-sm text-loss">The backtest returned no result series to chart.</p>;

    const specs: ChartSpec[] = [];
  let recoveryNote = false;
  if (groups.includes("performance")) specs.push(valueChart(data, run), drawdownChart(data, run));
  if (groups.includes("risk")) {
    specs.push(riskReturnChart(run), comparisonChart(run));
    const recovery = recoveryChart(data, run);
    if (recovery) specs.push(recovery);
    else recoveryNote = true;
  }
  if (groups.includes("trading")) specs.push(costChart(data, run), turnoverChart(data, run));
  const alloc = groups.includes("trading") && data.series[allocKey];

  return (
    <div className="flex flex-col gap-4">
      {specs.map((spec) => (
        <ChartCard key={spec.id} spec={spec} />
      ))}
      {recoveryNote && (
        <p className="rounded border border-line bg-panel p-4 text-sm text-muted">
          Recovery trajectory not shown: no series has trading days after its drawdown trough (or none had a drawdown),
          so there is nothing to plot.
        </p>
      )}
      {alloc && (
        <div className="flex flex-col gap-2">
          {keys.length > 1 && (
            <div className="flex items-center gap-2 text-sm">
              <label htmlFor="alloc-strategy">Allocation chart for</label>
              <select
                id="alloc-strategy"
                className="rounded border border-line bg-bg px-2 py-1"
                value={allocKey}
                onChange={(e) => setAllocKey(e.target.value)}
              >
                {keys.map((k) => (
                  <option key={k} value={k}>
                    {run.summary?.strategies[k].label}
                  </option>
                ))}
              </select>
            </div>
          )}
          <ChartCard spec={allocationChart(data, allocKey, run.summary.strategies[allocKey].label)} height={340} />
        </div>
      )}
    </div>
  );
}
