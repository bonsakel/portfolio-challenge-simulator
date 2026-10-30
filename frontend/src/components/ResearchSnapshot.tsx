"use client";

import { useEffect, useState } from "react";
import { apiGet, type ResearchSummary } from "@/lib/api";
import { label, money, pct } from "@/lib/format";

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: ResearchSummary };

export function ResearchSnapshot() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    apiGet<ResearchSummary>("/research/summary", controller.signal)
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

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tnum">
        <caption className="text-left text-xs text-muted pb-2">
          Downturn-period results from {state.data.source}. Starting capital $100,000; 0.10% cost on
          traded value; Sharpe uses a 0% risk-free rate. One RL seed — exploratory, not conclusive.
        </caption>
        <thead className="text-left text-muted border-b border-line">
          <tr>
            <th className="py-2 pr-3">Event</th>
            <th className="pr-3">Strategy</th>
            <th className="pr-3 text-right">Total return</th>
            <th className="pr-3 text-right">Sharpe</th>
            <th className="pr-3 text-right">Max drawdown</th>
            <th className="text-right">Costs</th>
          </tr>
        </thead>
        <tbody>
          {state.data.rows.map((r) => (
            <tr key={`${r.event}-${r.strategy}`} className="border-b border-line/50">
              <td className="py-1.5 pr-3">{label(r.event)}</td>
              <td className="pr-3">{label(r.strategy)}</td>
              <td className={`pr-3 text-right ${r.total_return < 0 ? "text-loss" : "text-gain"}`}>
                {pct(r.total_return)}
              </td>
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
