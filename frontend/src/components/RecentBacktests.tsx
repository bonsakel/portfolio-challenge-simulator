"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { apiGet, type BacktestRun } from "@/lib/api";
import { strategyName } from "@/components/ResearchMode";

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; runs: BacktestRun[] };

export function RecentBacktests() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    apiGet<BacktestRun[]>("/backtests?limit=5", controller.signal)
      .then((runs) => setState({ kind: "ready", runs }))
      .catch((e: Error) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: e.message });
      });
    return () => controller.abort();
  }, []);

  if (state.kind === "loading") return <p className="text-muted">Loading recent backtests…</p>;
  if (state.kind === "error")
    return (
      <p role="alert" className="text-loss">
        Could not load backtests: {state.message}
      </p>
    );
  if (state.runs.length === 0)
    return (
      <p className="text-sm text-muted">
        No custom backtests yet. Build a portfolio in the{" "}
        <Link href="/portfolio-builder" className="text-accent underline">
          Portfolio Builder
        </Link>
        , then run one in{" "}
        <Link href="/backtests" className="text-accent underline">
          Backtests
        </Link>
        .
      </p>
    );

  return (
    <>
      <ul className="flex flex-col gap-2 text-sm">
        {state.runs.map((run) => (
          <li key={run.id} className="flex flex-wrap gap-x-3 border-b border-line/50 pb-2">
            <span className="font-medium">
              #{run.id} {run.portfolio_name}
            </span>
            <span className="text-muted">
              {run.config.start} → {run.config.end} · {run.config.strategies.map(strategyName).join(", ")}
            </span>
            <span className={run.status === "failed" ? "text-loss" : "text-muted"}>{run.status}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-muted">
        Custom runs are exploratory. Open them in{" "}
        <Link href="/backtests" className="text-accent underline">
          Backtests
        </Link>{" "}
        or{" "}
        <Link href="/risk-analytics" className="text-accent underline">
          Risk Analytics
        </Link>
        .
      </p>
    </>
  );
}
