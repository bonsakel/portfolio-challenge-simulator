"use client";

import { useEffect, useState } from "react";
import { apiSend, type Issue, type ValidationReport } from "@/lib/api";
import { buildPayload, type PortfolioForm } from "@/lib/portfolio";

type State =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "local-error"; message: string }
  | { kind: "error"; message: string }
  | { kind: "report"; report: ValidationReport };

export function useValidation(form: PortfolioForm, debounceMs = 300): State {
  const [state, setState] = useState<State>({ kind: "idle" });

  useEffect(() => {
    if (form.holdings.length === 0) {
      setState({ kind: "idle" });
      return;
    }
    const built = buildPayload(form);
    if ("error" in built) {
      setState({ kind: "local-error", message: built.error });
      return;
    }
    setState({ kind: "checking" });
    const controller = new AbortController();
    const timer = setTimeout(() => {
      apiSend<ValidationReport>("POST", "/portfolios/validate", built.payload, controller.signal)
        .then((report) => setState({ kind: "report", report }))
        .catch((e: Error) => {
          if (!controller.signal.aborted) setState({ kind: "error", message: e.message });
        });
    }, debounceMs);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [form, debounceMs]);

  return state;
}

function IssueList({ issues }: { issues: Issue[] }) {
  return (
    <ul className="flex flex-col gap-1">
      {issues.map((i, k) => (
        <li key={`${i.code}-${i.symbol ?? ""}-${k}`} className={i.severity === "error" ? "text-loss" : "text-amber-300"}>
          <span className="font-medium">{i.severity === "error" ? "Error: " : "Warning: "}</span>
          {i.message}
        </li>
      ))}
    </ul>
  );
}

export function ValidationPanel({ state }: { state: State }) {
  return (
    <section aria-labelledby="validation-heading" className="rounded border border-line p-4 text-sm">
      <h3 id="validation-heading" className="font-medium mb-2">
        Constraint check
      </h3>
      <div aria-live="polite">
        {state.kind === "idle" && <p className="text-muted">Add assets to check constraints.</p>}
        {state.kind === "checking" && <p className="text-muted">Checking…</p>}
        {state.kind === "local-error" && (
          <p role="alert" className="text-loss">
            {state.message}
          </p>
        )}
        {state.kind === "error" && (
          <p role="alert" className="text-loss">
            Could not validate: {state.message}
          </p>
        )}
        {state.kind === "report" && (
          <div className="flex flex-col gap-2">
            <p className={state.report.valid ? "text-gain" : "text-loss"}>
              {state.report.valid
                ? "✓ Configuration is valid and feasible."
                : "✗ Configuration is invalid. Fix the errors below before running a backtest."}
            </p>
            {state.report.weight_sum !== null && (
              <p className="tnum text-muted">
                Weights sum to {(state.report.weight_sum * 100).toFixed(2)}%
                {state.report.min_feasible_holdings !== null &&
                  ` · cap needs ≥ ${state.report.min_feasible_holdings} assets`}
              </p>
            )}
            <IssueList issues={state.report.issues} />
          </div>
        )}
      </div>
    </section>
  );
}
