"use client";

import { useState } from "react";
import { apiSend, type CoverageReport } from "@/lib/api";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "report"; report: CoverageReport };

const STATUS_TEXT = { ok: "Available", unavailable: "No price history", provider_error: "Provider error" };

export function CoverageCheck({ symbols }: { symbols: string[] }) {
  const [state, setState] = useState<State>({ kind: "idle" });

  async function check() {
    setState({ kind: "loading" });
    try {
      const report = await apiSend<CoverageReport>("POST", "/data/coverage", { symbols });
      setState({ kind: "report", report });
    } catch (e) {
      setState({ kind: "error", message: (e as Error).message });
    }
  }

  return (
    <section aria-labelledby="coverage-heading" className="rounded border border-line bg-panel p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="coverage-heading" className="font-medium">
          Historical data coverage
        </h2>
        <button
          type="button"
          onClick={check}
          disabled={symbols.length === 0 || state.kind === "loading"}
          className="rounded border border-line px-3 py-1.5 text-sm hover:bg-line disabled:opacity-50"
        >
          {state.kind === "loading" ? "Checking…" : "Check data coverage"}
        </button>
      </div>
      <p className="text-xs text-muted mt-1">
        Downloads any history not yet cached (internet required). Dates missing for any asset are excluded
        for all assets; nothing is forward-filled.
      </p>

      {state.kind === "error" && (
        <p role="alert" className="mt-3 text-sm text-loss">
          Coverage check failed: {state.message}
        </p>
      )}

      {state.kind === "report" && <Report report={state.report} />}
    </section>
  );
}

function Report({ report }: { report: CoverageReport }) {
  return (
    <div className="mt-3 flex flex-col gap-3">
      <p role="status" className={`text-sm ${report.ready ? "text-gain" : "text-loss"}`}>
        {report.ready
          ? `All assets have usable history: ${report.common_start} to ${report.common_end} (${report.common_rows} common trading days).`
          : "Not ready: at least one asset has no usable history, or the common history is empty."}
        {report.ready && report.limiting_symbol && ` Shortest history: ${report.limiting_symbol}.`}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-muted border-b border-line">
            <tr>
              <th className="py-1 pr-3">Ticker</th>
              <th className="pr-3">Status</th>
              <th className="pr-3">Source</th>
              <th className="pr-3">First</th>
              <th className="pr-3">Last</th>
              <th className="pr-3 text-right">Rows</th>
              <th className="text-right">Gaps</th>
            </tr>
          </thead>
          <tbody>
            {report.symbols.map((s) => (
              <tr key={s.symbol} className="border-b border-line/50 align-top">
                <td className="py-1.5 pr-3 font-mono">{s.symbol}</td>
                <td className={`pr-3 ${s.status === "ok" ? "text-gain" : "text-loss"}`}>
                  {STATUS_TEXT[s.status]}
                  {s.message && <div className="text-xs text-muted">{s.message}</div>}
                </td>
                <td className="pr-3">{s.source ?? "—"}</td>
                <td className="pr-3 tnum">{s.first_date ?? "—"}</td>
                <td className="pr-3 tnum">{s.last_date ?? "—"}</td>
                <td className="pr-3 text-right tnum">{s.rows.toLocaleString("en-US")}</td>
                <td className="text-right tnum">{s.gaps.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {report.issues.length > 0 && (
        <ul className="list-disc pl-5 text-sm text-muted">
          {report.issues.map((i) => (
            <li key={i}>{i}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
