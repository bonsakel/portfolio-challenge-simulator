"use client";

import { useEffect, useRef, useState } from "react";
import type { ChartSpec } from "@/lib/charts";

// Plotly is loaded on demand in the browser only, so it never runs during server rendering.
function PlotlyPlot({ spec, height }: { spec: ChartSpec; height: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let alive = true;
    import("plotly.js-dist-min")
      .then((mod) => {
        if (!alive) return;
        return mod.default.react(el, spec.traces, { ...spec.layout, height, autosize: true }, {
          responsive: true,
          displaylogo: false,
        });
      })
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
      import("plotly.js-dist-min").then((mod) => mod.default.purge(el)).catch(() => undefined);
    };
  }, [spec, height]);

  if (failed) return <p className="text-sm text-loss">The chart could not be drawn. Use the table view instead.</p>;
  return <div ref={ref} role="img" aria-label={spec.title} style={{ height }} className="w-full" />;
}

function DataTable({ table }: { table: ChartSpec["table"] }) {
  return (
    <div className="max-h-80 overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-panel text-left text-muted border-b border-line">
          <tr>
            {table.columns.map((c, i) => (
              <th key={c} className={`py-1 pr-3 ${i > 0 ? "text-right" : ""}`}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, r) => (
            <tr key={r} className="border-b border-line/50">
              {row.map((cell, i) => (
                <td key={i} className={`py-1 pr-3 tnum ${i > 0 ? "text-right" : ""}`}>
                  {cell === null ? "–" : typeof cell === "number" ? cell.toLocaleString("en-US") : cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ChartCard({ spec, height = 320 }: { spec: ChartSpec; height?: number }) {
  const [view, setView] = useState<"chart" | "table">("chart");
  const headingId = `chart-${spec.id}`;
  return (
    <section aria-labelledby={headingId} className="rounded border border-line bg-panel p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 id={headingId} className="font-medium">
          {spec.title}
        </h3>
        <div className="flex gap-1 text-sm" role="group" aria-label={`${spec.title} view`}>
          {(["chart", "table"] as const).map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={view === v}
              onClick={() => setView(v)}
              className={`rounded border px-2 py-1 ${view === v ? "border-accent text-accent" : "border-line text-muted hover:bg-line"}`}
            >
              {v === "chart" ? "Chart" : "Table"}
            </button>
          ))}
        </div>
      </div>
      <p className="mt-1 mb-2 text-xs text-muted">{spec.description}</p>
      {view === "chart" ? <PlotlyPlot spec={spec} height={height} /> : <DataTable table={spec.table} />}
    </section>
  );
}
