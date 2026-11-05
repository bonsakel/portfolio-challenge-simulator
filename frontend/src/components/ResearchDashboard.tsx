"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  apiGet,
  type ResearchDashboardData,
  type ResearchDashboardEvent,
  type ResearchDashboardStrategy,
  type ResearchPipeline,
} from "@/lib/api";
import { label, money, pct } from "@/lib/format";

const EVENTS = [
  { key: "global_financial_crisis", name: "Global financial crisis" },
  { key: "covid_crash", name: "COVID-19 crash" },
  { key: "2022_downturn", name: "2022 downturn" },
] as const;
const SETUP_STEPS = ["Market downturns", "Strategies", "Starting capital", "Date ranges", "Review"];
const STRATEGIES = [
  { key: "equal_weight", name: "Equal Weight", color: "#4263eb" },
  { key: "mean_variance", name: "Mean Variance", color: "#f08c00" },
  { key: "minimum_variance", name: "Minimum Variance", color: "#2f9e74" },
  { key: "ppo", name: "PPO", color: "#9b59b6" },
  { key: "ddpg", name: "DDPG", color: "#e05263" },
] as const;
const EVENT_NAMES = Object.fromEntries(EVENTS.map(({ key, name }) => [key, name]));
const STRATEGY_BY_KEY = Object.fromEntries(STRATEGIES.map((strategy) => [strategy.key, strategy]));
type PlotTrace = Record<string, unknown>;
type PlotLayout = Record<string, unknown>;
type DateRange = [string, string];
type AnalysisView = "drawdown" | "risk_return" | "costs" | "turnover" | "recovery";

function downloadCSV(
  filename: string,
  columns: string[],
  rows: (string | number | null)[][],
) {
  const escape = (value: string | number | null) =>
    `"${String(value ?? "").replaceAll('"', '""')}"`;
  const content = [columns, ...rows].map((row) => row.map(escape).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

function DownloadTable({ filename, columns, rows }: {
  filename: string;
  columns: string[];
  rows: (string | number | null)[][];
}) {
  return (
    <button
      type="button"
      onClick={() => downloadCSV(filename, columns, rows)}
      className="rounded-md border border-line px-2 py-1 text-xs text-muted hover:border-accent hover:text-accent"
    >
      Download CSV
    </button>
  );
}

function plotCSV(traces: PlotTrace[], filename: string) {
  const matrix = traces.find((trace) => Array.isArray(trace.x) && Array.isArray(trace.y) && Array.isArray(trace.z));
  if (matrix) {
    const x = matrix.x as string[];
    const y = matrix.y as string[];
    const z = matrix.z as number[][];
    downloadCSV(filename, ["Event", ...x], y.map((event, index) => [event, ...z[index]]));
    return;
  }
  const lines = traces.filter((trace) =>
    typeof trace.name === "string"
    && Array.isArray(trace.x)
    && Array.isArray(trace.y)
    && trace.showlegend !== false,
  );
  const x = [...new Set(lines.flatMap((trace) => trace.x as (string | number)[]))];
  x.sort((left, right) =>
    typeof left === "number" && typeof right === "number"
      ? left - right
      : String(left).localeCompare(String(right)),
  );
  const values = lines.map((trace) => {
    const xs = trace.x as (string | number)[];
    const ys = trace.y as number[];
    return new Map(xs.map((point, index) => [point, ys[index]]));
  });
  downloadCSV(
    filename,
    ["Date / trading day", ...lines.map((trace) => String(trace.name))],
    x.map((point, index) => [point, ...values.map((line) => line.get(point) ?? null)]),
  );
}

function Plot({ title, traces, layout, height = 330 }: {
  title: string;
  traces: PlotTrace[];
  layout: PlotLayout;
  height?: number;
}) {
  const [failed, setFailed] = useState(false);
  const [plotlyReady, setPlotlyReady] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState(false);
  const plotId = `plot-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  const element = useRef<HTMLDivElement>(null);
  const plotly = useRef<typeof import("plotly.js-dist-min").default | null>(null);

  useEffect(() => {
    let active = true;
    import("plotly.js-dist-min")
      .then((module) => {
        if (!active) return;
        plotly.current = module.default;
        setPlotlyReady(true);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
      if (element.current && plotly.current) plotly.current.purge(element.current);
    };
  }, []);

  useEffect(() => {
    if (!element.current || !plotly.current) return;
    plotly.current.react(
      element.current,
      traces,
      { ...layout, height, autosize: true },
      {
        responsive: true,
        displaylogo: false,
        displayModeBar: "hover",
        modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"],
      },
    ).catch(() => setFailed(true));
  }, [height, layout, plotlyReady, traces]);

  async function downloadPNG() {
    if (!element.current || !plotly.current || exporting) return;

    setExporting(true);
    setExportError(false);
    const exportElement = document.createElement("div");
    const width = Math.max(element.current.clientWidth, 960);
    const exportHeight = height + 110;
    const marginValue = layout.margin;
    const margin = typeof marginValue === "object" && marginValue !== null
      ? marginValue as Record<string, number>
      : {};
    exportElement.style.position = "fixed";
    exportElement.style.left = "-10000px";
    exportElement.style.top = "0";
    exportElement.style.width = `${width}px`;
    exportElement.style.height = `${exportHeight}px`;
    document.body.appendChild(exportElement);

    try {
      await plotly.current.newPlot(
        exportElement,
        traces,
        {
          ...layout,
          width,
          height: exportHeight,
          autosize: false,
          title: {
            text: title,
            x: 0.02,
            xanchor: "left",
            y: 0.98,
            yanchor: "top",
            font: { color: "#1f2937", size: 22 },
          },
          margin: {
            ...margin,
            t: Math.max(margin.t ?? 0, 68),
            b: Math.max(margin.b ?? 0, 96),
          },
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
        },
        { responsive: false, displaylogo: false, staticPlot: true },
      );
      await plotly.current.downloadImage(exportElement, {
        format: "png",
        filename: plotId,
        width,
        height: exportHeight,
        scale: 4,
        setBackground: "#ffffff",
      });
    } catch {
      setExportError(true);
    } finally {
      plotly.current?.purge(exportElement);
      exportElement.remove();
      setExporting(false);
    }
  }

  if (failed) {
    return <p role="alert" className="p-5 text-sm text-loss">This chart could not be displayed.</p>;
  }
  return (
    <div>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <button
          type="button"
          onClick={downloadPNG}
          disabled={exporting || !plotlyReady}
          className="rounded-md px-2 py-1 text-xs text-muted hover:bg-bg hover:text-accent disabled:cursor-wait disabled:opacity-60"
        >
          {exporting ? "Preparing PNG…" : "Download PNG"}
        </button>
        <button
          type="button"
          onClick={() => plotCSV(traces, `${plotId}.csv`)}
          className="rounded-md px-2 py-1 text-xs text-muted hover:bg-bg hover:text-accent"
        >
          Download data
        </button>
      </div>
      {exportError && <p role="alert" className="px-2 py-1 text-xs text-loss">PNG export failed. Please try again.</p>}
      <div id={plotId} ref={element} role="img" aria-label={title} className="w-full" style={{ height }} />
    </div>
  );
}

function chartLayout(yTitle: string, extra: PlotLayout = {}): PlotLayout {
  return {
    margin: { l: 58, r: 18, t: 12, b: 42 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#647184", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.22, x: 0 },
    xaxis: { gridcolor: "#edf0f4", automargin: true, showline: false, zeroline: false },
    yaxis: { gridcolor: "#edf0f4", title: { text: yTitle }, automargin: true, zeroline: false },
    ...extra,
  };
}

function spread(values: (number | null)[]): { mean: number | null; min: number | null; max: number | null } {
  const valid = values.filter((value): value is number => value !== null && Number.isFinite(value));
  if (valid.length === 0) return { mean: null, min: null, max: null };
  return {
    mean: valid.reduce((sum, value) => sum + value, 0) / valid.length,
    min: Math.min(...valid),
    max: Math.max(...valid),
  };
}

function summarizePaths(paths: number[][]) {
  const means: number[] = [];
  const minimums: number[] = [];
  const maximums: number[] = [];
  for (let index = 0; index < (paths[0]?.length ?? 0); index += 1) {
    const values = paths.map((path) => path[index]);
    means.push(values.reduce((sum, value) => sum + value, 0) / values.length);
    minimums.push(Math.min(...values));
    maximums.push(Math.max(...values));
  }
  return { means, minimums, maximums };
}

function viewEvent(
  event: ResearchDashboardEvent,
  range: DateRange,
  initialCapital: number,
  referenceCapital: number,
  annualizationDays: number,
  riskFreeRate: number,
): ResearchDashboardEvent {
  const firstStrategy = Object.values(event.strategies)[0];
  if (!firstStrategy) return event;
  const indexes = firstStrategy.series.dates.flatMap((date, index) =>
    date >= range[0] && date <= range[1] ? [index] : [],
  );
  if (indexes.length === 0) return event;

  const capitalScale = initialCapital / referenceCapital;
  const strategies = Object.fromEntries(Object.entries(event.strategies).map(([key, result]) => {
    const original = result.series;
    const dates = indexes.map((index) => original.dates[index]);
    const sampleReturns = original.sample_returns.map((sample) =>
      indexes.map((index) => sample[index]),
    );
    const sampleCosts = original.sample_transaction_costs.map((sample) =>
      indexes.map((index) => sample[index]),
    );
    const valuePaths = sampleReturns.map((returns) => {
      let value = initialCapital;
      return returns.map((dailyReturn) => (value *= 1 + dailyReturn));
    });
    const drawdownPaths = valuePaths.map((path) => {
      let peak = initialCapital;
      return path.map((value) => {
        peak = Math.max(peak, value);
        return value / peak - 1;
      });
    });
    const costPaths = sampleCosts.map((costs) => {
      let cumulative = 0;
      return costs.map((cost) => (cumulative += cost * capitalScale));
    });
    const valueSummary = summarizePaths(valuePaths);
    const drawdownSummary = summarizePaths(drawdownPaths);
    const costSummary = summarizePaths(costPaths);

    const sampleMetrics = sampleReturns.map((returns, sampleIndex) => {
      const growth = returns.reduce((product, value) => product * (1 + value), 1);
      const meanExcess = returns.reduce((sum, value) => sum + value - riskFreeRate, 0) / returns.length;
      const variance = returns.length > 1
        ? returns.reduce((sum, value) => sum + (value - meanExcess - riskFreeRate) ** 2, 0) / (returns.length - 1)
        : 0;
      const dailyDeviation = Math.sqrt(variance);
      let wealth = 1;
      let peak = 1;
      let maximumDrawdown = 0;
      returns.forEach((value) => {
        wealth *= 1 + value;
        peak = Math.max(peak, wealth);
        maximumDrawdown = Math.min(maximumDrawdown, wealth / peak - 1);
      });
      return {
        total_return: growth - 1,
        annualized_return: growth > 0 ? growth ** (annualizationDays / returns.length) - 1 : null,
        annualized_volatility: dailyDeviation * Math.sqrt(annualizationDays),
        sharpe_ratio: dailyDeviation > 0
          ? meanExcess / dailyDeviation * Math.sqrt(annualizationDays)
          : null,
        maximum_drawdown: maximumDrawdown,
        transaction_costs: sampleCosts[sampleIndex].reduce((sum, value) => sum + value, 0) * capitalScale,
      };
    });
    const metrics = Object.fromEntries(
      Object.keys(sampleMetrics[0]).map((metric) => [
        metric,
        spread(sampleMetrics.map((row) => row[metric as keyof typeof row])),
      ]),
    ) as ResearchDashboardStrategy["metrics"];

    return [key, {
      ...result,
      metrics,
      series: {
        ...original,
        dates,
        sample_returns: sampleReturns,
        sample_transaction_costs: sampleCosts,
        portfolio_value: valueSummary.means,
        value_min: valueSummary.minimums,
        value_max: valueSummary.maximums,
        drawdown: drawdownSummary.means,
        drawdown_min: drawdownSummary.minimums,
        drawdown_max: drawdownSummary.maximums,
        cumulative_cost: costSummary.means,
        turnover: indexes.map((index) => original.turnover[index]),
        weights: Object.fromEntries(
          Object.entries(original.weights).map(([asset, values]) => [
            asset,
            indexes.map((index) => values[index]),
          ]),
        ),
      },
    }];
  }));
  return { ...event, strategies };
}

function bandTraces(
  name: string,
  dates: string[],
  lower: number[],
  upper: number[],
  color: string,
): PlotTrace[] {
  if (!lower.some((value, i) => value !== upper[i])) return [];
  return [
    {
      type: "scatter",
      mode: "lines",
      x: dates,
      y: lower,
      line: { width: 0, color },
      hoverinfo: "skip",
      showlegend: false,
    },
    {
      type: "scatter",
      mode: "lines",
      x: dates,
      y: upper,
      line: { width: 0, color },
      fill: "tonexty",
      fillcolor: `${color}22`,
      hoverinfo: "skip",
      showlegend: false,
      name: `${name} seed range`,
    },
  ];
}

function DownturnChart({ event, autoPlay = false }: {
  event: ResearchDashboardEvent;
  autoPlay?: boolean;
}) {
  const [playing, setPlaying] = useState(autoPlay);
  const firstStrategy = Object.values(event.strategies)[0];
  const length = firstStrategy?.series.dates.length ?? 0;
  const [progress, setProgress] = useState(autoPlay ? 0 : Math.max(0, length - 1));

  useEffect(() => {
    setProgress(autoPlay ? 0 : Math.max(0, length - 1));
    setPlaying(autoPlay);
  }, [event.event, autoPlay]);

  useEffect(() => {
    setProgress((current) => Math.min(current, Math.max(0, length - 1)));
    if (length < 2) setPlaying(false);
  }, [length]);

  useEffect(() => {
    if (!playing || length < 2) return;
    const timer = window.setInterval(() => {
      setProgress((current) => {
        if (current >= length - 1) {
          setPlaying(false);
          return current;
        }
        return Math.min(length - 1, current + Math.max(1, Math.ceil(length / 180)));
      });
    }, 70);
    return () => window.clearInterval(timer);
  }, [length, playing]);

  const visibleCount = Math.min(length, progress + 1);
  const traces: PlotTrace[] = [];
  Object.entries(event.strategies).forEach(([key, result]) => {
    const strategy = STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY];
    if (!strategy) return;
    const series = result.series;
    const dates = series.dates.slice(0, visibleCount);
    traces.push(
      ...bandTraces(
        strategy.name,
        dates,
        series.value_min.slice(0, visibleCount),
        series.value_max.slice(0, visibleCount),
        strategy.color,
      ),
      {
        type: "scatter",
        mode: "lines",
        name: strategy.name,
        x: dates,
        y: series.portfolio_value.slice(0, visibleCount),
        line: { color: strategy.color, width: 2.5 },
        hovertemplate: "%{y:$,.0f}<extra>%{fullData.name}</extra>",
      },
    );
  });
  const downturn = event.windows.downturn_test;
  const selectedStart = firstStrategy?.series.dates[0] ?? downturn?.[0];
  const selectedEnd = firstStrategy?.series.dates.at(-1) ?? downturn?.[1];
  const shadeStart = downturn && selectedStart ? (selectedStart > downturn[0] ? selectedStart : downturn[0]) : null;
  const shadeEnd = downturn && selectedEnd ? (selectedEnd < downturn[1] ? selectedEnd : downturn[1]) : null;
  const layout = chartLayout("Portfolio value", {
    yaxis: {
      gridcolor: "#edf0f4",
      title: { text: "Portfolio value" },
      tickprefix: "$",
      tickformat: ",.0f",
      automargin: true,
    },
    shapes: shadeStart && shadeEnd && shadeStart <= shadeEnd ? [{
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: shadeStart,
      x1: shadeEnd,
      y0: 0,
      y1: 1,
      fillcolor: "rgba(224, 82, 99, 0.08)",
      line: { width: 0 },
      layer: "below",
    }] : [],
    annotations: shadeStart && shadeEnd && shadeStart <= shadeEnd ? [{
      x: shadeStart,
      y: 1,
      xref: "x",
      yref: "paper",
      text: "Downturn",
      showarrow: false,
      xanchor: "left",
      yanchor: "bottom",
      font: { size: 10, color: "#bd3b48" },
    }] : [],
  });

  return (
    <section className="rounded-xl border border-line bg-panel p-4 md:p-5" aria-labelledby={`performance-${event.event}`}>
      <div className="mb-1 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id={`performance-${event.event}`} className="font-semibold">{EVENT_NAMES[event.event] ?? label(event.event)}</h3>
          <p className="mt-1 text-xs text-muted">
            {selectedStart} – {selectedEnd} · shaded area marks the downturn
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            if (progress >= length - 1) setProgress(0);
            setPlaying((current) => !current);
          }}
          disabled={length < 2}
          className="rounded-lg border border-line px-3 py-1.5 text-sm font-medium hover:border-accent hover:text-accent disabled:opacity-50"
        >
          {playing ? "Pause" : "Play"}
        </button>
      </div>
      <Plot title={`Portfolio value — ${EVENT_NAMES[event.event] ?? label(event.event)}`} traces={traces} layout={layout} height={355} />
      <div className="mt-1 flex items-center gap-3">
        <span className="w-24 shrink-0 text-xs tabular-nums text-muted">
          {firstStrategy?.series.dates[visibleCount - 1] ?? ""}
        </span>
        <input
          aria-label={`Performance date for ${EVENT_NAMES[event.event] ?? event.event}`}
          type="range"
          min={0}
          max={Math.max(0, length - 1)}
          value={Math.min(progress, Math.max(0, length - 1))}
          onChange={(e) => {
            setPlaying(false);
            setProgress(Number(e.target.value));
          }}
          className="h-1 w-full accent-accent"
        />
      </div>
    </section>
  );
}

function MetricCell({ stat, format, showRange }: {
  stat: { mean: number | null; min: number | null; max: number | null };
  format: (value: number) => string;
  showRange: boolean;
}) {
  if (stat.mean === null) return <span className="text-muted">—</span>;
  return (
    <span className="tnum">
      {format(stat.mean)}
      {showRange && stat.min !== null && stat.max !== null && stat.min !== stat.max && (
        <span className="mt-0.5 block whitespace-nowrap text-[11px] font-normal text-muted">
          {format(stat.min)} – {format(stat.max)}
        </span>
      )}
    </span>
  );
}

function MetricsTable({ event, data }: { event: ResearchDashboardEvent; data: ResearchDashboardData }) {
  const rows = Object.entries(event.strategies);
  const exportRows = rows.map(([key, result]) => [
    STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key),
    result.metrics.total_return.mean,
    result.metrics.total_return.min,
    result.metrics.total_return.max,
    result.metrics.annualized_return.mean,
    result.metrics.sharpe_ratio.mean,
    result.metrics.maximum_drawdown.mean,
    result.metrics.annualized_volatility.mean,
    result.metrics.transaction_costs.mean,
  ]);
  const showSeedRange = data.source === "five_seed_study";
  return (
    <section className="overflow-hidden rounded-xl border border-line bg-panel" aria-label={`${label(event.event)} strategy metrics`}>
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <h3 className="font-semibold">Selected-period metrics</h3>
        <DownloadTable
          filename={`${event.event}-metrics.csv`}
          columns={["Strategy", "Total return mean", "Total return min", "Total return max", "Annualized return", "Sharpe ratio", "Maximum drawdown", "Annualized volatility", "Trading costs"]}
          rows={exportRows}
        />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[790px] text-sm">
          <thead className="bg-bg text-xs text-muted">
            <tr>
              <th className="px-4 py-2.5 text-left font-medium">Strategy</th>
              <th className="px-3 py-2.5 text-right font-medium">Total return</th>
              <th className="px-3 py-2.5 text-right font-medium">Annualized</th>
              <th className="px-3 py-2.5 text-right font-medium">Sharpe</th>
              <th className="px-3 py-2.5 text-right font-medium">Max drawdown</th>
              <th className="px-3 py-2.5 text-right font-medium">Volatility</th>
              <th className="px-4 py-2.5 text-right font-medium">Trading costs</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, result]) => {
              const strategy = STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY];
              return (
                <tr key={key} className="border-t border-line/70">
                  <th scope="row" className="whitespace-nowrap px-4 py-3 text-left font-medium">
                    <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: strategy?.color }} />
                    {strategy?.name ?? label(key)}
                  </th>
                  <td className="px-3 py-3 text-right"><MetricCell stat={result.metrics.total_return} format={pct} showRange={showSeedRange && (key === "ppo" || key === "ddpg")} /></td>
                  <td className="px-3 py-3 text-right"><MetricCell stat={result.metrics.annualized_return} format={pct} showRange={showSeedRange && (key === "ppo" || key === "ddpg")} /></td>
                  <td className="px-3 py-3 text-right"><MetricCell stat={result.metrics.sharpe_ratio} format={(v) => v.toFixed(2)} showRange={showSeedRange && (key === "ppo" || key === "ddpg")} /></td>
                  <td className="px-3 py-3 text-right"><MetricCell stat={result.metrics.maximum_drawdown} format={pct} showRange={showSeedRange && (key === "ppo" || key === "ddpg")} /></td>
                  <td className="px-3 py-3 text-right"><MetricCell stat={result.metrics.annualized_volatility} format={pct} showRange={showSeedRange && (key === "ppo" || key === "ddpg")} /></td>
                  <td className="px-4 py-3 text-right"><MetricCell stat={result.metrics.transaction_costs} format={money} showRange={showSeedRange && (key === "ppo" || key === "ddpg")} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {data.source === "five_seed_study" && (
        <p className="border-t border-line px-4 py-2 text-xs text-muted">
          PPO and DDPG show the five-seed mean; the smaller line is the observed range.
        </p>
      )}
    </section>
  );
}

function DrawdownPanel({ event }: { event: ResearchDashboardEvent }) {
  const traces: PlotTrace[] = [];
  Object.entries(event.strategies).forEach(([key, result]) => {
    const strategy = STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY];
    if (!strategy) return;
    const series = result.series;
    traces.push(...bandTraces(
      strategy.name,
      series.dates,
      series.drawdown_min,
      series.drawdown_max,
      strategy.color,
    ), {
      type: "scatter",
      mode: "lines",
      name: strategy.name,
      x: series.dates,
      y: series.drawdown.map((value) => value * 100),
      line: { color: strategy.color, width: 2 },
      fill: "tozeroy",
      fillcolor: `${strategy.color}10`,
      hovertemplate: "%{y:.1f}%<extra>%{fullData.name}</extra>",
    });
  });
  return (
    <section className="rounded-xl border border-line bg-panel p-4" aria-label={`${label(event.event)} drawdown`}>
      <h3 className="mb-1 font-semibold">Drawdown</h3>
      <p className="mb-1 text-xs text-muted">Decline from each strategy’s prior peak</p>
      <Plot title={`Drawdown — ${EVENT_NAMES[event.event] ?? label(event.event)}`} traces={traces} layout={chartLayout("Drawdown (%)", {
        yaxis: { gridcolor: "#edf0f4", title: { text: "Drawdown" }, ticksuffix: "%", automargin: true },
      })} height={275} />
    </section>
  );
}

function RiskReturnPanel({ event, seedStudy }: {
  event: ResearchDashboardEvent;
  seedStudy: boolean;
}) {
  const traces: PlotTrace[] = Object.entries(event.strategies).flatMap(([key, result]) => {
    const strategy = STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY];
    const annualizedReturn = result.metrics.annualized_return;
    const volatility = result.metrics.annualized_volatility;
    if (!strategy || annualizedReturn.mean === null || volatility.mean === null) return [];
    const multiSeed = seedStudy && (key === "ppo" || key === "ddpg");
    const returnError = multiSeed && annualizedReturn.min !== null && annualizedReturn.max !== null
      ? {
          type: "data",
          symmetric: false,
          array: [annualizedReturn.max - annualizedReturn.mean],
          arrayminus: [annualizedReturn.mean - annualizedReturn.min],
          visible: true,
        }
      : undefined;
    const volatilityError = multiSeed && volatility.min !== null && volatility.max !== null
      ? {
          type: "data",
          symmetric: false,
          array: [volatility.max - volatility.mean],
          arrayminus: [volatility.mean - volatility.min],
          visible: true,
        }
      : undefined;
    return [{
      type: "scatter",
      mode: "markers",
      name: strategy.name,
      x: [volatility.mean * 100],
      y: [annualizedReturn.mean * 100],
      marker: { color: strategy.color, size: 12, line: { color: "#ffffff", width: 1.5 } },
      error_x: volatilityError,
      error_y: returnError,
      hovertemplate: "Volatility %{x:.1f}%<br>Annualized return %{y:.1f}%<extra>%{fullData.name}</extra>",
    }];
  });
  return (
    <section className="rounded-xl border border-line bg-panel p-4" aria-label={`${label(event.event)} risk and return`}>
      <h3 className="mb-1 font-semibold">Risk and return</h3>
      <p className="mb-1 text-xs text-muted">Annualized return versus volatility; seed range shown for RL</p>
      <Plot title={`Risk and return — ${EVENT_NAMES[event.event] ?? label(event.event)}`} traces={traces} layout={chartLayout("Annualized return (%)", {
        xaxis: { gridcolor: "#edf0f4", title: { text: "Annualized volatility (%)" }, ticksuffix: "%", automargin: true, zeroline: false },
        yaxis: { gridcolor: "#edf0f4", title: { text: "Annualized return (%)" }, ticksuffix: "%", automargin: true, zeroline: false },
        showlegend: false,
      })} height={275} />
    </section>
  );
}

function RecoveryCurvePanel({ event }: { event: ResearchDashboardEvent }) {
  const traces: PlotTrace[] = [];
  Object.entries(event.strategies).forEach(([key, result]) => {
    const strategy = STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY];
    if (!strategy) return;
    const series = result.series;
    const downturn = event.windows.downturn_test;
    const testIndexes = series.dates.flatMap((date, index) =>
      date >= downturn[0] && date <= downturn[1] ? [index] : [],
    );
    if (testIndexes.length === 0) return;
    const troughIndex = testIndexes.reduce(
      (lowest, index) => series.portfolio_value[index] < series.portfolio_value[lowest] ? index : lowest,
      testIndexes[0],
    );
    const trough = series.portfolio_value[troughIndex];
    const indexes = Array.from({ length: series.dates.length - troughIndex }, (_, i) => troughIndex + i);
    const days = indexes.map((index) => index - troughIndex);
    const lower = indexes.map((index) => (series.value_min[index] / trough - 1) * 100);
    const upper = indexes.map((index) => (series.value_max[index] / trough - 1) * 100);
    traces.push(
      ...bandTraces(strategy.name, days.map(String), lower, upper, strategy.color)
        .map((trace) => ({ ...trace, x: days })),
      {
        type: "scatter",
        mode: "lines",
        name: strategy.name,
        x: days,
        y: indexes.map((index) => (series.portfolio_value[index] / trough - 1) * 100),
        line: { color: strategy.color, width: 2 },
        hovertemplate: "%{x} trading days<br>%{y:.1f}%<extra>%{fullData.name}</extra>",
      },
    );
  });
  return (
    <section className="rounded-xl border border-line bg-panel p-4 md:p-5" aria-label={`${label(event.event)} recovery trajectory`}>
      <h3 className="mb-1 font-semibold">Recovery from each strategy’s trough</h3>
      <p className="mb-1 text-xs text-muted">Portfolio change from the downturn low; dashed marker is 24 months</p>
      <Plot title={`Recovery trajectory — ${EVENT_NAMES[event.event] ?? label(event.event)}`} traces={traces} layout={chartLayout("Change from trough (%)", {
        xaxis: { gridcolor: "#edf0f4", title: { text: "Trading days after trough" }, automargin: true, zeroline: false },
        yaxis: { gridcolor: "#edf0f4", title: { text: "Change from trough" }, ticksuffix: "%", automargin: true, zeroline: false },
        shapes: [{
          type: "line",
          x0: 504,
          x1: 504,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "#8a94a6", width: 1, dash: "dash" },
        }],
      })} height={300} />
    </section>
  );
}

function CostsPanel({ event }: { event: ResearchDashboardEvent }) {
  const traces = Object.entries(event.strategies).map(([key, result]) => ({
    type: "scatter",
    mode: "lines",
    name: STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key),
    x: result.series.dates,
    y: result.series.cumulative_cost,
    line: { color: STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.color, width: 2 },
    hovertemplate: "%{y:$,.0f}<extra>%{fullData.name}</extra>",
  }));
  return (
    <section className="rounded-xl border border-line bg-panel p-4" aria-label={`${label(event.event)} transaction costs`}>
      <h3 className="mb-1 font-semibold">Cumulative trading costs</h3>
      <p className="mb-1 text-xs text-muted">Fees paid over the test and recovery window</p>
      <Plot title={`Cumulative trading costs — ${EVENT_NAMES[event.event] ?? label(event.event)}`} traces={traces} layout={chartLayout("Costs", {
        yaxis: { gridcolor: "#edf0f4", title: { text: "Costs" }, tickprefix: "$", tickformat: ",.0f", automargin: true },
      })} height={275} />
    </section>
  );
}

function TurnoverPanel({ event }: { event: ResearchDashboardEvent }) {
  const first = Object.values(event.strategies)[0]?.series;
  if (!first) return null;
  const months = [...new Set(first.dates.map((date) => date.slice(0, 7)))];
  const traces = Object.entries(event.strategies).map(([key, result]) => {
    const totals = new Map<string, number>();
    result.series.dates.forEach((date, index) => {
      const month = date.slice(0, 7);
      totals.set(month, (totals.get(month) ?? 0) + result.series.turnover[index] * 100);
    });
    return {
      type: "bar",
      name: STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key),
      x: months,
      y: months.map((month) => totals.get(month) ?? 0),
      marker: { color: STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.color },
      hovertemplate: "%{y:.0f}%<extra>%{fullData.name}</extra>",
    };
  });
  return (
    <section className="rounded-xl border border-line bg-panel p-4" aria-label={`${label(event.event)} monthly turnover`}>
      <h3 className="mb-1 font-semibold">Monthly portfolio turnover</h3>
      <p className="mb-1 text-xs text-muted">Sum of trades as a share of portfolio value</p>
      <Plot title={`Monthly turnover — ${EVENT_NAMES[event.event] ?? label(event.event)}`} traces={traces} layout={chartLayout("Turnover (%)", {
        barmode: "group",
        xaxis: { gridcolor: "#edf0f4", title: { text: "Month" }, automargin: true },
        yaxis: { gridcolor: "#edf0f4", title: { text: "Turnover" }, ticksuffix: "%", automargin: true },
      })} height={275} />
    </section>
  );
}

function AllocationPanel({ event, strategyKey }: {
  event: ResearchDashboardEvent;
  strategyKey: string;
}) {
  const result = event.strategies[strategyKey];
  if (!result) return null;
  const assets = Object.keys(result.series.weights);
  const traces = assets.map((asset, index) => ({
    type: "scatter",
    mode: "lines",
    stackgroup: "allocation",
    name: asset,
    x: result.series.dates,
    y: result.series.weights[asset].map((value) => value * 100),
    line: { width: 0.6, color: ["#4263eb", "#f08c00", "#2f9e74", "#9b59b6", "#e05263", "#13a8a8", "#68788f", "#d66c99", "#88a43c"][index % 9] },
    hovertemplate: "%{y:.1f}%<extra>%{fullData.name}</extra>",
  }));
  const last = result.series.dates.length - 1;
  return (
    <section className="rounded-xl border border-line bg-panel p-4 md:p-5" aria-label="Sector allocation">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="font-semibold">Where the money went</h3>
          <p className="mt-1 text-xs text-muted">Daily portfolio weights across the nine sector ETFs</p>
        </div>
        <span className="text-sm font-medium">{STRATEGY_BY_KEY[strategyKey as keyof typeof STRATEGY_BY_KEY]?.name} · {EVENT_NAMES[event.event] ?? label(event.event)}</span>
      </div>
      <Plot title={`Sector allocation — ${EVENT_NAMES[event.event] ?? label(event.event)} — ${STRATEGY_BY_KEY[strategyKey as keyof typeof STRATEGY_BY_KEY]?.name ?? label(strategyKey)}`} traces={traces} layout={chartLayout("Portfolio weight", {
        yaxis: { range: [0, 100], gridcolor: "#edf0f4", title: { text: "Allocation" }, ticksuffix: "%", automargin: true },
      })} height={330} />
      <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-line pt-3 text-xs">
        {assets.map((asset) => (
          <span key={asset} className="text-muted">
            <b className="text-ink">{asset}</b> {pct(result.series.weights[asset][last] ?? 0, 1)}
          </span>
        ))}
        <span className="ml-auto text-muted">Weights at {result.series.dates[last]}</span>
        <DownloadTable
          filename={`${event.event}-${strategyKey}-latest-weights.csv`}
          columns={["ETF", "Weight"]}
          rows={assets.map((asset) => [asset, result.series.weights[asset][last] ?? 0])}
        />
      </div>
    </section>
  );
}

function EventHeatmap({ events, strategies }: {
  events: ResearchDashboardEvent[];
  strategies: string[];
}) {
  const strategyLabels = strategies.map((key) => STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key));
  const wrappedStrategyLabels = strategyLabels.map((name) =>
    name === "Mean Variance" ? "Mean<br>Variance"
      : name === "Minimum Variance" ? "Minimum<br>Variance"
        : name === "Equal Weight" ? "Equal<br>Weight"
          : name,
  );
  const traces: PlotTrace[] = [
    {
      type: "heatmap",
      x: strategyLabels,
      y: events.map((event) => EVENT_NAMES[event.event] ?? label(event.event)),
      z: events.map((event) => strategies.map((key) => (event.strategies[key]?.metrics.maximum_drawdown.mean ?? 0) * 100)),
      text: events.map((event) => strategies.map((key) => pct(event.strategies[key]?.metrics.maximum_drawdown.mean ?? 0))),
      texttemplate: "%{text}",
      colorscale: [[0, "#e76f51"], [0.55, "#f4d35e"], [1, "#52b788"]],
      reversescale: false,
      zmin: -70,
      zmax: 0,
      showscale: false,
      textfont: { color: "#243247", size: 12 },
      hovertemplate: "%{y}<br>%{x}<br>Max drawdown: %{text}<extra></extra>",
      xgap: 4,
      ygap: 4,
      showlegend: false,
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Larger drawdown",
      x: [],
      y: [],
      marker: { color: "#e76f51", symbol: "square", size: 12 },
      hoverinfo: "skip",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Closer to zero",
      x: [],
      y: [],
      marker: { color: "#52b788", symbol: "square", size: 12 },
      hoverinfo: "skip",
    },
  ];
  return (
    <section className="rounded-xl border border-line bg-panel p-4 md:p-5">
      <h3 className="font-semibold">Maximum drawdown across downturns</h3>
      <p className="mt-1 text-xs text-muted">Compare each strategy’s largest peak-to-trough loss. Closer to zero is better.</p>
      <Plot title="Maximum drawdown across downturns" traces={traces} layout={chartLayout("", {
      xaxis: {
        showgrid: false,
        side: "bottom",
        tickangle: 0,
        tickmode: "array",
        tickvals: strategyLabels,
        ticktext: wrappedStrategyLabels,
        tickfont: { size: 11 },
        automargin: true,
      },
      yaxis: { showgrid: false, automargin: true, tickfont: { size: 11 } },
      legend: { orientation: "h", y: -0.28, x: 0.5, xanchor: "center" },
      margin: { l: 150, r: 16, t: 10, b: 66 },
      })} height={Math.max(190, events.length * 56 + 68)} />
    </section>
  );
}

function RecoveryTable({ event }: { event: ResearchDashboardEvent }) {
  const rows = Object.entries(event.strategies);
  return (
    <section className="overflow-hidden rounded-xl border border-line bg-panel" aria-label={`${label(event.event)} recovery`}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
        <div>
          <h3 className="font-semibold">Recovery after the trough</h3>
          <p className="mt-1 text-xs text-muted">Fixed 24-month study window; not changed by the selected view dates</p>
        </div>
        <DownloadTable
          filename={`${event.event}-recovery.csv`}
          columns={["Strategy", "Return mean", "Return min", "Return max", "Recovered count", "Sample count", "Recovery time trading days"]}
          rows={rows.map(([key, result]) => [
            STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key),
            result.recovery.recovery_return_from_trough.mean,
            result.recovery.recovery_return_from_trough.min,
            result.recovery.recovery_return_from_trough.max,
            result.recovery.recovered_count,
            result.recovery.sample_count,
            result.recovery.recovery_time_trading_days.mean,
          ])}
        />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead className="bg-bg text-xs text-muted">
            <tr>
              <th className="px-4 py-2 text-left font-medium">Strategy</th>
              <th className="px-3 py-2 text-right font-medium">Return from trough</th>
              <th className="px-3 py-2 text-right font-medium">Recovered within 24 mo.</th>
              <th className="px-4 py-2 text-right font-medium">Time to recovery</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, result]) => (
              <tr key={key} className="border-t border-line/70">
                <th scope="row" className="px-4 py-2.5 text-left font-medium">
                  {STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key)}
                </th>
                <td className="px-3 py-2.5 text-right"><MetricCell stat={result.recovery.recovery_return_from_trough} format={pct} showRange={result.recovery.sample_count > 1} /></td>
                <td className="px-3 py-2.5 text-right tnum">
                  {result.recovery.recovered_count}/{result.recovery.sample_count}
                </td>
                <td className="px-4 py-2.5 text-right tnum">
                  {result.recovery.recovery_time_trading_days.mean === null
                    ? "Not recovered"
                    : `${Math.round(result.recovery.recovery_time_trading_days.mean)} days`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function toggleSelection(current: string[], key: string): string[] {
  if (current.includes(key)) return current.length === 1 ? current : current.filter((value) => value !== key);
  return [...current, key];
}

function PipelinePanel({ setupStep, showResults }: {
  setupStep: number;
  showResults: boolean;
}) {
  const [pipeline, setPipeline] = useState<ResearchPipeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    let active = true;
    let controller: AbortController | null = null;
    const refresh = () => {
      controller?.abort();
      const requestController = new AbortController();
      controller = requestController;
      apiGet<ResearchPipeline>("/research/pipeline", requestController.signal)
        .then((data) => {
          if (active) {
            setPipeline(data);
            setError(null);
          }
        })
        .catch((requestError: Error) => {
          if (active && !requestController.signal.aborted) setError(requestError.message);
        });
    };
    refresh();
    const interval = window.setInterval(refresh, 15_000);
    return () => {
      active = false;
      window.clearInterval(interval);
      controller?.abort();
    };
  }, []);

  const statusText = !showResults
    ? `Step ${setupStep + 1} of ${SETUP_STEPS.length} · ${SETUP_STEPS[setupStep]}`
    : pipeline?.status === "running"
      ? `Retraining · ${pipeline.completed_models}/${pipeline.total_models}`
      : pipeline?.robustness_ready
        ? "Five-seed results ready"
        : pipeline?.baseline_available
          ? "Baseline results ready"
          : "Preparing research";

  return (
    <aside className="self-start xl:sticky xl:top-6" aria-label="Research artifact pipeline">
      <section className="overflow-hidden rounded-xl border border-line bg-panel">
        <div className="flex items-start justify-between gap-2 p-4">
          <div>
            <h2 className="font-semibold">{showResults ? "Research pipeline" : "Your progress"}</h2>
            <p className="mt-1 text-xs text-accent">{statusText}</p>
          </div>
          <button
            type="button"
            aria-expanded={!collapsed}
            aria-label={collapsed ? "Show research pipeline" : "Compress research pipeline"}
            onClick={() => setCollapsed((value) => !value)}
            className="rounded-md border border-line px-2 py-1 text-xs text-muted hover:text-accent"
          >
            {collapsed ? "Show" : "Hide"}
          </button>
        </div>
        {!collapsed && (
          <div className="border-t border-line p-4">
            {!showResults && (
              <ol className="space-y-4">
                {SETUP_STEPS.map((step, index) => {
                  const done = index < setupStep;
                  const active = index === setupStep;
                  return (
                    <li key={step} aria-current={active ? "step" : undefined} className="relative flex gap-3">
                      {index < SETUP_STEPS.length - 1 && (
                        <span className="absolute left-[11px] top-6 h-[calc(100%+8px)] w-px bg-line" aria-hidden="true" />
                      )}
                      <span className={`z-10 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold ${done ? "border-gain bg-gain/10 text-gain" : active ? "border-accent bg-accent/10 text-accent" : "border-line text-muted"}`}>
                        {done ? "✓" : index + 1}
                      </span>
                      <div className="min-w-0 pt-0.5">
                        <p className={`text-sm ${active ? "font-semibold text-accent" : done ? "font-medium" : "text-muted"}`}>
                          {step}
                        </p>
                        <p className="mt-0.5 text-xs leading-relaxed text-muted">
                          {done ? "Complete" : active ? "Current question" : "Up next"}
                        </p>
                      </div>
                    </li>
                  );
                })}
              </ol>
            )}
            {showResults && error && <p role="alert" className="mb-3 text-xs text-loss">{error}</p>}
            {showResults && !pipeline && !error && <p role="status" className="text-xs text-muted">Checking pipeline status…</p>}
            {showResults && pipeline && (
              <ol className="space-y-4">
                {pipeline.steps.map((step, index) => {
                  const done = step.status === "complete";
                  const active = step.status === "current" || step.status === "in_progress";
                  return (
                    <li key={step.id} className="relative flex gap-3">
                      {index < pipeline.steps.length - 1 && (
                        <span className="absolute left-[11px] top-6 h-[calc(100%+8px)] w-px bg-line" aria-hidden="true" />
                      )}
                      <span className={`z-10 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold ${done ? "border-gain bg-gain/10 text-gain" : active ? "border-accent bg-accent/10 text-accent" : "border-line text-muted"}`}>
                        {done ? "✓" : index + 1}
                      </span>
                      <div className="min-w-0 pt-0.5">
                        <p className={`text-sm ${active ? "font-semibold text-accent" : done ? "font-medium" : "text-muted"}`}>
                          {step.label}
                        </p>
                        <p className="mt-0.5 text-xs leading-relaxed text-muted">{step.detail}</p>
                      </div>
                    </li>
                  );
                })}
              </ol>
            )}
          </div>
        )}
      </section>
    </aside>
  );
}

function DateRangeControl({ event, range, onChange }: {
  event: ResearchDashboardEvent;
  range: DateRange;
  onChange: (range: DateRange) => void;
}) {
  const series = Object.values(event.strategies)[0]?.series;
  const dates = series?.dates ?? [];
  const minimum = dates[0] ?? "";
  const maximum = dates.at(-1) ?? minimum;
  const prefix = event.event.replace(/[^a-z0-9-]/gi, "-");
  return (
    <div className="flex flex-wrap items-end gap-2">
      <label className="text-xs text-muted" htmlFor={`range-start-${prefix}`}>
        From
        <input
          id={`range-start-${prefix}`}
          aria-label={`View from ${EVENT_NAMES[event.event] ?? event.event}`}
          type="date"
          min={minimum}
          max={maximum}
          value={range[0]}
          onChange={(e) => onChange([e.target.value, e.target.value > range[1] ? e.target.value : range[1]])}
          className="mt-1 block rounded-md border border-line bg-panel px-2 py-1.5 text-sm text-ink"
        />
      </label>
      <label className="text-xs text-muted" htmlFor={`range-end-${prefix}`}>
        Through
        <input
          id={`range-end-${prefix}`}
          aria-label={`View through ${EVENT_NAMES[event.event] ?? event.event}`}
          type="date"
          min={minimum}
          max={maximum}
          value={range[1]}
          onChange={(e) => onChange([e.target.value < range[0] ? e.target.value : range[0], e.target.value])}
          className="mt-1 block rounded-md border border-line bg-panel px-2 py-1.5 text-sm text-ink"
        />
      </label>
    </div>
  );
}

export function ResearchDashboard() {
  const [selectedEvents, setSelectedEvents] = useState<string[]>(EVENTS.map((event) => event.key));
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(STRATEGIES.map((strategy) => strategy.key));
  const [allocationEvent, setAllocationEvent] = useState<string>(EVENTS[0].key);
  const [allocationStrategy, setAllocationStrategy] = useState<string>(STRATEGIES[0].key);
  const [analysisEvent, setAnalysisEvent] = useState<string>(EVENTS[0].key);
  const [analysisView, setAnalysisView] = useState<AnalysisView>("drawdown");
  const [dateRanges, setDateRanges] = useState<Record<string, DateRange>>({});
  const [capitalInput, setCapitalInput] = useState("100000");
  const [setupStep, setSetupStep] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [activeEvent, setActiveEvent] = useState<string>(EVENTS[0].key);
  const [retryCount, setRetryCount] = useState(0);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [state, setState] = useState<
    | { kind: "loading" }
    | { kind: "error"; message: string }
    | { kind: "ready"; data: ResearchDashboardData }
  >({ kind: "loading" });

  const eventKey = selectedEvents.join(",");
  const strategyKey = selectedStrategies.join(",");
  useEffect(() => {
    if (!eventKey || !strategyKey) return;
    const controller = new AbortController();
    setState({ kind: "loading" });
    const query = new URLSearchParams();
    eventKey.split(",").forEach((value) => query.append("events", value));
    strategyKey.split(",").forEach((value) => query.append("strategies", value));
    apiGet<ResearchDashboardData>(`/research/dashboard?${query}`, controller.signal)
      .then((data) => setState({ kind: "ready", data }))
      .catch((error: Error) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: error.message });
      });
    return () => controller.abort();
  }, [eventKey, strategyKey, retryCount]);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updatePreference = () => setReducedMotion(preference.matches);
    updatePreference();
    preference.addEventListener("change", updatePreference);
    return () => preference.removeEventListener("change", updatePreference);
  }, []);

  const readyData = state.kind === "ready" ? state.data : null;
  const initialCapital = Number(capitalInput);
  const validCapital = Number.isFinite(initialCapital) && initialCapital >= 1_000 && initialCapital <= 10_000_000;

  useEffect(() => {
    if (!readyData) return;
    setDateRanges((current) => {
      const next = { ...current };
      for (const event of readyData.events) {
        if (next[event.event]) continue;
        const dates = Object.values(event.strategies)[0]?.series.dates ?? [];
        if (dates.length > 0) next[event.event] = [dates[0], dates[dates.length - 1]];
      }
      return next;
    });
  }, [readyData]);

  const availableEvents = useMemo(() => {
    if (!readyData || !validCapital) return [];
    return readyData.events.map((event) => {
      const dates = Object.values(event.strategies)[0]?.series.dates ?? [];
      const range = dateRanges[event.event] ?? [dates[0] ?? "", dates[dates.length - 1] ?? ""];
      return viewEvent(
        event,
        range,
        initialCapital,
        readyData.reference_capital,
        readyData.annualization_trading_days,
        readyData.risk_free_rate_daily,
      );
    });
  }, [dateRanges, initialCapital, readyData, validCapital]);
  const allocation = useMemo(() => {
    const event = availableEvents.find((item) => item.event === allocationEvent) ?? availableEvents[0];
    const strategy = event?.strategies[allocationStrategy] ? allocationStrategy : Object.keys(event?.strategies ?? {})[0];
    return event && strategy ? { event, strategy } : null;
  }, [allocationEvent, allocationStrategy, availableEvents]);
  const detailEvent = availableEvents.find((event) => event.event === analysisEvent) ?? availableEvents[0];
  const rawDetailEvent = readyData?.events.find((event) => event.event === detailEvent?.event) ?? detailEvent;
  const currentDateRange = (event: ResearchDashboardEvent): DateRange => {
    const dates = Object.values(event.strategies)[0]?.series.dates ?? [];
    return dateRanges[event.event] ?? [dates[0] ?? "", dates.at(-1) ?? dates[0] ?? ""];
  };
  const rawSelectedEvents = readyData
    ? selectedEvents.flatMap((key) => readyData.events.filter((event) => event.event === key))
    : [];
  const activeEventData = availableEvents.find((event) => event.event === activeEvent) ?? availableEvents[0];
  const canAdvance = setupStep < 2
    || (setupStep === 2 && validCapital)
    || (setupStep >= 3 && state.kind === "ready" && validCapital && rawSelectedEvents.length === selectedEvents.length);

  function advanceSetup() {
    if (!canAdvance) return;
    if (setupStep < SETUP_STEPS.length - 1) {
      setSetupStep((step) => step + 1);
      return;
    }

    const firstEvent = selectedEvents[0];
    setActiveEvent(firstEvent);
    setAnalysisEvent(firstEvent);
    setAllocationEvent(firstEvent);
    setShowResults(true);
  }

  return (
    <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_300px]">
      <div className="min-w-0 space-y-6">
        {!showResults ? (
          <>
            <header className="max-w-3xl">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-accent">Portfolio strategy study</p>
              <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Set up your comparison</h1>
              <p className="mt-2 text-sm text-muted">Answer a few quick questions to explore how strategies performed.</p>
            </header>

            <section className="space-y-6 rounded-xl border border-line bg-panel p-5 md:p-7" aria-label={`Setup step ${setupStep + 1}: ${SETUP_STEPS[setupStep]}`}>
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-accent">Step {setupStep + 1} of {SETUP_STEPS.length}</p>
                <p className="text-xs text-muted">{SETUP_STEPS[setupStep]}</p>
              </div>

              {setupStep === 0 && (
                <div>
                  <h2 className="text-xl font-semibold">Which downturns do you want to compare?</h2>
                  <p className="mt-1 text-sm text-muted">Choose one or more historical periods.</p>
                  <div className="mt-5 grid gap-3 sm:grid-cols-3">
                    {EVENTS.map((event) => {
                      const selected = selectedEvents.includes(event.key);
                      const period = event.key === "global_financial_crisis" ? "2007–2011"
                        : event.key === "covid_crash" ? "2020–2022"
                          : "2022–2024";
                      return (
                        <button
                          type="button"
                          key={event.key}
                          aria-pressed={selected}
                          onClick={() => setSelectedEvents((current) => toggleSelection(current, event.key))}
                          className={`min-h-24 rounded-xl border p-4 text-left transition-colors ${selected ? "border-accent bg-accent/5" : "border-line hover:border-accent/50"}`}
                        >
                          <span className="flex items-center justify-between gap-2 font-medium">
                            {event.name}
                            <span className={`flex h-5 w-5 items-center justify-center rounded-full border text-xs ${selected ? "border-accent bg-accent text-white" : "border-line text-transparent"}`}>✓</span>
                          </span>
                          <span className="mt-2 block text-sm text-muted">{period}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {setupStep === 1 && (
                <div>
                  <h2 className="text-xl font-semibold">Which strategies should we include?</h2>
                  <p className="mt-1 text-sm text-muted">Select at least one. All five are selected by default.</p>
                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    {STRATEGIES.map((strategy) => {
                      const selected = selectedStrategies.includes(strategy.key);
                      return (
                        <button
                          type="button"
                          key={strategy.key}
                          aria-pressed={selected}
                          onClick={() => setSelectedStrategies((current) => toggleSelection(current, strategy.key))}
                          className={`flex items-center gap-3 rounded-xl border p-4 text-left transition-colors ${selected ? "border-accent bg-accent/5" : "border-line opacity-70 hover:opacity-100"}`}
                        >
                          <span className="h-3 w-3 shrink-0 rounded-full" style={{ backgroundColor: strategy.color }} />
                          <span className="font-medium">{strategy.name}</span>
                          <span className={`ml-auto text-sm ${selected ? "text-accent" : "text-muted"}`}>{selected ? "Selected" : "Add"}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {setupStep === 2 && (
                <div>
                  <h2 className="text-xl font-semibold">How much would you like to invest?</h2>
                  <p className="mt-1 text-sm text-muted">We’ll rebase the saved results to your starting amount.</p>
                  <label htmlFor="initial-capital" className="mt-5 block max-w-sm text-sm font-medium">
                    Starting capital
                    <span className="mt-1 flex items-center rounded-lg border border-line bg-bg px-3">
                      <span className="text-muted">$</span>
                      <input
                        id="initial-capital"
                        aria-label="Starting capital"
                        type="number"
                        min={1_000}
                        max={10_000_000}
                        step={1_000}
                        value={capitalInput}
                        onChange={(e) => setCapitalInput(e.target.value)}
                        className="w-full bg-transparent px-2 py-3 text-base tabular-nums outline-none"
                      />
                    </span>
                  </label>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {[25_000, 100_000, 250_000, 500_000].map((amount) => (
                      <button
                        type="button"
                        key={amount}
                        aria-pressed={capitalInput === String(amount)}
                        onClick={() => setCapitalInput(String(amount))}
                        className={`rounded-full border px-3 py-1.5 text-xs ${capitalInput === String(amount) ? "border-accent bg-accent/5 text-accent" : "border-line text-muted hover:text-ink"}`}
                      >
                        {money(amount)}
                      </button>
                    ))}
                  </div>
                  {!validCapital && (
                    <p role="alert" className="mt-3 text-sm text-loss">Enter an amount between $1,000 and $10,000,000.</p>
                  )}
                  <p className="mt-4 max-w-xl text-xs leading-relaxed text-muted">
                    Fees (0.10%), the 30% position cap, and model training settings stay fixed so the comparison remains consistent.
                  </p>
                </div>
              )}

              {setupStep === 3 && (
                <div>
                  <h2 className="text-xl font-semibold">Which dates should the charts show?</h2>
                  <p className="mt-1 text-sm text-muted">The full available period is selected by default. You can adjust dates later.</p>
                  {state.kind === "loading" && (
                    <p role="status" className="mt-5 text-sm text-muted">Loading the saved date ranges…</p>
                  )}
                  {state.kind === "ready" && rawSelectedEvents.length > 0 && (
                    <div className="mt-5 space-y-4">
                      {rawSelectedEvents.map((event) => (
                        <div key={event.event} className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-bg p-3">
                          <span className="text-sm font-medium">{EVENT_NAMES[event.event] ?? label(event.event)}</span>
                          <DateRangeControl
                            event={event}
                            range={currentDateRange(event)}
                            onChange={(next) => setDateRanges((current) => ({ ...current, [event.event]: next }))}
                          />
                        </div>
                      ))}
                      <button
                        type="button"
                        onClick={() => setDateRanges((current) => {
                          const next = { ...current };
                          for (const event of rawSelectedEvents) {
                            const dates = Object.values(event.strategies)[0]?.series.dates ?? [];
                            if (dates.length > 0) next[event.event] = [dates[0], dates[dates.length - 1]];
                          }
                          return next;
                        })}
                        className="text-xs font-medium text-accent hover:underline"
                      >
                        Reset to full available periods
                      </button>
                    </div>
                  )}
                  {state.kind === "error" && (
                    <div className="mt-5 rounded-lg border border-loss/30 bg-loss/5 p-3">
                      <p role="alert" className="text-sm text-loss">Saved date ranges could not be loaded: {state.message}</p>
                      <button type="button" onClick={() => setRetryCount((count) => count + 1)} className="mt-2 text-sm font-medium text-accent">
                        Retry
                      </button>
                    </div>
                  )}
                </div>
              )}

              {setupStep === 4 && (
                <div>
                  <h2 className="text-xl font-semibold">Ready to compare</h2>
                  <p className="mt-1 text-sm text-muted">Check your choices, then watch the selected strategy paths unfold.</p>
                  <dl className="mt-5 divide-y divide-line rounded-lg border border-line">
                    <div className="grid gap-1 p-3 sm:grid-cols-[140px_1fr]">
                      <dt className="text-xs font-medium text-muted">Downturns</dt>
                      <dd className="text-sm">{selectedEvents.map((key) => EVENT_NAMES[key] ?? label(key)).join(", ")}</dd>
                    </div>
                    <div className="grid gap-1 p-3 sm:grid-cols-[140px_1fr]">
                      <dt className="text-xs font-medium text-muted">Strategies</dt>
                      <dd className="text-sm">{selectedStrategies.map((key) => STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key)).join(", ")}</dd>
                    </div>
                    <div className="grid gap-1 p-3 sm:grid-cols-[140px_1fr]">
                      <dt className="text-xs font-medium text-muted">Starting capital</dt>
                      <dd className="text-sm">{money(initialCapital)}</dd>
                    </div>
                    <div className="grid gap-1 p-3 sm:grid-cols-[140px_1fr]">
                      <dt className="text-xs font-medium text-muted">Date windows</dt>
                      <dd className="space-y-1 text-sm">
                        {rawSelectedEvents.map((event) => {
                          const range = currentDateRange(event);
                          return <p key={event.event}>{EVENT_NAMES[event.event] ?? label(event.event)}: {range[0]} – {range[1]}</p>;
                        })}
                      </dd>
                    </div>
                  </dl>
                </div>
              )}

              {state.kind === "error" && setupStep !== 3 && (
                <div className="rounded-lg border border-loss/30 bg-loss/5 p-3">
                  <p role="alert" className="text-sm text-loss">Research results could not be loaded: {state.message}</p>
                  <button type="button" onClick={() => setRetryCount((count) => count + 1)} className="mt-2 text-sm font-medium text-accent">
                    Retry
                  </button>
                </div>
              )}

              <div className="flex items-center justify-between border-t border-line pt-4">
                <button
                  type="button"
                  disabled={setupStep === 0}
                  onClick={() => setSetupStep((step) => Math.max(0, step - 1))}
                  className="rounded-lg border border-line px-4 py-2 text-sm font-medium hover:border-accent disabled:invisible"
                >
                  Back
                </button>
                <button
                  type="button"
                  disabled={!canAdvance}
                  onClick={advanceSetup}
                  className="rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-white hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {setupStep === SETUP_STEPS.length - 1 ? "Show results" : "Continue"}
                </button>
              </div>
            </section>
          </>
        ) : (
          <>
            <header className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-accent">Portfolio strategy study</p>
                <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Strategy comparison</h1>
              </div>
              <button
                type="button"
                onClick={() => {
                  setShowResults(false);
                  setSetupStep(SETUP_STEPS.length - 1);
                }}
                className="rounded-lg border border-line px-3 py-2 text-sm font-medium hover:border-accent hover:text-accent"
              >
                Change setup
              </button>
            </header>

            {state.kind === "loading" && (
              <p role="status" className="rounded-xl border border-line bg-panel p-5 text-sm text-muted">Loading selected research results…</p>
            )}
            {state.kind === "error" && (
              <div className="rounded-xl border border-line bg-panel p-5">
                <p role="alert" className="text-sm text-loss">Research results could not be loaded: {state.message}</p>
                <button type="button" onClick={() => setRetryCount((count) => count + 1)} className="mt-3 text-sm font-medium text-accent">Retry</button>
              </div>
            )}
            {readyData && validCapital && (
              <>
                <p className="text-xs text-muted">
                  {readyData.source === "five_seed_study"
                    ? `Values rebased to ${money(initialCapital)} · PPO / DDPG show the ${readyData.seed_count}-seed mean and range.`
                    : `Values rebased to ${money(initialCapital)} · currently showing original single-seed RL results.`}
                </p>

                <section aria-label="Selected downturn results" className="space-y-4">
                  <div className="flex flex-wrap gap-2" role="group" aria-label="Choose downturn results">
                    {availableEvents.map((event) => {
                      const selected = activeEventData?.event === event.event;
                      return (
                        <button
                          type="button"
                          aria-pressed={selected}
                          key={event.event}
                          onClick={() => {
                            setActiveEvent(event.event);
                            setAnalysisEvent(event.event);
                            setAllocationEvent(event.event);
                          }}
                          className={`rounded-lg border px-3 py-2 text-sm ${selected ? "border-accent bg-accent/5 font-medium text-accent" : "border-line text-muted hover:text-ink"}`}
                        >
                          {EVENT_NAMES[event.event] ?? label(event.event)}
                        </button>
                      );
                    })}
                  </div>
                  {activeEventData && (
                    <>
                      <div className="flex flex-wrap items-end justify-between gap-3">
                        <div>
                          <h2 className="text-xl font-semibold">{EVENT_NAMES[activeEventData.event] ?? label(activeEventData.event)}</h2>
                          <p className="mt-1 text-xs text-muted">Performance animates across the selected date range.</p>
                        </div>
                        <DateRangeControl
                          event={activeEventData}
                          range={currentDateRange(activeEventData)}
                          onChange={(next) => setDateRanges((current) => ({ ...current, [activeEventData.event]: next }))}
                        />
                      </div>
                      <DownturnChart
                        key={activeEventData.event}
                        event={activeEventData}
                        autoPlay={!reducedMotion && activeEventData.event === selectedEvents[0]}
                      />
                      {reducedMotion && activeEventData.event === selectedEvents[0] && (
                        <p className="text-xs text-muted">Animation is paused to respect your reduced-motion setting. Press Play to start it.</p>
                      )}
                      <MetricsTable event={activeEventData} data={readyData} />
                    </>
                  )}
                </section>

                {detailEvent && rawDetailEvent && (
                  <section className="space-y-4 rounded-xl border border-line bg-bg p-4 md:p-5" aria-label="Detailed result view">
                    <div className="flex flex-wrap items-end justify-between gap-3">
                      <div>
                        <h2 className="font-semibold">Explore detailed results</h2>
                        <p className="mt-1 text-xs text-muted">Choose one view at a time; it uses the selected event and date range.</p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <label className="sr-only" htmlFor="detail-event">Detailed view downturn</label>
                        <select
                          id="detail-event"
                          className="rounded-lg border border-line bg-panel px-3 py-2 text-sm"
                          value={detailEvent.event}
                          onChange={(e) => setAnalysisEvent(e.target.value)}
                        >
                          {availableEvents.map((event) => (
                            <option key={event.event} value={event.event}>{EVENT_NAMES[event.event] ?? label(event.event)}</option>
                          ))}
                        </select>
                        <label className="sr-only" htmlFor="detail-view">Detailed chart</label>
                        <select
                          id="detail-view"
                          className="rounded-lg border border-line bg-panel px-3 py-2 text-sm"
                          value={analysisView}
                          onChange={(e) => setAnalysisView(e.target.value as AnalysisView)}
                        >
                          <option value="drawdown">Drawdown</option>
                          <option value="risk_return">Risk and return</option>
                          <option value="costs">Trading costs</option>
                          <option value="turnover">Monthly turnover</option>
                          <option value="recovery">Recovery</option>
                        </select>
                      </div>
                    </div>
                    {analysisView === "drawdown" && <DrawdownPanel event={detailEvent} />}
                    {analysisView === "risk_return" && <RiskReturnPanel event={detailEvent} seedStudy={readyData.source === "five_seed_study"} />}
                    {analysisView === "costs" && <CostsPanel event={detailEvent} />}
                    {analysisView === "turnover" && <TurnoverPanel event={detailEvent} />}
                    {analysisView === "recovery" && (
                      <div className="space-y-4">
                        <RecoveryCurvePanel event={rawDetailEvent} />
                        <RecoveryTable event={rawDetailEvent} />
                      </div>
                    )}
                  </section>
                )}

                <div className="grid gap-4 lg:grid-cols-2">
                  <EventHeatmap events={availableEvents} strategies={readyData.strategies} />
                  {allocation && (
                    <div className="space-y-3">
                      <div className="flex flex-wrap gap-2">
                        <label className="sr-only" htmlFor="allocation-event">Allocation downturn</label>
                        <select
                          id="allocation-event"
                          className="rounded-lg border border-line bg-panel px-3 py-2 text-sm"
                          value={allocation.event.event}
                          onChange={(e) => setAllocationEvent(e.target.value)}
                        >
                          {availableEvents.map((event) => (
                            <option key={event.event} value={event.event}>{EVENT_NAMES[event.event] ?? label(event.event)}</option>
                          ))}
                        </select>
                        <label className="sr-only" htmlFor="allocation-strategy">Allocation strategy</label>
                        <select
                          id="allocation-strategy"
                          className="rounded-lg border border-line bg-panel px-3 py-2 text-sm"
                          value={allocation.strategy}
                          onChange={(e) => setAllocationStrategy(e.target.value)}
                        >
                          {Object.keys(allocation.event.strategies).map((key) => (
                            <option key={key} value={key}>{STRATEGY_BY_KEY[key as keyof typeof STRATEGY_BY_KEY]?.name ?? label(key)}</option>
                          ))}
                        </select>
                      </div>
                      <AllocationPanel event={allocation.event} strategyKey={allocation.strategy} />
                    </div>
                  )}
                </div>

                <details className="rounded-xl border border-line bg-panel px-4 py-3">
                  <summary className="cursor-pointer text-sm font-medium">Study setup</summary>
                  <div className="mt-4 grid gap-3 text-sm sm:grid-cols-4">
                    {[
                      ["Market data", "Nine US sector ETFs; VIX informs RL state"],
                      ["Common rules", "0.10% trading cost · 30% position cap"],
                      ["Traditional", "Equal Weight monthly; Mean and Minimum Variance daily"],
                      ["RL evaluation", "Train → validate → freeze → test downturn and recovery"],
                    ].map(([title, detail], index) => (
                      <div key={title} className="rounded-lg bg-bg p-3">
                        <div className="mb-1 text-xs font-semibold text-accent">{String(index + 1).padStart(2, "0")} · {title}</div>
                        <p className="text-xs leading-relaxed text-muted">{detail}</p>
                      </div>
                    ))}
                  </div>
                  <p className="mt-3 text-xs text-muted">
                    Historical results are specific to these market windows. Seed variation is not statistical proof of general superiority.
                  </p>
                </details>
              </>
            )}
          </>
        )}
      </div>
      <PipelinePanel setupStep={setupStep} showResults={showResults} />
    </div>
  );
}
