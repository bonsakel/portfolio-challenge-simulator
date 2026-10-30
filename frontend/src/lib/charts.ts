import type { BacktestResultData, BacktestRun, BacktestSeries } from "@/lib/api";

// Plain Plotly trace/layout objects built only from calculation outputs; nothing is hardcoded.
export type Trace = Record<string, unknown>;
export type ChartSpec = {
  id: string;
  title: string;
  description: string;
  traces: Trace[];
  layout: Record<string, unknown>;
  table: { columns: string[]; rows: (string | number | null)[][] };
};

const STRATEGY_COLORS: Record<string, string> = {
  equal_weight: "#5aa9ff",
  minimum_variance: "#3ecf8e",
  mean_variance: "#f5a524",
  ppo: "#c084fc",
  ddpg: "#ff6b6b",
};
const BENCHMARK_COLOR = "#9aa5bd";
const PALETTE = ["#5aa9ff", "#3ecf8e", "#f5a524", "#c084fc", "#ff6b6b", "#22d3ee", "#f472b6", "#a3e635", "#fb923c"];

export const colorFor = (key: string) => STRATEGY_COLORS[key] ?? PALETTE[0];

type Line = { name: string; color: string; dashed?: boolean; dates: string[]; y: number[] };

export function monthEnd(dates: string[]): number[] {
  const idx: number[] = [];
  dates.forEach((d, i) => {
    if (i === dates.length - 1 || dates[i + 1].slice(0, 7) !== d.slice(0, 7)) idx.push(i);
  });
  return idx;
}

function lineTable(lines: Line[], digits: number): ChartSpec["table"] {
  const first = lines[0];
  const idx = monthEnd(first.dates);
  return {
    columns: ["Month-end date", ...lines.map((l) => l.name)],
    rows: idx.map((i) => [first.dates[i], ...lines.map((l) => Number(l.y[i].toFixed(digits)))]),
  };
}

const baseLayout = (yTitle: string, extra: Record<string, unknown> = {}) => ({
  margin: { l: 64, r: 24, t: 8, b: 40 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e6eaf2", size: 12 },
  hovermode: "x unified",
  legend: { orientation: "h", yanchor: "top", y: -0.18 },
  xaxis: { gridcolor: "#27314a", automargin: true },
  yaxis: { gridcolor: "#27314a", title: { text: yTitle }, automargin: true },
  ...extra,
});

function lineChart(
  id: string,
  title: string,
  description: string,
  lines: Line[],
  yTitle: string,
  opts: { digits?: number; yFormat?: string; hover: string; extra?: Record<string, unknown> },
): ChartSpec {
  return {
    id,
    title,
    description,
    traces: lines.map((l) => ({
      type: "scatter",
      mode: "lines",
      name: l.name,
      x: l.dates,
      y: l.y,
      line: { color: l.color, width: 2, dash: l.dashed ? "dash" : "solid" },
      hovertemplate: opts.hover,
    })),
    layout: baseLayout(yTitle, {
      ...(opts.yFormat
        ? { yaxis: { gridcolor: "#27314a", title: { text: yTitle }, tickformat: opts.yFormat, automargin: true } }
        : {}),
      ...opts.extra,
    }),
    table: lineTable(lines, opts.digits ?? 2),
  };
}

function linesFor(
  data: BacktestResultData,
  run: BacktestRun,
  pick: (s: { dates: string[] } & Record<string, unknown>) => number[],
): Line[] {
  const lines: Line[] = Object.entries(data.series).map(([key, s]) => ({
    name: run.summary?.strategies[key]?.label ?? key,
    color: colorFor(key),
    dates: s.dates,
    y: pick(s as never),
  }));
  if (data.benchmark && run.summary?.benchmark) {
    lines.push({
      name: run.summary.benchmark.label,
      color: BENCHMARK_COLOR,
      dashed: true,
      dates: data.benchmark.dates,
      y: pick(data.benchmark as never),
    });
  }
  return lines;
}

export function valueChart(data: BacktestResultData, run: BacktestRun): ChartSpec {
  return lineChart(
    "value",
    "Portfolio value",
    "Daily closing value in US dollars from the same starting capital, net of transaction costs; the benchmark is buy and hold.",
    linesFor(data, run, (s) => s.portfolio_value as number[]),
    "Value (USD)",
    { digits: 0, yFormat: "$,.0f", hover: "%{y:$,.0f}<extra>%{fullData.name}</extra>" },
  );
}

export function drawdownChart(data: BacktestResultData, run: BacktestRun): ChartSpec {
  return lineChart(
    "drawdown",
    "Drawdown",
    "Decline from the highest value reached so far (the starting capital counts as the first peak). 0% means at a peak.",
    linesFor(data, run, (s) => (s.drawdown as number[]).map((d) => d * 100)),
    "Drawdown (%)",
    { digits: 2, yFormat: ".0f", hover: "%{y:.2f}%<extra>%{fullData.name}</extra>" },
  );
}

export function costChart(data: BacktestResultData, run: BacktestRun): ChartSpec {
  const lines = Object.entries(data.series).map(([key, s]) => {
    let total = 0;
    return {
      name: run.summary?.strategies[key]?.label ?? key,
      color: colorFor(key),
      dates: s.dates,
      y: s.transaction_cost.map((c) => (total += c)),
    };
  });
  return lineChart(
    "costs",
    "Cumulative transaction costs",
    "Running total of fees paid in US dollars (proportional fee on traded value). The benchmark is buy and hold and pays none.",
    lines,
    "Cumulative cost (USD)",
    { digits: 0, yFormat: "$,.0f", hover: "%{y:$,.0f}<extra>%{fullData.name}</extra>" },
  );
}

export function turnoverChart(data: BacktestResultData, run: BacktestRun): ChartSpec {
  const entries = Object.entries(data.series);
  const months = monthEnd(entries[0][1].dates).map((i) => entries[0][1].dates[i].slice(0, 7));
  const monthly = (s: BacktestSeries) => {
    const sums = new Map<string, number>();
    s.dates.forEach((d, i) => sums.set(d.slice(0, 7), (sums.get(d.slice(0, 7)) ?? 0) + s.turnover[i]));
    return months.map((m) => Number(((sums.get(m) ?? 0) * 100).toFixed(2)));
  };
  const values = entries.map(([, s]) => monthly(s));
  return {
    id: "turnover",
    title: "Turnover by month",
    description:
      "Sum of absolute weight changes traded in each calendar month, in percent of portfolio value (100% = the whole portfolio traded once).",
    traces: entries.map(([key], i) => ({
      type: "bar",
      name: run.summary?.strategies[key]?.label ?? key,
      x: months,
      y: values[i],
      marker: { color: colorFor(key) },
      hovertemplate: "%{y:.1f}%<extra>%{fullData.name}</extra>",
    })),
    layout: baseLayout("Turnover (% of portfolio)", {
      barmode: "group",
      legend: { orientation: "h", yanchor: "top", y: -0.3 },
      xaxis: { title: { text: "Month" }, gridcolor: "#27314a", automargin: true },
    }),
    table: {
      columns: ["Month", ...entries.map(([k]) => run.summary?.strategies[k]?.label ?? k)],
      rows: months.map((m, r) => [m, ...values.map((v) => v[r])]),
    },
  };
}

export function allocationChart(data: BacktestResultData, strategyKey: string, label: string): ChartSpec {
  const s = data.series[strategyKey];
  const assets = Object.keys(s.weights);
  const idx = monthEnd(s.dates);
  return {
    id: "allocation",
    title: `Allocation over time: ${label}`,
    description: "Portfolio weights after each day's rebalance decision, stacked to 100%. Long-only and fully invested.",
    traces: assets.map((a, i) => ({
      type: "scatter",
      mode: "lines",
      stackgroup: "one",
      name: a,
      x: s.dates,
      y: s.weights[a].map((w) => w * 100),
      line: { width: 0.5, color: PALETTE[i % PALETTE.length] },
      hovertemplate: "%{y:.1f}%<extra>%{fullData.name}</extra>",
    })),
    layout: baseLayout("Weight (%)", { yaxis: { range: [0, 100], gridcolor: "#27314a", title: { text: "Weight (%)" } } }),
    table: {
      columns: ["Month-end date", ...assets],
      rows: idx.map((i) => [s.dates[i], ...assets.map((a) => Number((s.weights[a][i] * 100).toFixed(2)))]),
    },
  };
}

// Axis range with room around the points so marker labels are not clipped.
function padded(values: number[]): [number, number] {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.35, 1);
  return [lo - pad, hi + pad];
}

type Point = { name: string; color: string; vol: number; ret: number };

function points(run: BacktestRun): Point[] {
  const out: Point[] = Object.entries(run.summary?.strategies ?? {}).map(([key, m]) => ({
    name: m.label,
    color: colorFor(key),
    vol: m.annualized_volatility,
    ret: m.annualized_return,
  }));
  const b = run.summary?.benchmark;
  if (b) out.push({ name: b.label, color: BENCHMARK_COLOR, vol: b.annualized_volatility, ret: b.annualized_return });
  return out;
}

export function riskReturnChart(run: BacktestRun): ChartSpec {
  const pts = points(run);
  return {
    id: "risk-return",
    title: "Risk versus return",
    description: "Annualized volatility (x) against annualized return (y) over the whole period, 252 trading days per year.",
    traces: pts.map((p) => ({
      type: "scatter",
      mode: "markers+text",
      name: p.name,
      x: [p.vol * 100],
      y: [p.ret * 100],
      text: [p.name],
      textposition: "top center",
      marker: { size: 12, color: p.color },
      hovertemplate: "Volatility %{x:.1f}%<br>Return %{y:.1f}%<extra>%{fullData.name}</extra>",
    })),
    layout: baseLayout("Annualized return (%)", {
      hovermode: "closest",
      showlegend: false,
      xaxis: { gridcolor: "#27314a", title: { text: "Annualized volatility (%)" }, automargin: true, range: padded(pts.map((p) => p.vol * 100)) },
      yaxis: { gridcolor: "#27314a", title: { text: "Annualized return (%)" }, automargin: true, range: padded(pts.map((p) => p.ret * 100)) },
    }),
    table: {
      columns: ["Series", "Annualized volatility (%)", "Annualized return (%)"],
      rows: pts.map((p) => [p.name, Number((p.vol * 100).toFixed(2)), Number((p.ret * 100).toFixed(2))]),
    },
  };
}

export function comparisonChart(run: BacktestRun): ChartSpec {
  const rows: { name: string; color: string; total: number; dd: number }[] = Object.entries(
    run.summary?.strategies ?? {},
  ).map(([key, m]) => ({ name: m.label, color: colorFor(key), total: m.total_return, dd: m.maximum_drawdown }));
  const b = run.summary?.benchmark;
  if (b) rows.push({ name: b.label, color: BENCHMARK_COLOR, total: b.total_return, dd: b.maximum_drawdown });
  return {
    id: "comparison",
    title: "Total return and maximum drawdown",
    description: "Whole-period total return and the deepest drawdown for each strategy, on identical dates.",
    traces: [
      {
        type: "bar",
        name: "Total return",
        x: rows.map((r) => r.name),
        y: rows.map((r) => r.total * 100),
        marker: { color: "#5aa9ff" },
        hovertemplate: "%{y:.1f}%<extra>Total return</extra>",
      },
      {
        type: "bar",
        name: "Maximum drawdown",
        x: rows.map((r) => r.name),
        y: rows.map((r) => r.dd * 100),
        marker: { color: "#ff6b6b" },
        hovertemplate: "%{y:.1f}%<extra>Maximum drawdown</extra>",
      },
    ],
    layout: baseLayout("Percent", { barmode: "group", hovermode: "closest" }),
    table: {
      columns: ["Series", "Total return (%)", "Maximum drawdown (%)"],
      rows: rows.map((r) => [r.name, Number((r.total * 100).toFixed(2)), Number((r.dd * 100).toFixed(2))]),
    },
  };
}

// Value divided by the pre-drawdown peak, counted in trading days from each trough.
export function recoveryChart(data: BacktestResultData, run: BacktestRun): ChartSpec | null {
  const lines: Line[] = [];
  const consider = (
    name: string,
    color: string,
    dashed: boolean,
    dates: string[],
    values: number[],
    recovery: { status: string; trough_date?: string; peak_value?: number } | undefined,
  ) => {
    if (!recovery || recovery.status === "no_drawdown" || !recovery.trough_date || !recovery.peak_value) return;
    const start = dates.indexOf(recovery.trough_date);
    if (start < 0 || start >= dates.length - 1) return; // no observations after the trough to plot
    const peak = recovery.peak_value;
    const y = values.slice(start).map((v) => (v / peak) * 100);
    lines.push({ name, color, dashed, dates: y.map((_, i) => String(i)), y });
  };
  for (const [key, s] of Object.entries(data.series)) {
    const m = run.summary?.strategies[key];
    consider(m?.label ?? key, colorFor(key), false, s.dates, s.portfolio_value, m?.recovery);
  }
  if (data.benchmark && run.summary?.benchmark) {
    const b = run.summary.benchmark;
    consider(b.label, BENCHMARK_COLOR, true, data.benchmark.dates, data.benchmark.portfolio_value, b.recovery);
  }
  if (lines.length === 0) return null;
  const longest = Math.max(...lines.map((l) => l.dates.length));
  return {
    id: "recovery",
    title: "Recovery trajectory",
    description:
      "From each series' maximum-drawdown trough: value as a percent of the peak that preceded the trough, by trading days since the trough. 100% means the prior peak is regained. The line ends at the end of the tested window.",
    traces: [
      ...lines.map((l) => ({
        type: "scatter",
        mode: "lines",
        name: l.name,
        x: l.dates.map(Number),
        y: l.y,
        line: { color: l.color, width: 2, dash: l.dashed ? "dash" : "solid" },
        hovertemplate: "%{y:.1f}% of prior peak<extra>%{fullData.name}</extra>",
      })),
      {
        type: "scatter",
        mode: "lines",
        name: "Prior peak (100%)",
        x: [0, longest - 1],
        y: [100, 100],
        line: { color: "#e6eaf2", width: 1, dash: "dot" },
        hoverinfo: "skip",
      },
    ],
    layout: baseLayout("Value as % of prior peak", {
      legend: { orientation: "h", yanchor: "top", y: -0.3 },
      xaxis: { gridcolor: "#27314a", title: { text: "Trading days since trough" }, automargin: true },
    }),
    table: {
      columns: ["Trading days since trough", ...lines.map((l) => l.name)],
      rows: Array.from({ length: longest }, (_, i) => i)
        .filter((i) => i % 21 === 0 || i === longest - 1)
        .map((i) => [i, ...lines.map((l) => (i < l.y.length ? Number(l.y[i].toFixed(2)) : null))]),
    },
  };
}
