import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RiskAnalyticsView } from "@/components/RiskAnalyticsView";
import type { BacktestResultData, BacktestRun, BacktestSummary, StrategySummary } from "@/lib/api";
import { allocationChart, comparisonChart, costChart, drawdownChart, monthEnd, recoveryChart, riskReturnChart, turnoverChart, valueChart } from "@/lib/charts";

afterEach(() => vi.unstubAllGlobals());

const strategy: StrategySummary = {
  label: "Equal Weight",
  rebalance: "monthly",
  turnover_threshold: 0,
  period_start: "2020-01-02",
  period_end: "2020-03-02",
  trading_days: 4,
  initial_value: 100,
  ending_value: 95,
  total_return: -0.05,
  annualized_return: -0.5,
  annualized_volatility: 0.3,
  sharpe_ratio: -1.2,
  maximum_drawdown: -0.1,
  transaction_costs: 1.5,
  total_turnover: 0.2,
  rebalance_count: 2,
  downside: { sortino_ratio: null, var_95_daily: null, cvar_95_daily: null, worst_day: -0.05 },
  recovery: {
    status: "not_recovered",
    max_drawdown: -0.1,
    peak_date: "2020-01-03",
    peak_value: 110,
    trough_date: "2020-02-03",
    trough_value: 99,
    recovery_return_from_trough: -0.04,
    recovery_date: null,
    recovery_trading_days: null,
  },
};

const summary: BacktestSummary = {
  data: {
    symbols: [],
    common_start: "2019-12-31",
    common_end: "2020-03-02",
    common_price_rows: 5,
    dates_dropped_per_symbol: {},
    notes: [],
    price_field: "Adj Close",
    provider: "yfinance",
  },
  period: { requested_start: "2020-01-02", requested_end: "2020-03-02", first_return_date: "2020-01-02", last_return_date: "2020-03-02", trading_days: 4 },
  disclaimer: "Not investment advice.",
  base_date: "2019-12-31",
  benchmark: {
    symbol: "SPY",
    source: "cache",
    label: "SPY (buy and hold)",
    trading_days: 4,
    initial_value: 100,
    ending_value: 90,
    total_return: -0.1,
    annualized_return: -0.6,
    annualized_volatility: 0.4,
    sharpe_ratio: -1.5,
    maximum_drawdown: -0.2,
    downside: { sortino_ratio: null, var_95_daily: null, cvar_95_daily: null, worst_day: -0.07 },
    recovery: { status: "no_drawdown", max_drawdown: 0 },
  },
  strategies: { equal_weight: strategy },
};

const data: BacktestResultData = {
  series: {
    equal_weight: {
      dates: ["2020-01-02", "2020-01-03", "2020-02-03", "2020-03-02"],
      portfolio_value: [105, 110, 99, 95],
      net_return: [0.05, 0.0476, -0.1, -0.0404],
      drawdown: [0, 0, -0.1, -0.1364],
      turnover: [0, 0.1, 0, 0.1],
      transaction_cost: [0, 0.7, 0, 0.8],
      weights: { AAA: [0.5, 0.4, 0.5, 0.6], BBB: [0.5, 0.6, 0.5, 0.4] },
    },
  },
  benchmark: {
    dates: ["2020-01-02", "2020-01-03", "2020-02-03", "2020-03-02"],
    portfolio_value: [101, 102, 95, 90],
    net_return: [0.01, 0.0099, -0.0686, -0.0526],
    drawdown: [0, 0, -0.0686, -0.1176],
  },
};

const run: BacktestRun = {
  id: 3,
  status: "succeeded",
  progress: 1,
  error: null,
  portfolio_id: 1,
  portfolio_name: "Mix",
  created_at: "2026-01-01T00:00:00Z",
  finished_at: "2026-01-01T00:00:01Z",
  config: { start: "2020-01-02", end: "2020-03-02", strategies: ["equal_weight"] },
  summary,
};

describe("chart specs use the calculation outputs", () => {
  it("value chart carries the strategy and the benchmark with the exact values", () => {
    const spec = valueChart(data, run);
    expect(spec.traces.map((t) => t.name)).toEqual(["Equal Weight", "SPY (buy and hold)"]);
    expect(spec.traces[0].y).toEqual([105, 110, 99, 95]);
    expect(spec.traces[1].y).toEqual([101, 102, 95, 90]);
    expect(spec.traces[0].x).toEqual(data.series.equal_weight.dates);
  });

  it("drawdown chart shows percent values and the table is month-end", () => {
    const spec = drawdownChart(data, run);
    expect((spec.traces[0].y as number[])[2]).toBeCloseTo(-10);
    expect(spec.table.rows.map((r) => r[0])).toEqual(["2020-01-03", "2020-02-03", "2020-03-02"]);
    expect(monthEnd(["2020-01-02", "2020-01-31", "2020-02-03"])).toEqual([1, 2]);
  });

  it("costs accumulate and turnover is summed by calendar month in percent", () => {
    const cost = costChart(data, run).traces[0].y as number[];
    expect(cost[3]).toBeCloseTo(1.5);
    expect(cost[1]).toBeCloseTo(0.7);
    const turnover = turnoverChart(data, run);
    expect(turnover.traces[0].x).toEqual(["2020-01", "2020-02", "2020-03"]);
    expect(turnover.traces[0].y).toEqual([10, 0, 10]);
  });

  it("allocation stacks every asset and the table is in percent", () => {
    const spec = allocationChart(data, "equal_weight", "Equal Weight");
    expect(spec.traces.map((t) => t.name)).toEqual(["AAA", "BBB"]);
    expect(spec.traces[0].y).toEqual([50, 40, 50, 60]);
    expect(spec.table.columns).toEqual(["Month-end date", "AAA", "BBB"]);
  });

  it("risk/return and comparison charts include the benchmark", () => {
    const rr = riskReturnChart(run);
    expect(rr.traces).toHaveLength(2);
    expect(rr.traces[1].x).toEqual([40]);
    expect(rr.traces[1].y).toEqual([-60]);
    const cmp = comparisonChart(run);
    expect(cmp.table.rows).toEqual([
      ["Equal Weight", -5, -10],
      ["SPY (buy and hold)", -10, -20],
    ]);
  });

  it("recovery chart is rebased to the prior peak, starts at the trough and skips series without a drawdown", () => {
    const spec = recoveryChart(data, run)!;
    const strategyTrace = spec.traces[0];
    expect(strategyTrace.x).toEqual([0, 1]);
    expect((strategyTrace.y as number[])[0]).toBeCloseTo((99 / 110) * 100);
    expect((strategyTrace.y as number[])[1]).toBeCloseTo((95 / 110) * 100);
    expect(spec.traces.map((t) => t.name)).toEqual(["Equal Weight", "Prior peak (100%)"]);
    const none = { ...run, summary: { ...summary, strategies: { equal_weight: { ...strategy, recovery: { status: "no_drawdown" as const, max_drawdown: 0 } } }, benchmark: null } };
    expect(recoveryChart({ ...data, benchmark: null }, none)).toBeNull();
    // A trough on the last day leaves nothing to plot after it.
    const atEnd = { ...run, summary: { ...summary, benchmark: null, strategies: { equal_weight: { ...strategy, recovery: { ...strategy.recovery, trough_date: "2020-03-02" } } } } };
    expect(recoveryChart({ ...data, benchmark: null }, atEnd)).toBeNull();
  });
});

describe("RiskAnalyticsView", () => {
  function stub(runs: BacktestRun[]) {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        const body = url.includes("/results") ? data : runs;
        return { ok: true, status: 200, statusText: "ok", json: async () => body } as Response;
      }),
    );
  }

  it("tells the user what to do when there are no completed backtests", async () => {
    stub([]);
    render(<RiskAnalyticsView />);
    expect(await screen.findByText(/No completed backtests yet/)).toBeTruthy();
  });

  it("shows risk, recovery status and n/a for undefined downside metrics", async () => {
    stub([run, { ...run, id: 2, status: "failed" }]);
    render(<RiskAnalyticsView />);
    expect(await screen.findByText("Risk and return metrics")).toBeTruthy();
    expect(screen.getAllByText("n/a").length).toBeGreaterThan(0);
    expect(screen.getByText("Not recovered in window")).toBeTruthy();
    expect(screen.getByText("No drawdown")).toBeTruthy();
    expect(screen.getByText("−5.0%")).toBeTruthy();
    expect(screen.getAllByText("SPY (buy and hold)").length).toBeGreaterThan(0);
    expect(screen.getByRole("option", { name: /#3 Mix/ })).toBeTruthy();
    expect(screen.queryByRole("option", { name: /#2 Mix/ })).toBeNull();
  });

  it("switches a chart to its accessible table view", async () => {
    stub([run]);
    render(<RiskAnalyticsView />);
    const heading = await screen.findByRole("heading", { name: "Risk versus return" });
    const card = heading.closest("section")!;
    fireEvent.click(card.querySelector("button[aria-pressed='false']")!);
    await waitFor(() => expect(card.querySelector("table")).toBeTruthy());
    expect(card.textContent).toContain("Annualized volatility (%)");
  });
});
