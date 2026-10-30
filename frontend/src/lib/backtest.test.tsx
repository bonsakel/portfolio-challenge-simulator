import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BacktestRunner } from "@/components/BacktestRunner";
import type { BacktestRun, BacktestSummary, Portfolio } from "@/lib/api";
import { buildRequest, defaultForm } from "@/lib/backtest";

afterEach(() => vi.unstubAllGlobals());

type Handler = (url: string, init?: RequestInit) => { status?: number; body?: unknown } | "network";

function mockFetch(handler: Handler) {
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const r = handler(url, init);
    if (r === "network") throw new TypeError("fetch failed");
    const { status = 200, body } = r;
    return { ok: status < 400, status, statusText: "err", json: async () => body } as Response;
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const portfolio: Portfolio = {
  id: 7,
  name: "Sectors",
  initial_capital: 100000,
  max_weight: 0.3,
  weight_mode: "equal",
  holdings: ["XLK", "XLE", "XLF", "XLV"].map((symbol) => ({ symbol, weight: null })),
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  validation: { valid: true, issues: [], effective_weights: {}, weight_sum: 1, min_feasible_holdings: 4 },
};

const metrics = {
  rebalance: "monthly" as const,
  turnover_threshold: 0,
  period_start: "2020-02-03",
  period_end: "2020-12-31",
  trading_days: 230,
  initial_value: 100000,
  ending_value: 112345,
  total_return: 0.12345,
  annualized_return: 0.13,
  annualized_volatility: 0.2,
  sharpe_ratio: null,
  maximum_drawdown: -0.3,
  transaction_costs: 250,
  total_turnover: 1.5,
  rebalance_count: 11,
  downside: { sortino_ratio: 0.8, var_95_daily: -0.02, cvar_95_daily: -0.03, worst_day: -0.06 },
  recovery: {
    status: "recovered" as const,
    max_drawdown: -0.3,
    peak_date: "2020-02-19",
    peak_value: 100000,
    trough_date: "2020-03-23",
    trough_value: 70000,
    trading_days_peak_to_trough: 23,
    recovery_return_from_trough: 0.6,
    recovery_date: "2020-08-12",
    recovery_trading_days: 98,
  },
};

const summary: BacktestSummary = {
  data: {
    symbols: [{ symbol: "XLK", source: "downloaded", first_date: "1998-12-22", last_date: "2026-10-27", rows: 7005 }],
    common_start: "1998-12-22",
    common_end: "2026-10-27",
    common_price_rows: 7005,
    dates_dropped_per_symbol: { XLK: 0 },
    notes: [],
    price_field: "Adj Close",
    provider: "yfinance",
  },
  period: {
    requested_start: "2020-02-01",
    requested_end: "2020-12-31",
    first_return_date: "2020-02-03",
    last_return_date: "2020-12-31",
    trading_days: 230,
  },
  disclaimer: "Historical simulation. Not investment advice.",
  base_date: "2020-01-31",
  benchmark: null,
  strategies: { equal_weight: { label: "Equal Weight", ...metrics } },
};

const resultsFixture = {
  series: {
    equal_weight: {
      dates: ["2020-02-03", "2020-02-04", "2020-03-02"],
      portfolio_value: [100500, 101000, 99000],
      net_return: [0.005, 0.005, -0.02],
      drawdown: [0, 0, -0.0198],
      turnover: [0, 0, 0.1],
      transaction_cost: [0, 0, 10],
      weights: { XLK: [0.5, 0.5, 0.4], XLE: [0.5, 0.5, 0.6] },
    },
  },
  benchmark: null,
};

const run = (patch: Partial<BacktestRun>): BacktestRun => ({
  id: 1,
  status: "queued",
  progress: 0,
  error: null,
  portfolio_id: 7,
  portfolio_name: "Sectors",
  created_at: "2026-10-28T00:00:00Z",
  finished_at: null,
  config: { start: "2020-02-01", end: "2020-12-31", strategies: ["equal_weight"] },
  summary: null,
  ...patch,
});

function baseHandler(extra: Handler): Handler {
  return (url, init) => {
    if (url.endsWith("/portfolios")) return { body: [portfolio] };
    if (url.includes("/research/summary")) return { body: { source: "x", rows: [] } };
    if (url.includes("/backtests?limit")) return { body: [] };
    if (url.endsWith("/results")) return { body: resultsFixture };
    return extra(url, init);
  };
}

async function choosePortfolioAndRun() {
  await screen.findByLabelText("Portfolio");
  fireEvent.change(screen.getByLabelText("Portfolio"), { target: { value: "7" } });
  fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));
}

describe("buildRequest", () => {
  const ok = { ...defaultForm(new Date("2026-10-28T12:00:00Z")), portfolioId: 7 };

  it("converts percent inputs to fractions and keeps defaults null", () => {
    const r = buildRequest(ok);
    expect("request" in r && r.request).toMatchObject({
      portfolio_id: 7,
      start: "2021-10-28",
      end: "2026-10-28",
      lookback_days: 252,
      transaction_cost_rate: 0.001,
      rebalance: null,
      turnover_threshold: null,
      risk_free_rate_annual: 0,
      initial_allocation: "equal",
      benchmark: "SPY",
    });
    const custom = buildRequest({ ...ok, thresholdPct: "2", rebalance: "daily", costPct: "0.25", benchmark: " " });
    expect("request" in custom && custom.request).toMatchObject({
      benchmark: null,
      turnover_threshold: 0.02,
      rebalance: "daily",
      transaction_cost_rate: 0.0025,
    });
  });

  it("rejects unusable input with a specific message", () => {
    expect(buildRequest({ ...ok, portfolioId: null })).toEqual({ error: "Choose a saved portfolio." });
    expect(buildRequest({ ...ok, strategies: [] })).toEqual({ error: "Select at least one strategy." });
    expect(buildRequest({ ...ok, start: "2026-01-01", end: "2025-01-01" })).toEqual({
      error: "Start date must be before the end date.",
    });
    expect("error" in buildRequest({ ...ok, lookback: "10" })).toBe(true);
    expect("error" in buildRequest({ ...ok, benchmark: "not a ticker!" })).toBe(true);
    expect("error" in buildRequest({ ...ok, costPct: "abc" })).toBe(true);
    expect("error" in buildRequest({ ...ok, thresholdPct: "99" })).toBe(true);
  });
});

describe("BacktestRunner", () => {
  it("guides the user when there are no saved portfolios", async () => {
    mockFetch(() => ({ body: [] }));
    render(<BacktestRunner />);
    expect(await screen.findByText(/No saved portfolios yet/)).toBeTruthy();
  });

  it("requires a portfolio before running and posts nothing", async () => {
    const fn = mockFetch(baseHandler(() => ({ body: {} })));
    render(<BacktestRunner />);
    fireEvent.click(await screen.findByRole("button", { name: "Run backtest" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Choose a saved portfolio.");
    expect(fn.mock.calls.some(([, i]) => i?.method === "POST")).toBe(false);
  });

  it("starts a run, polls to completion and shows computed results", async () => {
    let polls = 0;
    const fn = mockFetch(
      baseHandler((url, init) => {
        if (init?.method === "POST") return { status: 202, body: run({}) };
        polls += 1;
        return {
          body:
            polls < 2
              ? run({ status: "running", progress: 0.5 })
              : run({ status: "succeeded", progress: 1, summary }),
        };
      }),
    );
    render(<BacktestRunner pollMs={10} />);
    await choosePortfolioAndRun();
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("50%"));
    await waitFor(() => expect(screen.getByText(/Results: Sectors/)).toBeTruthy());
    expect(screen.getByText("$112,345")).toBeTruthy();
    expect(screen.getByText("+12.3%")).toBeTruthy();
    expect(screen.getByText("n/a")).toBeTruthy(); // undefined Sharpe is shown, not as 0
    expect(screen.getByText(/same 230 trading days/)).toBeTruthy();
    expect(screen.getByText(/Not investment advice/)).toBeTruthy();
    const post = fn.mock.calls.find(([, i]) => i?.method === "POST")!;
    expect(JSON.parse(post[1]!.body as string)).toMatchObject({ portfolio_id: 7, transaction_cost_rate: 0.001 });
    expect(screen.getByRole("link", { name: "Download CSV" }).getAttribute("href")).toContain("/backtests/1/export.csv");
  });

  it("shows the backend's failure reason and offers retry", async () => {
    mockFetch(
      baseHandler((url, init) =>
        init?.method === "POST"
          ? { status: 202, body: run({}) }
          : { body: run({ status: "failed", progress: 1, error: "Price history is unavailable. ZZZZ: no data" }) },
      ),
    );
    render(<BacktestRunner pollMs={10} />);
    await choosePortfolioAndRun();
    await waitFor(() => expect(screen.getByText(/Backtest #1 failed: Price history is unavailable/)).toBeTruthy());
    expect(screen.getByRole("button", { name: "Retry backtest" })).toBeTruthy();
  });

  it("shows validation errors returned when starting", async () => {
    mockFetch(
      baseHandler((url, init) =>
        init?.method === "POST" ? { status: 422, body: { detail: "Portfolio is not valid: too few assets." } } : { body: {} },
      ),
    );
    render(<BacktestRunner pollMs={10} />);
    await choosePortfolioAndRun();
    expect((await screen.findByRole("alert")).textContent).toContain("Portfolio is not valid: too few assets.");
  });
});
