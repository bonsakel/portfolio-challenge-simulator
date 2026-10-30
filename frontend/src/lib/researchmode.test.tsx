import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ResearchMode, strategyName } from "@/components/ResearchMode";
import type { ResearchOverview } from "@/lib/api";

afterEach(() => vi.unstubAllGlobals());

const row = (strategy: string, total_return: number) => ({
  event: "covid_crash",
  strategy,
  period_start: "2020-02-19",
  period_end: "2020-03-23",
  trading_days: 24,
  total_return,
  annualized_return: -0.9,
  annualized_volatility: 0.8,
  sharpe_ratio: -4.1,
  maximum_drawdown: -0.3,
  transaction_costs: 120,
});
const overview: ResearchOverview = {
  source: "summary_metrics.csv",
  locked: true,
  settings: {
    assets: ["XLB", "XLE"],
    initial_capital: 100000,
    max_weight: 0.3,
    transaction_cost_rate: 0.001,
    long_only: true,
    fully_invested: true,
    price_field: "adjusted_close",
    volatility_source: "CBOE",
    risk_free_rate_daily: 0,
    annualization_trading_days: 252,
    initial_weights: "equal_weight",
    equal_weight: { rebalance: "monthly" },
    mean_variance: { lookback_trading_days: 252, rebalance: "daily", turnover_threshold: 0.02 },
    minimum_variance: { lookback_trading_days: 252, rebalance: "daily", turnover_threshold: 0.02 },
    observation_size: 191,
    random_seed: 42,
  },
  events: {
    covid_crash: {
      training: ["2016-02-19", "2019-02-18"],
      validation: ["2019-02-19", "2020-02-18"],
      downturn_test: ["2020-02-19", "2020-03-23"],
      recovery_observation: ["2020-03-24", "2022-03-23"],
    },
  },
  downturn: [row("minimum_variance", -0.297), row("ppo", -0.4)],
  recovery: [
    {
      event: "covid_crash",
      strategy: "ppo",
      trough_date: "2020-03-23",
      trough_value: 62338,
      recovery_return_from_trough: 0.5,
      recovered_within_24_months: false,
      recovery_time_trading_days: null,
      recovery_date: null,
      ending_value: 93707,
    },
  ],
  findings: [{ text: "Every strategy lost money in every downturn test.", evidence: "2 of 2 downturn total returns are negative" }],
  caveats: ["One random seed per RL model."],
};

function stub(handler: (url: string, init?: RequestInit) => unknown, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => ({
      ok: status < 400,
      status,
      statusText: "x",
      json: async () => handler(url, init),
    })),
  );
}

describe("Research Mode", () => {
  it("names strategies with correct casing", () => {
    expect(strategyName("ppo")).toBe("PPO");
    expect(strategyName("minimum_variance")).toBe("Minimum Variance");
  });

  it("shows the locked configuration, computed findings, and the unrecovered message", async () => {
    stub(() => overview);
    render(<ResearchMode />);
    await waitFor(() => expect(screen.getByText("Locked configuration")).toBeTruthy());
    expect(screen.getByText(/Every strategy lost money/)).toBeTruthy();
    expect(screen.getByText("Not recovered within 24 months")).toBeTruthy();
    expect(screen.getByText(/0.10% of traded value/)).toBeTruthy();
    expect(screen.getByText(/30% maximum per asset/)).toBeTruthy();
    expect(screen.getByText("−29.7%")).toBeTruthy();
  });

  it("recomputes an event on demand and reports a mismatch", async () => {
    stub((url, init) =>
      init?.method === "POST"
        ? {
            event: "covid_crash",
            period_start: "a",
            period_end: "b",
            all_match: false,
            tolerance: 1e-6,
            checks: [{ strategy: "ppo", metric: "total_return", stored: 1, recomputed: 2, matches: false }],
          }
        : overview,
    );
    render(<ResearchMode />);
    fireEvent.click(await screen.findByText("Recompute Covid Crash"));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("PPO total_return"));
  });

  it("shows a clear error when results are missing", async () => {
    stub(() => ({ detail: "Research results not found." }), 404);
    render(<ResearchMode />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Research results not found"));
  });
});
