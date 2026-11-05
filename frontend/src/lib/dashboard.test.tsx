import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import Plotly from "plotly.js-dist-min";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ResearchDashboard } from "@/components/ResearchDashboard";

vi.mock("plotly.js-dist-min", () => {
  const methods = {
    react: vi.fn().mockResolvedValue(undefined),
    newPlot: vi.fn().mockResolvedValue(undefined),
    downloadImage: vi.fn().mockResolvedValue("data:image/png;base64,"),
    purge: vi.fn(),
  };
  return { ...methods, default: methods };
});

afterEach(() => vi.unstubAllGlobals());

const assets = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"];
const metrics = {
  total_return: { mean: -0.3, min: -0.3, max: -0.3 },
  annualized_return: { mean: -0.9, min: -0.9, max: -0.9 },
  annualized_volatility: { mean: 0.8, min: 0.8, max: 0.8 },
  sharpe_ratio: { mean: -4.1, min: -4.1, max: -4.1 },
  maximum_drawdown: { mean: -0.35, min: -0.35, max: -0.35 },
  transaction_costs: { mean: 120, min: 120, max: 120 },
};
const result = {
  metrics,
  recovery: {
    recovery_return_from_trough: { mean: 0.5, min: 0.5, max: 0.5 },
    recovery_time_trading_days: { mean: null, min: null, max: null },
    ending_value: { mean: 90_000, min: 90_000, max: 90_000 },
    recovered_count: 0,
    sample_count: 1,
  },
  series: {
    dates: ["2020-02-19", "2020-03-23"],
    sample_returns: [[0, -0.3]],
    sample_transaction_costs: [[0, 120]],
    portfolio_value: [100_000, 70_000],
    value_min: [100_000, 70_000],
    value_max: [100_000, 70_000],
    drawdown: [0, -0.3],
    drawdown_min: [0, -0.3],
    drawdown_max: [0, -0.3],
    cumulative_cost: [0, 120],
    turnover: [0, 0.12],
    weights: Object.fromEntries(assets.map((asset) => [asset, [1 / 9, 1 / 9]])),
    seed_count: 1,
  },
};
const dashboard = {
  source: "original_single_seed" as const,
  seed_count: 1,
  reference_capital: 100_000,
  annualization_trading_days: 252,
  risk_free_rate_daily: 0,
  strategies: ["ppo"],
  events: [{
    event: "covid_crash",
    windows: {
      downturn_test: ["2020-02-19", "2020-03-23"],
      recovery_observation: ["2020-03-24", "2022-03-23"],
    },
    strategies: { ppo: result },
  }],
};

describe("ResearchDashboard", () => {
  it("guides setup, tracks progress, then autoplays the selected comparison", async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => ({
      ok: true,
      status: 200,
      statusText: "ok",
      json: async () => String(input).includes("/pipeline")
        ? {
            status: "not_started",
            completed_models: 0,
            completed_simulations: 0,
            total_models: 30,
            robustness_ready: false,
            baseline_available: true,
            steps: [
              { id: "data", label: "Market data", status: "complete", detail: "Aligned" },
              { id: "training", label: "Train models", status: "pending", detail: "5 seeds" },
              { id: "simulation", label: "Test", status: "pending", detail: "Not started" },
              { id: "comparison", label: "Compare", status: "current", detail: "Review results" },
            ],
          }
        : dashboard,
    }) as Response);
    vi.stubGlobal("fetch", fetcher);

    render(<ResearchDashboard />);

    expect(await screen.findByRole("heading", { name: "Set up your comparison" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Which downturns do you want to compare?" })).toBeTruthy();
    expect(screen.getByText("Step 1 of 5 · Market downturns")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Global financial crisis/ }));
    fireEvent.click(screen.getByRole("button", { name: /2022 downturn/ }));

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(screen.getByRole("heading", { name: "Which strategies should we include?" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Equal Weight/ }));
    fireEvent.click(screen.getByRole("button", { name: /Mean Variance/ }));
    fireEvent.click(screen.getByRole("button", { name: /Minimum Variance/ }));
    fireEvent.click(screen.getByRole("button", { name: /DDPG/ }));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(screen.getByRole("heading", { name: "How much would you like to invest?" })).toBeTruthy();
    fireEvent.change(screen.getByRole("spinbutton", { name: "Starting capital" }), {
      target: { value: "200000" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByRole("heading", { name: "Which dates should the charts show?" })).toBeTruthy();
    expect(await screen.findByLabelText("View from COVID-19 crash")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(screen.getByRole("heading", { name: "Ready to compare" })).toBeTruthy();
    expect(screen.getByText("Step 5 of 5 · Review")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show results" }));

    expect(await screen.findByText("Selected-period metrics")).toBeTruthy();
    expect(screen.getByText("Where the money went")).toBeTruthy();
    expect(screen.getByText("XLB")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    expect(screen.getByRole("slider", { name: "Performance date for COVID-19 crash" })).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "Download data" }).length).toBeGreaterThan(1);
    expect(screen.getByText("Baseline results ready")).toBeTruthy();

    const pngButton = screen.getAllByRole("button", { name: "Download PNG" })[0] as HTMLButtonElement;
    await waitFor(() => expect(pngButton.disabled).toBe(false));
    fireEvent.click(pngButton);
    await waitFor(() => expect(Plotly.downloadImage).toHaveBeenCalled());
    const exportLayout = vi.mocked(Plotly.newPlot).mock.calls.at(-1)?.[2];
    expect(exportLayout).toMatchObject({
      title: { text: expect.stringContaining("Portfolio value") },
      legend: { orientation: "h" },
      xaxis: expect.any(Object),
      yaxis: expect.any(Object),
    });
    expect(vi.mocked(Plotly.downloadImage).mock.calls.at(-1)?.[1]).toMatchObject({
      format: "png",
      scale: 4,
      setBackground: "#ffffff",
    });

    expect(await screen.findByText("$240")).toBeTruthy();
    const dateInput = screen.getByLabelText<HTMLInputElement>("View from COVID-19 crash");
    fireEvent.change(dateInput, { target: { value: "2020-03-23" } });
    expect(dateInput.value).toBe("2020-03-23");

    fireEvent.click(screen.getByRole("button", { name: "Compress research pipeline" }));
    expect(screen.getByRole("button", { name: "Show research pipeline" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Change setup" }));
    expect(screen.getByRole("heading", { name: "Ready to compare" })).toBeTruthy();
  });
});
