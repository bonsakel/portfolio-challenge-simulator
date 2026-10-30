import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StrategyLab } from "@/components/StrategyLab";
import type { ModelInfo, Portfolio, TrainingJob } from "@/lib/api";
import { buildRequest, defaultForm } from "@/lib/backtest";
import { buildTrainingRequest, defaultTrainingForm, estimateSeconds, formatDuration } from "@/lib/training";

afterEach(() => vi.unstubAllGlobals());

const nine = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"];
const portfolio: Portfolio = {
  id: 3,
  name: "Nine ETFs",
  initial_capital: 100000,
  max_weight: 0.3,
  weight_mode: "equal",
  holdings: nine.map((symbol) => ({ symbol, weight: null })),
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  validation: { valid: true, issues: [], effective_weights: {}, weight_sum: 1, min_feasible_holdings: 4 },
};
const model: ModelInfo = {
  id: "research/covid_crash/ppo",
  algorithm: "PPO",
  label: "PPO - covid crash (research)",
  source: "research",
  universe: nine,
  max_weight: 0.3,
  transaction_cost_rate: 0.001,
  observation_size: 191,
  action_size: 9,
  training_period: ["2016-02-19", "2019-02-18"],
  validation_period: ["2019-02-19", "2020-02-18"],
  seed: 42,
  trained_on: "covid_crash",
  timesteps: 100000,
  package_versions: {},
  compatibility: null,
};
const job: TrainingJob = {
  id: 5,
  algorithm: "PPO",
  status: "running",
  progress: 0.25,
  estimated_seconds: 67,
  error: null,
  model_id: null,
  config: {
    total_timesteps: 100000,
    training_start: "2016-02-19",
    training_end: "2019-02-18",
    validation_start: "2019-02-19",
    validation_end: "2020-02-18",
  },
  created_at: "2026-01-01T00:00:00Z",
  started_at: null,
  finished_at: null,
};

describe("training form logic", () => {
  const ok = { ...defaultTrainingForm(), portfolioId: 3 };

  it("builds a request only with an ordered, bounded, complete form", () => {
    const built = buildTrainingRequest(ok, false);
    expect("request" in built && built.request.total_timesteps).toBe(100000);
    expect("request" in built && built.request.confirm).toBe(false);
    expect("request" in built && built.request.transaction_cost_rate).toBeCloseTo(0.001);
    expect(buildTrainingRequest({ ...ok, portfolioId: null }, false)).toHaveProperty("error");
    expect(buildTrainingRequest({ ...ok, valStart: "2018-01-01" }, false)).toHaveProperty("error");
    expect(buildTrainingRequest({ ...ok, timesteps: "100" }, false)).toHaveProperty("error");
    expect(buildTrainingRequest({ ...ok, timesteps: "50000.5" }, false)).toHaveProperty("error");
    expect(buildTrainingRequest({ ...ok, seed: "-1" }, false)).toHaveProperty("error");
  });

  it("estimates and formats durations", () => {
    expect(estimateSeconds("PPO", 100000)).toBe(67);
    expect(estimateSeconds("DDPG", 100000)).toBe(333);
    expect(formatDuration(20)).toBe("20 s");
    expect(formatDuration(333)).toBe("6 min");
    expect(formatDuration(4000)).toBe("1 h 7 min");
  });
});

describe("backtest request with RL strategies", () => {
  it("requires a chosen model for PPO and sends it explicitly", () => {
    const form = { ...defaultForm(), portfolioId: 3, strategies: ["equal_weight" as const, "ppo" as const] };
    expect(buildRequest(form)).toHaveProperty("error");
    const built = buildRequest({ ...form, rlModels: { ppo: "research/covid_crash/ppo" } });
    expect("request" in built && built.request.rl_models).toEqual({ ppo: "research/covid_crash/ppo" });
  });

  it("drops models for strategies that are not selected", () => {
    const form = { ...defaultForm(), portfolioId: 3, rlModels: { ppo: "research/covid_crash/ppo" } };
    const built = buildRequest(form);
    expect("request" in built && built.request.rl_models).toEqual({});
  });
});

describe("StrategyLab", () => {
  function setup(jobs: TrainingJob[] = []) {
    const calls: { url: string; method: string }[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        const method = init?.method ?? "GET";
        calls.push({ url, method });
        let body: unknown = [];
        if (url.endsWith("/models")) body = { models: [model], vix_range: ["2003-03-03", "2024-10-11"], vix_source: "vix.csv" };
        else if (url.includes("/portfolios")) body = [portfolio];
        else if (url.includes("/training-jobs") && method === "POST") body = { ...job, status: "queued", progress: 0 };
        else if (url.includes("/training-jobs")) body = jobs;
        return { ok: true, status: 200, statusText: "ok", json: async () => body } as Response;
      }),
    );
    return calls;
  }

  it("explains strategies and lists models with their recorded universe and periods", async () => {
    setup();
    render(<StrategyLab />);
    expect(await screen.findByText("PPO - covid crash (research)")).toBeTruthy();
    expect(screen.getByText(/2016-02-19 to 2019-02-18/)).toBeTruthy();
    expect(screen.getByText(/XLB, XLE, XLF/)).toBeTruthy();
    expect(screen.getByText(/Mean Variance \(max Sharpe\)/)).toBeTruthy();
    expect(screen.getByText(/2003-03-03 to 2024-10-11/)).toBeTruthy();
  });

  it("never starts training without an explicit confirmation", async () => {
    const calls = setup();
    render(<StrategyLab />);
    await screen.findByText("PPO - covid crash (research)");
    await screen.findByRole("option", { name: "Nine ETFs" });
    fireEvent.change(screen.getByLabelText("Portfolio (9 assets)"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: /Review training run/ }));
    expect(await screen.findByRole("alertdialog")).toBeTruthy();
    expect(calls.some((c) => c.method === "POST")).toBe(false);

    const start = screen.getByRole("button", { name: "Start training" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    fireEvent.click(screen.getByLabelText(/I understand/));
    expect(start.disabled).toBe(false);
    fireEvent.click(start);
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url.endsWith("/training-jobs"))).toBe(true));
  });

  it("editing the form after review withdraws the confirmation prompt", async () => {
    setup();
    render(<StrategyLab />);
    await screen.findByRole("option", { name: "Nine ETFs" });
    fireEvent.change(screen.getByLabelText("Portfolio (9 assets)"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: /Review training run/ }));
    await screen.findByRole("alertdialog");
    fireEvent.change(screen.getByLabelText("Training steps"), { target: { value: "20000" } });
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("shows progress and a cancel button for the active job, and disables a second start", async () => {
    const calls = setup([job]);
    render(<StrategyLab pollMs={10_000} />);
    expect(await screen.findByText(/25% of steps validated/)).toBeTruthy();
    expect((screen.getByRole("button", { name: /Review training run/ }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Cancel training" }));
    await waitFor(() => expect(calls.some((c) => c.url.endsWith("/training-jobs/5/cancel") && c.method === "POST")).toBe(true));
  });
});
