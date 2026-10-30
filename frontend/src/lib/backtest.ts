import type { BacktestRequest, StrategyKey } from "@/lib/api";
import { parseNumber } from "@/lib/portfolio";

export const RL_KEYS = ["ppo", "ddpg"] as const;
export const OPTIMIZER_KEYS: StrategyKey[] = ["mean_variance", "minimum_variance"];

export const STRATEGIES: { key: StrategyKey; name: string; blurb: string }[] = [
  {
    key: "equal_weight",
    name: "Equal Weight",
    blurb: "Targets 1/N in every asset. Research default: rebalance monthly, no trade threshold.",
  },
  {
    key: "mean_variance",
    name: "Mean Variance (max Sharpe)",
    blurb:
      "Each decision uses only the preceding lookback window: historical mean returns and sample covariance, maximizing the Sharpe ratio under long-only and per-asset caps. Research default: daily, 2-pt trade threshold.",
  },
  {
    key: "minimum_variance",
    name: "Minimum Variance",
    blurb:
      "Minimizes estimated variance from the sample covariance of the preceding lookback window, long-only with the same caps. Research default: daily, 2-pt trade threshold.",
  },
  {
    key: "ppo",
    name: "PPO (reinforcement learning)",
    blurb:
      "A pretrained neural-network policy that sees 20 days of returns, current weights and the VIX, and chooses weights every day. It is frozen: no learning happens during the backtest. It runs only with a compatible model and only on dates after that model's training and validation data.",
  },
  {
    key: "ddpg",
    name: "DDPG (reinforcement learning)",
    blurb:
      "Same environment, inputs, constraints, costs and evaluation as PPO, with a different learning algorithm (deterministic actor-critic). Frozen model, daily decisions, compatible model required.",
  },
];

export type BacktestForm = {
  portfolioId: number | null;
  strategies: StrategyKey[];
  start: string;
  end: string;
  lookback: string;
  costPct: string;
  rebalance: "" | "daily" | "monthly";
  thresholdPct: string;
  riskFreePct: string;
  allocation: "equal" | "portfolio_weights";
  benchmark: string;
  rlModels: Partial<Record<"ppo" | "ddpg", string>>;
};

export const isoDate = (d: Date) => d.toISOString().slice(0, 10);

export function defaultForm(today = new Date()): BacktestForm {
  const start = new Date(today);
  start.setFullYear(start.getFullYear() - 5);
  return {
    portfolioId: null,
    strategies: ["equal_weight", "minimum_variance"],
    start: isoDate(start),
    end: isoDate(today),
    lookback: "252",
    costPct: "0.10",
    rebalance: "",
    thresholdPct: "",
    riskFreePct: "0",
    allocation: "equal",
    benchmark: "SPY",
    rlModels: {},
  };
}

export function buildRequest(form: BacktestForm): { request: BacktestRequest } | { error: string } {
  if (form.portfolioId === null) return { error: "Choose a saved portfolio." };
  if (form.strategies.length === 0) return { error: "Select at least one strategy." };
  if (!form.start || !form.end) return { error: "Choose a start and end date." };
  if (form.start >= form.end) return { error: "Start date must be before the end date." };

  const lookback = parseNumber(form.lookback);
  if (lookback === null || !Number.isInteger(lookback) || lookback < 20 || lookback > 1260)
    return { error: "Lookback must be a whole number of trading days between 20 and 1260." };
  const cost = parseNumber(form.costPct);
  if (cost === null || cost < 0 || cost >= 5) return { error: "Transaction cost must be between 0% and 5%." };
  const rf = parseNumber(form.riskFreePct);
  if (rf === null || rf < 0 || rf > 25) return { error: "Risk-free rate must be between 0% and 25%." };
  let threshold: number | null = null;
  if (form.thresholdPct.trim() !== "") {
    const t = parseNumber(form.thresholdPct);
    if (t === null || t < 0 || t > 50) return { error: "Trade threshold must be between 0 and 50 percentage points." };
    threshold = t / 100;
  }
  for (const key of RL_KEYS)
    if (form.strategies.includes(key) && !form.rlModels[key])
      return { error: `Choose a compatible trained model for ${key.toUpperCase()}, or train one in the Strategy Lab.` };
  const benchmark = form.benchmark.trim().toUpperCase();
  if (benchmark !== "" && !/^[A-Z][A-Z0-9-]{0,9}$/.test(benchmark))
    return { error: "Benchmark must be a ticker such as SPY, or empty for none." };
  return {
    request: {
      portfolio_id: form.portfolioId,
      strategies: form.strategies,
      rl_models: Object.fromEntries(
        RL_KEYS.filter((k) => form.strategies.includes(k) && form.rlModels[k]).map((k) => [k, form.rlModels[k]]),
      ),
      start: form.start,
      end: form.end,
      lookback_days: lookback,
      transaction_cost_rate: cost / 100,
      rebalance: form.rebalance === "" ? null : form.rebalance,
      turnover_threshold: threshold,
      risk_free_rate_annual: rf / 100,
      initial_allocation: form.allocation,
      benchmark: benchmark === "" ? null : benchmark,
    },
  };
}
