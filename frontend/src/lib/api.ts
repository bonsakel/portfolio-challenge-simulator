const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

export async function apiSend<T>(
  method: "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      signal,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Cannot reach the PortfolioLab backend. Is it running on port 8000?");
  }
  if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
  return (response.status === 204 ? undefined : await response.json()) as T;
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const detail = (await response.json()).detail;
    if (Array.isArray(detail)) return detail.map((d: { msg: string }) => d.msg).join("; ");
    return String(detail ?? response.statusText);
  } catch {
    return response.statusText;
  }
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { signal, cache: "no-store" });
  } catch {
    throw new ApiError("Cannot reach the PortfolioLab backend. Is it running on port 8000?");
  }
  if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
  return (await response.json()) as T;
}

export type Asset = { symbol: string; name: string; exchange: string; asset_type: string };
export type SearchResponse = { query: string; results: Asset[]; source: string };

export type Issue = {
  severity: "error" | "warning";
  code: string;
  message: string;
  symbol: string | null;
};
export type ValidationReport = {
  valid: boolean;
  issues: Issue[];
  effective_weights: Record<string, number>;
  weight_sum: number | null;
  min_feasible_holdings: number | null;
};
export type PortfolioPayload = {
  name: string;
  initial_capital: number;
  max_weight: number;
  weight_mode: "equal" | "custom";
  holdings: { symbol: string; weight: number | null }[];
};
export type Portfolio = PortfolioPayload & {
  id: number;
  created_at: string;
  updated_at: string;
  validation: ValidationReport;
};

export type Health = { status: string; version: string; server_time_utc: string };

export type DownturnRow = {
  event: string;
  strategy: string;
  period_start: string;
  period_end: string;
  trading_days: number;
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  maximum_drawdown: number;
  transaction_costs: number;
};

export type ResearchSummary = { source: string; rows: DownturnRow[] };

export type ResearchRecoveryRow = {
  event: string;
  strategy: string;
  trough_date: string;
  trough_value: number;
  recovery_return_from_trough: number;
  recovered_within_24_months: boolean;
  recovery_time_trading_days: number | null;
  recovery_date: string | null;
  ending_value: number;
};
export type ResearchOverview = {
  source: string;
  locked: boolean;
  settings: {
    assets: string[];
    initial_capital: number;
    max_weight: number;
    transaction_cost_rate: number;
    long_only: boolean;
    fully_invested: boolean;
    price_field: string;
    volatility_source: string;
    risk_free_rate_daily: number;
    annualization_trading_days: number;
    initial_weights: string;
    equal_weight: { rebalance: string };
    mean_variance: { lookback_trading_days: number; rebalance: string; turnover_threshold: number };
    minimum_variance: { lookback_trading_days: number; rebalance: string; turnover_threshold: number };
    observation_size: number;
    random_seed: number;
  };
  events: Record<string, Record<string, [string, string]>>;
  downturn: DownturnRow[];
  recovery: ResearchRecoveryRow[];
  findings: { text: string; evidence: string }[];
  caveats: string[];
};
export type ResearchVerification = {
  event: string;
  period_start: string;
  period_end: string;
  all_match: boolean;
  tolerance: number;
  checks: { strategy: string; metric: string; stored: number; recomputed: number; matches: boolean }[];
};

export type SymbolCoverage = {
  symbol: string;
  status: "ok" | "unavailable" | "provider_error";
  source: "cache" | "downloaded" | null;
  first_date: string | null;
  last_date: string | null;
  rows: number;
  retrieved_at: string | null;
  gaps: { after: string; before: string; calendar_days: number }[];
  message: string | null;
};
export type CoverageReport = {
  symbols: SymbolCoverage[];
  requested_start: string | null;
  requested_end: string | null;
  common_start: string | null;
  common_end: string | null;
  common_rows: number;
  limiting_symbol: string | null;
  ready: boolean;
  issues: string[];
};
export type DataStatus = {
  symbol_index: { available: boolean; count: number; retrieved_at: string | null; sources: string[] };
  provider: { name: string; price_field: string; attribution: string; real_time: boolean };
  cached_symbols: {
    symbol: string;
    provider: string;
    price_field: string;
    retrieved_at: string;
    first_date: string;
    last_date: string;
    rows: number;
  }[];
  cached_range: [string, string] | null;
  failures: { symbol: string; kind: string; message: string; at: string }[];
  disclaimer: string;
};

export type StrategyKey = "equal_weight" | "mean_variance" | "minimum_variance" | "ppo" | "ddpg";
export type RecoveryAnalysis = {
  status: "recovered" | "not_recovered" | "no_drawdown";
  max_drawdown: number;
  peak_date?: string;
  peak_value?: number;
  trough_date?: string;
  trough_value?: number;
  trading_days_peak_to_trough?: number;
  recovery_return_from_trough?: number;
  recovery_date?: string | null;
  recovery_trading_days?: number | null;
};
export type DownsideMetrics = {
  sortino_ratio: number | null;
  var_95_daily: number | null;
  cvar_95_daily: number | null;
  worst_day: number | null;
};
export type ModelInfo = {
  id: string;
  algorithm: "PPO" | "DDPG";
  label: string;
  source: "research" | "user";
  universe: string[];
  max_weight: number;
  transaction_cost_rate: number;
  observation_size: number;
  action_size: number;
  training_period: [string, string];
  validation_period: [string, string];
  seed: number | null;
  trained_on: string;
  timesteps: number | null;
  package_versions: Record<string, string>;
  compatibility: { compatible: boolean; issues: string[]; warnings: string[] } | null;
};
export type ModelsResponse = { models: ModelInfo[]; vix_range: [string, string] | null; vix_source: string };
export type TrainingJob = {
  id: number;
  algorithm: "PPO" | "DDPG";
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  estimated_seconds: number;
  error: string | null;
  model_id: string | null;
  config: Record<string, unknown> & { total_timesteps: number; training_start: string; training_end: string; validation_start: string; validation_end: string };
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};
export type TrainingRequest = {
  portfolio_id: number;
  algorithm: "PPO" | "DDPG";
  training_start: string;
  training_end: string;
  validation_start: string;
  validation_end: string;
  total_timesteps: number;
  transaction_cost_rate: number;
  seed: number;
  confirm: boolean;
};
export type StrategySummary = {
  label: string;
  model?: (Omit<ModelInfo, "compatibility"> & { warnings: string[] }) | null;
  rebalance: "daily" | "monthly";
  turnover_threshold: number;
  period_start: string;
  period_end: string;
  trading_days: number;
  initial_value: number;
  ending_value: number;
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number | null;
  maximum_drawdown: number;
  transaction_costs: number;
  total_turnover: number;
  rebalance_count: number;
  downside: DownsideMetrics;
  recovery: RecoveryAnalysis;
};
export type BenchmarkSummary = {
  symbol: string;
  source: string | null;
  label: string;
  trading_days: number;
  initial_value: number;
  ending_value: number;
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number | null;
  maximum_drawdown: number;
  downside: DownsideMetrics;
  recovery: RecoveryAnalysis;
};
export type BacktestSeries = {
  dates: string[];
  portfolio_value: number[];
  net_return: number[];
  drawdown: number[];
  turnover: number[];
  transaction_cost: number[];
  weights: Record<string, number[]>;
};
export type BacktestResultData = {
  series: Record<string, BacktestSeries>;
  benchmark: { dates: string[]; portfolio_value: number[]; net_return: number[]; drawdown: number[] } | null;
};
export type BacktestSummary = {
  data: {
    symbols: { symbol: string; source: string | null; first_date: string | null; last_date: string | null; rows: number }[];
    common_start: string;
    common_end: string;
    common_price_rows: number;
    dates_dropped_per_symbol: Record<string, number>;
    notes: string[];
    price_field: string;
    provider: string;
  };
  period: {
    requested_start: string;
    requested_end: string;
    first_return_date: string;
    last_return_date: string;
    trading_days: number;
  };
  disclaimer: string;
  base_date: string;
  benchmark: BenchmarkSummary | null;
  strategies: Record<string, StrategySummary>;
};
export type BacktestRun = {
  id: number;
  status: "queued" | "running" | "succeeded" | "failed";
  progress: number;
  error: string | null;
  portfolio_id: number | null;
  portfolio_name: string;
  created_at: string;
  finished_at: string | null;
  config: Record<string, unknown> & { start: string; end: string; strategies: StrategyKey[] };
  summary: BacktestSummary | null;
};
export type BacktestRequest = {
  portfolio_id: number;
  strategies: StrategyKey[];
  rl_models: Partial<Record<"ppo" | "ddpg", string>>;
  start: string;
  end: string;
  lookback_days: number;
  transaction_cost_rate: number;
  rebalance: "daily" | "monthly" | null;
  turnover_threshold: number | null;
  risk_free_rate_annual: number;
  initial_allocation: "equal" | "portfolio_weights";
  benchmark: string | null;
};
export const API_BASE = BASE;
