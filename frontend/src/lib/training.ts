import type { TrainingRequest } from "@/lib/api";
import { parseNumber } from "@/lib/portfolio";

// Rough CPU speeds measured on the development machine; shown only as an estimate.
const STEPS_PER_SECOND = { PPO: 1500, DDPG: 300 } as const;

export function estimateSeconds(algorithm: "PPO" | "DDPG", timesteps: number): number {
  return Math.round(timesteps / STEPS_PER_SECOND[algorithm]);
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

export type TrainingForm = {
  portfolioId: number | null;
  algorithm: "PPO" | "DDPG";
  trainStart: string;
  trainEnd: string;
  valStart: string;
  valEnd: string;
  timesteps: string;
  costPct: string;
  seed: string;
};

// The research COVID-crash split: three years of training, one year of validation.
export const defaultTrainingForm = (): TrainingForm => ({
  portfolioId: null,
  algorithm: "PPO",
  trainStart: "2016-02-19",
  trainEnd: "2019-02-18",
  valStart: "2019-02-19",
  valEnd: "2020-02-18",
  timesteps: "100000",
  costPct: "0.10",
  seed: "42",
});

export function buildTrainingRequest(
  form: TrainingForm,
  confirm: boolean,
): { request: TrainingRequest } | { error: string } {
  if (form.portfolioId === null) return { error: "Choose a saved portfolio with exactly 9 assets." };
  if (!form.trainStart || !form.trainEnd || !form.valStart || !form.valEnd) return { error: "Fill in all four dates." };
  if (!(form.trainStart < form.trainEnd && form.trainEnd < form.valStart && form.valStart < form.valEnd))
    return { error: "Dates must run in order: training start, training end, validation start, validation end." };
  const steps = parseNumber(form.timesteps);
  if (steps === null || !Number.isInteger(steps) || steps < 5000 || steps > 300000)
    return { error: "Training steps must be a whole number between 5,000 and 300,000." };
  const cost = parseNumber(form.costPct);
  if (cost === null || cost < 0 || cost >= 5) return { error: "Transaction cost must be between 0% and 5%." };
  const seed = parseNumber(form.seed);
  if (seed === null || !Number.isInteger(seed) || seed < 0) return { error: "Seed must be a non-negative whole number." };
  return {
    request: {
      portfolio_id: form.portfolioId,
      algorithm: form.algorithm,
      training_start: form.trainStart,
      training_end: form.trainEnd,
      validation_start: form.valStart,
      validation_end: form.valEnd,
      total_timesteps: steps,
      transaction_cost_rate: cost / 100,
      seed,
      confirm,
    },
  };
}
