import type { Portfolio, PortfolioPayload } from "@/lib/api";

export type HoldingForm = { symbol: string; name: string; weightPct: string };
export type PortfolioForm = {
  name: string;
  capital: string;
  maxWeightPct: string;
  mode: "equal" | "custom";
  holdings: HoldingForm[];
};

export const DEFAULT_FORM: PortfolioForm = {
  name: "",
  capital: "100000",
  maxWeightPct: "30",
  mode: "equal",
  holdings: [],
};

/** Strict numeric parse: rejects blanks and trailing junk that parseFloat would accept. */
export function parseNumber(text: string): number | null {
  const trimmed = text.trim().replace(/,/g, "");
  if (trimmed === "" || !/^[+-]?(\d+\.?\d*|\.\d+)$/.test(trimmed)) return null;
  return Number(trimmed);
}

export type BuildResult = { payload: PortfolioPayload } | { error: string };

export function buildPayload(form: PortfolioForm, nameFallback = "Untitled portfolio"): BuildResult {
  const capital = parseNumber(form.capital);
  if (capital === null) return { error: "Initial capital must be a number." };
  const cap = parseNumber(form.maxWeightPct);
  if (cap === null) return { error: "Maximum weight must be a number (percent)." };
  return {
    payload: {
      name: form.name.trim() || nameFallback,
      initial_capital: capital,
      max_weight: cap / 100,
      weight_mode: form.mode,
      holdings: form.holdings.map((h) => {
        if (form.mode === "equal") return { symbol: h.symbol, weight: null };
        const pct = parseNumber(h.weightPct);
        return { symbol: h.symbol, weight: pct === null ? null : pct / 100 };
      }),
    },
  };
}

export function formFromPortfolio(p: Portfolio): PortfolioForm {
  return {
    name: p.name,
    capital: String(p.initial_capital),
    maxWeightPct: String(+(p.max_weight * 100).toFixed(6)),
    mode: p.weight_mode,
    holdings: p.holdings.map((h) => ({
      symbol: h.symbol,
      name: "",
      weightPct: h.weight === null ? "" : String(+(h.weight * 100).toFixed(6)),
    })),
  };
}
