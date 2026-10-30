export const pct = (value: number, digits = 1) =>
  `${value < 0 ? "−" : value > 0 ? "+" : ""}${Math.abs(value * 100).toFixed(digits)}%`;

export const pctPlain = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

export const money = (value: number) =>
  value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const label = (slug: string) =>
  slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
