import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CoverageCheck } from "@/components/CoverageCheck";
import { DataStatusView } from "@/components/DataStatusView";
import type { CoverageReport, DataStatus } from "@/lib/api";

interface DomMatchers {
  toBeInTheDocument(): void;
  toHaveTextContent(text: string): void;
  toBeDisabled(): void;
}
declare module "vitest" {
  interface Assertion extends DomMatchers {}
  interface AsymmetricMatchersContaining extends DomMatchers {}
}

expect.extend({
  toBeInTheDocument: (el: HTMLElement | null) => ({
    pass: !!el && el.isConnected,
    message: () => "expected element to be in the document",
  }),
  toHaveTextContent: (el: HTMLElement, text: string) => ({
    pass: (el.textContent ?? "").includes(text),
    message: () => `expected "${el.textContent}" to contain "${text}"`,
  }),
  toBeDisabled: (el: HTMLButtonElement) => ({
    pass: el.disabled,
    message: () => "expected element to be disabled",
  }),
});

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

const report: CoverageReport = {
  symbols: [
    {
      symbol: "XLK", status: "ok", source: "downloaded", first_date: "1998-12-22", last_date: "2026-10-27",
      rows: 7005, retrieved_at: "2026-10-28T00:00:00Z", gaps: [], message: null,
    },
    {
      symbol: "NOPE", status: "unavailable", source: null, first_date: null, last_date: null,
      rows: 0, retrieved_at: null, gaps: [], message: "yfinance returned no price history for NOPE.",
    },
  ],
  requested_start: null, requested_end: null, common_start: "1998-12-22", common_end: "2026-10-27",
  common_rows: 7005, limiting_symbol: "XLK", ready: false, issues: ["NOPE: no price history."],
};

const status: DataStatus = {
  symbol_index: { available: true, count: 11433, retrieved_at: "2026-10-01", sources: ["nasdaqlisted.txt"] },
  provider: { name: "yfinance", price_field: "Adj Close", attribution: "Yahoo Finance via yfinance", real_time: false },
  cached_symbols: [
    { symbol: "XLK", provider: "yfinance", price_field: "Adj Close", retrieved_at: "2026-10-28T10:00:00Z",
      first_date: "1998-12-22", last_date: "2026-10-27", rows: 7005 },
  ],
  cached_range: ["1998-12-22", "2026-10-27"],
  failures: [{ symbol: "NOPE", kind: "no_data", message: "no price history", at: "2026-10-28T10:00:00Z" }],
  disclaimer: "Data may be delayed.",
};

describe("CoverageCheck", () => {
  it("is disabled with no holdings", () => {
    render(<CoverageCheck symbols={[]} />);
    expect(screen.getByRole("button", { name: "Check data coverage" })).toBeDisabled();
  });

  it("separates available from unavailable symbols and shows not-ready", async () => {
    const fn = mockFetch(() => ({ body: report }));
    render(<CoverageCheck symbols={["XLK", "NOPE"]} />);
    fireEvent.click(screen.getByRole("button", { name: "Check data coverage" }));
    expect(await screen.findByText("Available")).toBeInTheDocument();
    expect(screen.getByText("No price history")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Not ready");
    expect(screen.getByText("NOPE: no price history.")).toBeInTheDocument();
    expect(JSON.parse(fn.mock.calls[0][1]!.body as string)).toEqual({ symbols: ["XLK", "NOPE"] });
  });

  it("shows the common range when ready", async () => {
    mockFetch(() => ({ body: { ...report, symbols: [report.symbols[0]], ready: true, issues: [] } }));
    render(<CoverageCheck symbols={["XLK"]} />);
    fireEvent.click(screen.getByRole("button", { name: "Check data coverage" }));
    expect(await screen.findByRole("status")).toHaveTextContent("1998-12-22 to 2026-10-27 (7005 common");
  });

  it("reports an unreachable backend", async () => {
    mockFetch(() => "network");
    render(<CoverageCheck symbols={["XLK"]} />);
    fireEvent.click(screen.getByRole("button", { name: "Check data coverage" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the PortfolioLab backend");
  });
});

describe("DataStatusView", () => {
  it("renders provider, cache and failures", async () => {
    mockFetch(() => ({ body: status }));
    render(<DataStatusView />);
    expect(screen.getByText(/Loading data status/)).toBeInTheDocument();
    expect(await screen.findByText(/yfinance \(Adj Close, not real-time\)/)).toBeInTheDocument();
    expect(screen.getByText("1998-12-22 to 2026-10-27")).toBeInTheDocument();
    expect(screen.getByText(/11,433 symbols/)).toBeInTheDocument();
    expect(screen.getByText(/no price history/)).toBeInTheDocument();
  });

  it("shows an error when the backend is down", async () => {
    mockFetch(() => "network");
    render(<DataStatusView />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load data status");
  });

  it("does not clear the cache unless confirmed", async () => {
    const fn = mockFetch(() => ({ body: status }));
    vi.stubGlobal("confirm", vi.fn(() => false));
    window.confirm = vi.fn(() => false);
    render(<DataStatusView />);
    fireEvent.click(await screen.findByRole("button", { name: "Clear cache" }));
    expect(fn.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false);
  });

  it("clears the cache after confirmation and reloads", async () => {
    let cleared = false;
    const fn = mockFetch((url, init) => {
      if (init?.method === "DELETE") {
        cleared = true;
        return { body: { cleared_symbols: 1 } };
      }
      return { body: cleared ? { ...status, cached_symbols: [], cached_range: null } : status };
    });
    window.confirm = vi.fn(() => true);
    render(<DataStatusView />);
    fireEvent.click(await screen.findByRole("button", { name: "Clear cache" }));
    expect(await screen.findByText("Cleared 1 cached symbol(s).")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Nothing cached yet")).toBeInTheDocument());
    expect(String(fn.mock.calls.find(([, i]) => i?.method === "DELETE")![0])).toContain("confirm=true");
  });

  it("re-downloads a single symbol", async () => {
    const fn = mockFetch((url) =>
      url.endsWith("/data/refresh") ? { body: { ...report, symbols: [report.symbols[0]] } } : { body: status },
    );
    render(<DataStatusView />);
    fireEvent.click(await screen.findByRole("button", { name: "Re-download XLK" }));
    expect(await screen.findByText(/Re-downloaded XLK: 7005 rows/)).toBeInTheDocument();
    expect(fn.mock.calls.some(([u]) => String(u).endsWith("/data/refresh"))).toBe(true);
  });
});
