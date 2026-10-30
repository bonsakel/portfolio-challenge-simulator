import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AssetSearch } from "@/components/AssetSearch";
import { PortfolioBuilder } from "@/components/PortfolioBuilder";
import type { ValidationReport } from "@/lib/api";
import { buildPayload, DEFAULT_FORM, parseNumber } from "@/lib/portfolio";

afterEach(() => vi.unstubAllGlobals());

type Handler = (url: string, init?: RequestInit) => { status?: number; body?: unknown };

function mockFetch(handler: Handler) {
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const { status = 200, body } = handler(url, init);
    return {
      ok: status < 400,
      status,
      statusText: "err",
      json: async () => body,
    } as Response;
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const AAPL = { symbol: "AAPL", name: "Apple Inc. - Common Stock", exchange: "NASDAQ", asset_type: "Equity" };
const XLK = { symbol: "XLK", name: "Technology Select Sector SPDR", exchange: "NYSE Arca", asset_type: "ETF" };

const okReport: ValidationReport = {
  valid: true,
  issues: [],
  effective_weights: {},
  weight_sum: 1,
  min_feasible_holdings: 4,
};

describe("portfolio helpers", () => {
  it("parses numbers strictly", () => {
    expect(parseNumber("1,000.5")).toBe(1000.5);
    expect(parseNumber("")).toBeNull();
    expect(parseNumber("12abc")).toBeNull();
    expect(parseNumber("1e5")).toBeNull();
  });

  it("converts percent inputs to fractions and drops weights in equal mode", () => {
    const form = {
      ...DEFAULT_FORM,
      holdings: [{ symbol: "AAPL", name: "", weightPct: "60" }],
    };
    const equal = buildPayload(form);
    expect("payload" in equal && equal.payload.holdings[0].weight).toBeNull();
    const custom = buildPayload({ ...form, mode: "custom" });
    expect("payload" in custom && custom.payload.holdings[0].weight).toBeCloseTo(0.6);
    expect("payload" in custom && custom.payload.max_weight).toBeCloseTo(0.3);
  });

  it("reports unusable numeric fields instead of sending them", () => {
    expect(buildPayload({ ...DEFAULT_FORM, capital: "abc" })).toEqual({
      error: "Initial capital must be a number.",
    });
  });
});

describe("AssetSearch", () => {
  it("shows loading then results, and adds an asset", async () => {
    mockFetch(() => ({ body: { query: "a", results: [AAPL, XLK], source: "NASDAQ Trader" } }));
    const onAdd = vi.fn();
    render(<AssetSearch selected={["XLK"]} onAdd={onAdd} debounceMs={0} />);
    fireEvent.change(screen.getByLabelText("Search stocks and ETFs"), { target: { value: "app" } });
    expect(screen.getByText("Searching…")).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Apple Inc. - Common Stock")).toBeTruthy());
    expect(screen.getByText(/Source: NASDAQ Trader/)).toBeTruthy();
    expect((screen.getByLabelText("XLK already added") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByLabelText("Add AAPL"));
    expect(onAdd).toHaveBeenCalledWith(AAPL);
  });

  it("shows a no-results state", async () => {
    mockFetch(() => ({ body: { query: "zz", results: [], source: "x" } }));
    render(<AssetSearch selected={[]} onAdd={() => {}} debounceMs={0} />);
    fireEvent.change(screen.getByLabelText("Search stocks and ETFs"), { target: { value: "zzzz" } });
    await waitFor(() => expect(screen.getByText(/No matching symbols found/)).toBeTruthy());
  });

  it("shows the backend error message", async () => {
    mockFetch(() => ({ status: 503, body: { detail: "Symbol index not built." } }));
    render(<AssetSearch selected={[]} onAdd={() => {}} debounceMs={0} />);
    fireEvent.change(screen.getByLabelText("Search stocks and ETFs"), { target: { value: "a" } });
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Symbol index not built."));
  });
});

function builderBackend(report = okReport) {
  return mockFetch((url, init) => {
    if (url.includes("/assets/search"))
      return { body: { query: "x", results: [AAPL, XLK], source: "NASDAQ Trader" } };
    if (url.endsWith("/portfolios/validate")) return { body: report };
    if (url.endsWith("/portfolios") && init?.method === "POST")
      return { status: 201, body: { id: 7, name: "Untitled portfolio", validation: report } };
    if (url.endsWith("/portfolios")) return { body: [] };
    return { status: 404, body: { detail: "nope" } };
  });
}

async function addAsset(symbol: string) {
  fireEvent.change(screen.getByLabelText("Search stocks and ETFs"), { target: { value: "x" } });
  fireEvent.click(await screen.findByLabelText(`Add ${symbol}`));
}

describe("PortfolioBuilder", () => {
  it("adds and removes holdings", async () => {
    builderBackend();
    render(<PortfolioBuilder debounceMs={0} />);
    expect(screen.getByText(/No assets yet/)).toBeTruthy();
    await addAsset("AAPL");
    await addAsset("XLK");
    expect(screen.getByText("Holdings (2)")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Remove AAPL"));
    expect(screen.getByText("Holdings (1)")).toBeTruthy();
    expect(screen.queryByLabelText("Remove AAPL")).toBeNull();
  });

  it("shows helpful errors from the backend validation", async () => {
    builderBackend({
      ...okReport,
      valid: false,
      issues: [
        {
          severity: "error",
          code: "cap_infeasible",
          message: "A 30% per-asset cap requires at least 4 assets to be fully invested; you have 1.",
          symbol: null,
        },
      ],
    });
    render(<PortfolioBuilder debounceMs={0} />);
    await addAsset("AAPL");
    await waitFor(() => expect(screen.getByText(/requires at least 4 assets/)).toBeTruthy());
    expect(screen.getByText(/Configuration is invalid/)).toBeTruthy();
  });

  it("shows equal weights computed by the backend", async () => {
    builderBackend({ ...okReport, effective_weights: { AAPL: 0.5, XLK: 0.5 } });
    render(<PortfolioBuilder debounceMs={0} />);
    await addAsset("AAPL");
    await addAsset("XLK");
    await waitFor(() => expect(screen.getAllByText("50.00").length).toBe(2));
  });

  it("shows a local error for a non-numeric capital without calling validate", async () => {
    const fetchMock = builderBackend();
    render(<PortfolioBuilder debounceMs={0} />);
    await addAsset("AAPL");
    await waitFor(() => expect(screen.getByText(/Configuration is valid/)).toBeTruthy());
    fireEvent.change(screen.getByLabelText("Initial capital (USD)"), { target: { value: "lots" } });
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Initial capital must be a number."));
    const sentCapitals = fetchMock.mock.calls
      .filter(([u]) => String(u).endsWith("/validate"))
      .map(([, i]) => JSON.parse(String(i?.body)).initial_capital);
    expect(sentCapitals.length).toBeGreaterThan(0); // the valid state before the bad edit
    expect(sentCapitals.every((c) => c === 100000)).toBe(true);
  });

  it("saves a portfolio and confirms", async () => {
    const fetchMock = builderBackend();
    render(<PortfolioBuilder debounceMs={0} />);
    await addAsset("AAPL");
    fireEvent.click(screen.getByText("Save portfolio"));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Saved"));
    const post = fetchMock.mock.calls.find(
      ([u, i]) => String(u).endsWith("/portfolios") && i?.method === "POST",
    );
    const body = JSON.parse(String(post?.[1]?.body));
    expect(body).toMatchObject({ initial_capital: 100000, max_weight: 0.3, weight_mode: "equal" });
    expect(body.holdings).toEqual([{ symbol: "AAPL", weight: null }]);
  });

  it("shows a save failure", async () => {
    mockFetch((url, init) => {
      if (url.includes("/assets/search")) return { body: { query: "x", results: [AAPL], source: "s" } };
      if (url.endsWith("/validate")) return { body: okReport };
      if (init?.method === "POST") return { status: 422, body: { detail: "bad data" } };
      return { body: [] };
    });
    render(<PortfolioBuilder debounceMs={0} />);
    await addAsset("AAPL");
    fireEvent.click(screen.getByText("Save portfolio"));
    await waitFor(() => expect(within(screen.getByRole("status")).getByText(/Save failed: bad data/)).toBeTruthy());
  });
});
