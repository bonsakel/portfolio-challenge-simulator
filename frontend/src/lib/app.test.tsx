import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { label, pct } from "@/lib/format";

afterEach(() => vi.restoreAllMocks());

describe("format", () => {
  it("formats signed percentages", () => {
    expect(pct(-0.41)).toBe("−41.0%");
    expect(pct(0.0256, 2)).toBe("+2.56%");
  });
  it("labels slugs", () => {
    expect(label("global_financial_crisis")).toBe("Global Financial Crisis");
  });
});

describe("ConnectionStatus", () => {
  it("shows connected with the backend version", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "ok", version: "0.1.0", server_time_utc: "" }),
      }),
    );
    render(<ConnectionStatus />);
    await waitFor(() => expect(screen.getByText(/Backend connected · v0.1.0/)).toBeTruthy());
  });

  it("shows an offline message when unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
    render(<ConnectionStatus />);
    await waitFor(() => expect(screen.getByText("Backend offline")).toBeTruthy());
    expect(screen.getByText(/Cannot reach the PortfolioLab backend/)).toBeTruthy();
  });
});
