"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet, apiSend, type CoverageReport, type DataStatus } from "@/lib/api";

type Load =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; status: DataStatus };

export function DataStatusView() {
  const [load, setLoad] = useState<Load>({ kind: "loading" });
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setLoad({ kind: "ready", status: await apiGet<DataStatus>("/data/status") });
    } catch (e) {
      setLoad({ kind: "error", message: (e as Error).message });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function redownload(symbol: string) {
    setBusy(true);
    setNotice(null);
    try {
      const report = await apiSend<CoverageReport>("POST", "/data/refresh", { symbols: [symbol] });
      const row = report.symbols[0];
      setNotice(
        row.status === "ok"
          ? { ok: true, text: `Re-downloaded ${symbol}: ${row.rows} rows to ${row.last_date}.` }
          : { ok: false, text: `Re-download of ${symbol} failed: ${row.message ?? row.status}` },
      );
    } catch (e) {
      setNotice({ ok: false, text: `Re-download failed: ${(e as Error).message}` });
    } finally {
      setBusy(false);
      await refresh();
    }
  }

  async function clearCache() {
    if (!window.confirm("Delete all cached price history? It will be downloaded again when needed.")) return;
    setBusy(true);
    setNotice(null);
    try {
      const r = await apiSend<{ cleared_symbols: number }>("DELETE", "/data/cache?confirm=true");
      setNotice({ ok: true, text: `Cleared ${r.cleared_symbols} cached symbol(s).` });
    } catch (e) {
      setNotice({ ok: false, text: `Clear failed: ${(e as Error).message}` });
    } finally {
      setBusy(false);
      await refresh();
    }
  }

  if (load.kind === "loading") return <p className="text-sm text-muted">Loading data status…</p>;
  if (load.kind === "error")
    return (
      <p role="alert" className="text-sm text-loss">
        Could not load data status: {load.message}
      </p>
    );

  const { status } = load;
  return (
    <div className="flex flex-col gap-6">
      <section className="rounded border border-line bg-panel p-4 text-sm">
        <h2 className="font-medium mb-2">Providers</h2>
        <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-[12rem_1fr]">
          <dt className="text-muted">Price provider</dt>
          <dd>
            {status.provider.name} ({status.provider.price_field}, not real-time)
          </dd>
          <dt className="text-muted">Attribution</dt>
          <dd>{status.provider.attribution}</dd>
          <dt className="text-muted">Symbol index</dt>
          <dd>
            {status.symbol_index.available
              ? `${status.symbol_index.count.toLocaleString("en-US")} symbols, retrieved ${status.symbol_index.retrieved_at ?? "unknown"}`
              : "Not built. Run: python -m app.update_symbol_index"}
          </dd>
          <dt className="text-muted">Cached range</dt>
          <dd className="tnum">
            {status.cached_range ? `${status.cached_range[0]} to ${status.cached_range[1]}` : "Nothing cached yet"}
          </dd>
        </dl>
        <p className="mt-3 text-xs text-muted">{status.disclaimer}</p>
      </section>

      <section className="rounded border border-line bg-panel p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="font-medium">Cached price history ({status.cached_symbols.length})</h2>
          <button
            type="button"
            onClick={clearCache}
            disabled={busy || status.cached_symbols.length === 0}
            className="rounded border border-line px-3 py-1.5 text-sm hover:bg-line disabled:opacity-50"
          >
            Clear cache
          </button>
        </div>
        {notice && (
          <p role="status" className={`mb-2 text-sm ${notice.ok ? "text-gain" : "text-loss"}`}>
            {notice.text}
          </p>
        )}
        {status.cached_symbols.length === 0 ? (
          <p className="text-sm text-muted">
            No history cached. Use “Check data coverage” in the Portfolio Builder to download it.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-muted border-b border-line">
                <tr>
                  <th className="py-1 pr-3">Ticker</th>
                  <th className="pr-3">First</th>
                  <th className="pr-3">Last</th>
                  <th className="pr-3 text-right">Rows</th>
                  <th className="pr-3">Retrieved</th>
                  <th>
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {status.cached_symbols.map((c) => (
                  <tr key={c.symbol} className="border-b border-line/50">
                    <td className="py-1.5 pr-3 font-mono">{c.symbol}</td>
                    <td className="pr-3 tnum">{c.first_date}</td>
                    <td className="pr-3 tnum">{c.last_date}</td>
                    <td className="pr-3 text-right tnum">{c.rows.toLocaleString("en-US")}</td>
                    <td className="pr-3 tnum">{c.retrieved_at.slice(0, 16).replace("T", " ")}</td>
                    <td className="text-right">
                      <button
                        type="button"
                        onClick={() => redownload(c.symbol)}
                        disabled={busy}
                        aria-label={`Re-download ${c.symbol}`}
                        className="rounded border border-line px-2 py-1 text-xs hover:bg-line disabled:opacity-50"
                      >
                        Re-download
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded border border-line bg-panel p-4">
        <h2 className="font-medium mb-2">Recent download failures ({status.failures.length})</h2>
        {status.failures.length === 0 ? (
          <p className="text-sm text-muted">None.</p>
        ) : (
          <ul className="text-sm flex flex-col gap-1">
            {status.failures.map((f) => (
              <li key={`${f.symbol}-${f.at}`}>
                <span className="font-mono">{f.symbol}</span> — {f.message}{" "}
                <span className="text-muted tnum">({f.at.slice(0, 16).replace("T", " ")})</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
