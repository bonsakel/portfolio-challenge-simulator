"use client";

import { useEffect, useState } from "react";
import { apiGet, type Asset, type SearchResponse } from "@/lib/api";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "done"; results: Asset[]; source: string };

type Props = {
  selected: string[];
  onAdd: (asset: Asset) => void;
  debounceMs?: number;
};

export function AssetSearch({ selected, onAdd, debounceMs = 300 }: Props) {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setState({ kind: "idle" });
      return;
    }
    setState({ kind: "loading" });
    const controller = new AbortController();
    const timer = setTimeout(() => {
      apiGet<SearchResponse>(`/assets/search?q=${encodeURIComponent(q)}&limit=15`, controller.signal)
        .then((r) => setState({ kind: "done", results: r.results, source: r.source }))
        .catch((e: Error) => {
          if (!controller.signal.aborted) setState({ kind: "error", message: e.message });
        });
    }, debounceMs);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, debounceMs]);

  return (
    <div>
      <label htmlFor="asset-search" className="block text-sm font-medium mb-1">
        Search stocks and ETFs
      </label>
      <input
        id="asset-search"
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ticker or company name, e.g. XLK or Microsoft"
        autoComplete="off"
        maxLength={40}
        className="w-full rounded border border-line bg-bg px-3 py-2 text-sm"
      />
      <div aria-live="polite" className="mt-2 text-sm">
        {state.kind === "idle" && (
          <p className="text-muted">Type a ticker or name to search currently listed US securities.</p>
        )}
        {state.kind === "loading" && <p className="text-muted">Searching…</p>}
        {state.kind === "error" && (
          <p role="alert" className="text-loss">
            Search failed: {state.message}
          </p>
        )}
        {state.kind === "done" && state.results.length === 0 && (
          <p className="text-muted">No matching symbols found for “{query.trim()}”.</p>
        )}
        {state.kind === "done" && state.results.length > 0 && (
          <>
            <ul className="divide-y divide-line border border-line rounded max-h-72 overflow-y-auto">
              {state.results.map((a) => {
                const added = selected.includes(a.symbol);
                return (
                  <li key={a.symbol} className="flex items-center gap-3 px-3 py-2">
                    <span className="w-16 font-mono font-medium">{a.symbol}</span>
                    <span className="min-w-0 flex-1 truncate" title={a.name}>
                      {a.name}
                    </span>
                    <span className="text-xs text-muted whitespace-nowrap">
                      {a.asset_type} · {a.exchange}
                    </span>
                    <button
                      type="button"
                      disabled={added}
                      onClick={() => onAdd(a)}
                      aria-label={added ? `${a.symbol} already added` : `Add ${a.symbol}`}
                      className="rounded border border-line px-2 py-1 text-xs hover:bg-line disabled:opacity-50"
                    >
                      {added ? "Added" : "Add"}
                    </button>
                  </li>
                );
              })}
            </ul>
            <p className="mt-1 text-xs text-muted">Source: {state.source}</p>
          </>
        )}
      </div>
    </div>
  );
}
