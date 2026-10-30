"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet, apiSend, type Portfolio } from "@/lib/api";
import { money } from "@/lib/format";

type Props = {
  refreshKey: number;
  activeId: number | null;
  onOpen: (portfolio: Portfolio) => void;
  onDeleted: (id: number) => void;
};

export function SavedPortfolios({ refreshKey, activeId, onOpen, onDeleted }: Props) {
  const [items, setItems] = useState<Portfolio[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    apiGet<Portfolio[]>("/portfolios")
      .then((p) => {
        setItems(p);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(load, [load, refreshKey]);

  async function remove(p: Portfolio) {
    if (!window.confirm(`Delete saved portfolio “${p.name}”? This cannot be undone.`)) return;
    try {
      await apiSend("DELETE", `/portfolios/${p.id}`);
      onDeleted(p.id);
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <section aria-labelledby="saved-heading" className="rounded border border-line bg-panel p-4">
      <h2 id="saved-heading" className="font-medium mb-3">
        Saved portfolios
      </h2>
      {error && (
        <p role="alert" className="text-loss text-sm">
          {error}
        </p>
      )}
      {items === null && !error && <p className="text-sm text-muted">Loading…</p>}
      {items?.length === 0 && <p className="text-sm text-muted">No saved portfolios yet.</p>}
      <ul className="flex flex-col gap-2">
        {items?.map((p) => (
          <li
            key={p.id}
            className={`rounded border px-3 py-2 text-sm ${p.id === activeId ? "border-accent" : "border-line"}`}
          >
            <div className="font-medium truncate">{p.name}</div>
            <div className="text-xs text-muted tnum">
              {p.holdings.length} assets · {money(p.initial_capital)} ·{" "}
              {p.validation.valid ? "valid" : "has issues"}
            </div>
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                onClick={() => onOpen(p)}
                className="rounded border border-line px-2 py-1 text-xs hover:bg-line"
              >
                Open
              </button>
              <button
                type="button"
                onClick={() => remove(p)}
                className="rounded border border-line px-2 py-1 text-xs hover:bg-line text-loss"
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
