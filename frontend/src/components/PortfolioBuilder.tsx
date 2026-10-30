"use client";

import { useState } from "react";
import { AssetSearch } from "@/components/AssetSearch";
import { CoverageCheck } from "@/components/CoverageCheck";
import { SavedPortfolios } from "@/components/SavedPortfolios";
import { ValidationPanel, useValidation } from "@/components/ValidationPanel";
import { apiSend, type Asset, type Portfolio } from "@/lib/api";
import {
  DEFAULT_FORM,
  buildPayload,
  formFromPortfolio,
  type PortfolioForm,
} from "@/lib/portfolio";

const inputClass = "w-full rounded border border-line bg-bg px-3 py-2 text-sm tnum";

export function PortfolioBuilder({ debounceMs = 300 }: { debounceMs?: number }) {
  const [form, setForm] = useState<PortfolioForm>(DEFAULT_FORM);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [saveMessage, setSaveMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const validation = useValidation(form, debounceMs);

  const effective = validation.kind === "report" ? validation.report.effective_weights : {};

  function update(patch: Partial<PortfolioForm>) {
    setForm((f) => ({ ...f, ...patch }));
    setSaveMessage(null);
  }

  function addAsset(asset: Asset) {
    setForm((f) =>
      f.holdings.some((h) => h.symbol === asset.symbol)
        ? f
        : { ...f, holdings: [...f.holdings, { symbol: asset.symbol, name: asset.name, weightPct: "" }] },
    );
    setSaveMessage(null);
  }

  function removeAsset(symbol: string) {
    update({ holdings: form.holdings.filter((h) => h.symbol !== symbol) });
  }

  function setWeight(symbol: string, weightPct: string) {
    update({ holdings: form.holdings.map((h) => (h.symbol === symbol ? { ...h, weightPct } : h)) });
  }

  function open(p: Portfolio) {
    setForm(formFromPortfolio(p));
    setEditingId(p.id);
    setSaveMessage(null);
  }

  function reset() {
    setForm(DEFAULT_FORM);
    setEditingId(null);
    setSaveMessage(null);
  }

  async function save() {
    const built = buildPayload(form);
    if ("error" in built) {
      setSaveMessage({ ok: false, text: built.error });
      return;
    }
    try {
      const saved =
        editingId === null
          ? await apiSend<Portfolio>("POST", "/portfolios", built.payload)
          : await apiSend<Portfolio>("PUT", `/portfolios/${editingId}`, built.payload);
      setEditingId(saved.id);
      setForm((f) => ({ ...f, name: saved.name }));
      setRefreshKey((k) => k + 1);
      setSaveMessage({
        ok: true,
        text: saved.validation.valid
          ? `Saved “${saved.name}”.`
          : `Saved “${saved.name}” as a draft with issues; it must be valid before it can be backtested.`,
      });
    } catch (e) {
      setSaveMessage({ ok: false, text: `Save failed: ${(e as Error).message}` });
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <div className="flex min-w-0 flex-col gap-6">
        <section className="rounded border border-line bg-panel p-4">
          <AssetSearch
            selected={form.holdings.map((h) => h.symbol)}
            onAdd={addAsset}
            debounceMs={debounceMs}
          />
        </section>

        <section aria-labelledby="config-heading" className="rounded border border-line bg-panel p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 id="config-heading" className="font-medium">
              {editingId === null ? "New portfolio" : `Editing portfolio #${editingId}`}
            </h2>
            <button type="button" onClick={reset} className="text-xs text-accent hover:underline">
              Start new
            </button>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <label htmlFor="pf-name" className="block text-sm mb-1">
                Portfolio name
              </label>
              <input
                id="pf-name"
                className={inputClass}
                value={form.name}
                maxLength={80}
                onChange={(e) => update({ name: e.target.value })}
              />
            </div>
            <div>
              <label htmlFor="pf-capital" className="block text-sm mb-1">
                Initial capital (USD)
              </label>
              <input
                id="pf-capital"
                inputMode="decimal"
                className={inputClass}
                value={form.capital}
                onChange={(e) => update({ capital: e.target.value })}
              />
            </div>
            <div>
              <label htmlFor="pf-cap" className="block text-sm mb-1">
                Max weight per asset (%)
              </label>
              <input
                id="pf-cap"
                inputMode="decimal"
                className={inputClass}
                value={form.maxWeightPct}
                onChange={(e) => update({ maxWeightPct: e.target.value })}
                aria-describedby="pf-cap-help"
              />
              <p id="pf-cap-help" className="text-xs text-muted mt-1">
                Long-only, fully invested. Research baseline is 30%.
              </p>
            </div>
          </div>

          <fieldset className="mt-4 flex gap-6 text-sm">
            <legend className="mb-1">Target weights</legend>
            {(["equal", "custom"] as const).map((m) => (
              <label key={m} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="weight-mode"
                  checked={form.mode === m}
                  onChange={() => update({ mode: m })}
                />
                {m === "equal" ? "Equal weight (1/N)" : "Custom weights"}
              </label>
            ))}
          </fieldset>

          <div className="mt-4">
            <h3 className="text-sm font-medium mb-2">Holdings ({form.holdings.length})</h3>
            {form.holdings.length === 0 ? (
              <p className="text-sm text-muted">No assets yet. Use the search above to add some.</p>
            ) : (
              <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-muted border-b border-line">
                  <tr>
                    <th className="py-1 pr-3">Ticker</th>
                    <th className="pr-3">Name</th>
                    <th className="pr-3 text-right">Target weight (%)</th>
                    <th>
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {form.holdings.map((h) => (
                    <tr key={h.symbol} className="border-b border-line/50">
                      <td className="py-1.5 pr-3 font-mono">{h.symbol}</td>
                      <td className="pr-3 truncate max-w-[16rem]">{h.name || "—"}</td>
                      <td className="pr-3 text-right tnum">
                        {form.mode === "custom" ? (
                          <input
                            aria-label={`Target weight for ${h.symbol} in percent`}
                            inputMode="decimal"
                            className="w-24 rounded border border-line bg-bg px-2 py-1 text-right"
                            value={h.weightPct}
                            onChange={(e) => setWeight(h.symbol, e.target.value)}
                          />
                        ) : effective[h.symbol] !== undefined ? (
                          (effective[h.symbol] * 100).toFixed(2)
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="text-right">
                        <button
                          type="button"
                          onClick={() => removeAsset(h.symbol)}
                          aria-label={`Remove ${h.symbol}`}
                          className="rounded border border-line px-2 py-1 text-xs hover:bg-line"
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            )}
          </div>

          <div className="mt-4">
            <ValidationPanel state={validation} />
          </div>

          <div className="mt-4">
            <CoverageCheck symbols={form.holdings.map((h) => h.symbol)} />
          </div>

          <div className="mt-4 flex items-center gap-4">
            <button
              type="button"
              onClick={save}
              disabled={form.holdings.length === 0}
              className="rounded bg-accent px-4 py-2 text-sm font-medium text-bg disabled:opacity-50"
            >
              {editingId === null ? "Save portfolio" : "Save changes"}
            </button>
            {saveMessage && (
              <p role="status" className={`text-sm ${saveMessage.ok ? "text-gain" : "text-loss"}`}>
                {saveMessage.text}
              </p>
            )}
          </div>
        </section>
      </div>

      <SavedPortfolios
        refreshKey={refreshKey}
        activeId={editingId}
        onOpen={open}
        onDeleted={(id) => {
          if (id === editingId) reset();
        }}
      />
    </div>
  );
}
