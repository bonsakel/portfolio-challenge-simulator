"use client";

import { API_BASE, type BacktestRun } from "@/lib/api";
import { money, pct, pctPlain } from "@/lib/format";

const ratio = (v: number | null) => (v === null ? "n/a" : v.toFixed(2));

export function BacktestResults({ run }: { run: BacktestRun }) {
  const s = run.summary;
  if (!s) return null;
  const entries = Object.entries(s.strategies);
  return (
    <section aria-labelledby="results-heading" className="rounded border border-line bg-panel p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="results-heading" className="font-medium">
          Results: {run.portfolio_name} (run #{run.id})
        </h2>
        <a
          href={`${API_BASE}/backtests/${run.id}/export.csv`}
          className="rounded border border-line px-3 py-1.5 text-sm hover:bg-line"
        >
          Download CSV
        </a>
      </div>
      <p className="mt-1 text-xs text-muted">
        All strategies were evaluated over the same {s.period.trading_days} trading days,{" "}
        {s.period.first_return_date} to {s.period.last_return_date}
        {s.period.first_return_date !== s.period.requested_start || s.period.last_return_date !== s.period.requested_end
          ? ` (requested ${s.period.requested_start} to ${s.period.requested_end}; the first and last common trading days were used)`
          : ""}
        . Starting capital {money(entries[0][1].initial_value)}. Annualized with 252 trading days.
      </p>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-muted border-b border-line">
            <tr>
              <th className="py-1 pr-3">Strategy</th>
              <th className="pr-3 text-right">Ending value</th>
              <th className="pr-3 text-right">Total return</th>
              <th className="pr-3 text-right">Annualized return</th>
              <th className="pr-3 text-right">Annualized volatility</th>
              <th className="pr-3 text-right">Sharpe</th>
              <th className="pr-3 text-right">Max drawdown</th>
              <th className="pr-3 text-right">Transaction costs</th>
              <th className="pr-3 text-right">Turnover (sum)</th>
              <th className="text-right">Rebalances</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, m]) => (
              <tr key={key} className="border-b border-line/50 align-top">
                <td className="py-1.5 pr-3">
                  {m.label}
                  <div className="text-xs text-muted">
                    {m.model
                      ? `agent decides daily · model: ${m.model.label}, seed ${m.model.seed ?? "n/a"}`
                      : `${m.rebalance}, trade threshold ${(m.turnover_threshold * 100).toFixed(1)} pt`}
                  </div>
                </td>
                <td className="pr-3 text-right tnum">{money(m.ending_value)}</td>
                <td className={`pr-3 text-right tnum ${m.total_return < 0 ? "text-loss" : "text-gain"}`}>
                  {pct(m.total_return)}
                </td>
                <td className="pr-3 text-right tnum">{pct(m.annualized_return)}</td>
                <td className="pr-3 text-right tnum">{pctPlain(m.annualized_volatility)}</td>
                <td className="pr-3 text-right tnum">{ratio(m.sharpe_ratio)}</td>
                <td className="pr-3 text-right tnum">{pct(m.maximum_drawdown)}</td>
                <td className="pr-3 text-right tnum">{money(m.transaction_costs)}</td>
                <td className="pr-3 text-right tnum">{(m.total_turnover * 100).toFixed(0)}%</td>
                <td className="text-right tnum">{m.rebalance_count}</td>
              </tr>
            ))}
            {s.benchmark && (
              <tr className="border-b border-line/50 align-top text-muted">
                <td className="py-1.5 pr-3">
                  {s.benchmark.label}
                  <div className="text-xs">benchmark, no trading costs</div>
                </td>
                <td className="pr-3 text-right tnum">{money(s.benchmark.ending_value)}</td>
                <td className="pr-3 text-right tnum">{pct(s.benchmark.total_return)}</td>
                <td className="pr-3 text-right tnum">{pct(s.benchmark.annualized_return)}</td>
                <td className="pr-3 text-right tnum">{pctPlain(s.benchmark.annualized_volatility)}</td>
                <td className="pr-3 text-right tnum">{ratio(s.benchmark.sharpe_ratio)}</td>
                <td className="pr-3 text-right tnum">{pct(s.benchmark.maximum_drawdown)}</td>
                <td className="pr-3 text-right tnum">–</td>
                <td className="pr-3 text-right tnum">–</td>
                <td className="text-right tnum">–</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <h3 className="mt-4 text-sm font-medium">Data used</h3>
      <p className="text-xs text-muted">
        {s.data.provider}, {s.data.price_field}; common price history {s.data.common_start} to{" "}
        {s.data.common_end} ({s.data.common_price_rows.toLocaleString("en-US")} dates present for every asset).
      </p>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-muted border-b border-line">
            <tr>
              <th className="py-1 pr-3">Ticker</th>
              <th className="pr-3">Source</th>
              <th className="pr-3">First</th>
              <th className="pr-3">Last</th>
              <th className="text-right">Dates dropped (not common)</th>
            </tr>
          </thead>
          <tbody>
            {s.data.symbols.map((d) => (
              <tr key={d.symbol} className="border-b border-line/50">
                <td className="py-1 pr-3 font-mono">{d.symbol}</td>
                <td className="pr-3">{d.source}</td>
                <td className="pr-3 tnum">{d.first_date}</td>
                <td className="pr-3 tnum">{d.last_date}</td>
                <td className="text-right tnum">{s.data.dates_dropped_per_symbol[d.symbol] ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {s.data.notes.length > 0 && (
        <ul className="mt-2 list-disc pl-5 text-xs text-muted">
          {s.data.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}

      <p className="mt-4 text-xs text-muted">{s.disclaimer}</p>
    </section>
  );
}
