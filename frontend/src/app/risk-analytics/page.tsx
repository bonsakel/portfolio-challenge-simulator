import { RiskAnalyticsView } from "@/components/RiskAnalyticsView";

export default function RiskAnalyticsPage() {
  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Risk Analytics</h1>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          Risk, drawdown and recovery measures for a completed backtest, calculated from its daily results. Historical
          simulation only; past results do not predict future performance.
        </p>
      </header>
      <RiskAnalyticsView />
    </div>
  );
}
