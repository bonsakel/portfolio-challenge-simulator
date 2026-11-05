import { redirect } from "next/navigation";
import { BacktestRunner } from "@/components/BacktestRunner";
import { EXPANDED_MODE } from "@/lib/features";

export default function BacktestsPage() {
  if (!EXPANDED_MODE) redirect("/");
  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Backtests</h1>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          Run Equal Weight, Mean Variance and Minimum Variance on a saved portfolio and compare them over identical
          dates, optionally against a buy-and-hold benchmark. Results are historical simulations, not forecasts or
          advice. See the Risk Analytics page for risk, drawdown and recovery measures.
        </p>
      </header>
      <BacktestRunner />
    </div>
  );
}
