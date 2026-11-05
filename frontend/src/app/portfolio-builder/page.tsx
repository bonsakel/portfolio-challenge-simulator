import { redirect } from "next/navigation";
import { PortfolioBuilder } from "@/components/PortfolioBuilder";
import { EXPANDED_MODE } from "@/lib/features";

export default function PortfolioBuilderPage() {
  if (!EXPANDED_MODE) redirect("/");
  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Portfolio Builder</h1>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          Search for stocks and ETFs, set capital and constraints, and save a portfolio. Search covers
          currently listed US securities only; being listed does not guarantee price history, so use
          “Check data coverage” to verify it. Historical tests on a hand-picked universe
          are selected-universe studies, not survivorship-free.
        </p>
      </header>
      <PortfolioBuilder />
    </div>
  );
}
