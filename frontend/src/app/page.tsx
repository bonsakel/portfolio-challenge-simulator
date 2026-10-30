import { RecentBacktests } from "@/components/RecentBacktests";
import { ResearchSnapshot } from "@/components/ResearchSnapshot";

export default function OverviewPage() {
  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-sm text-muted mt-1 max-w-2xl">
          PortfolioLab runs historical simulations for research and education. Results are not
          investment advice and are not predictions of future performance.
        </p>
      </header>
      <section aria-labelledby="research-heading" className="rounded border border-line bg-panel p-5">
        <h2 id="research-heading" className="font-medium mb-3">
          CSC-475 research results (read from local artifacts)
        </h2>
        <ResearchSnapshot />
      </section>
      <section aria-labelledby="recent-heading" className="rounded border border-line bg-panel p-5">
        <h2 id="recent-heading" className="font-medium mb-3">
          Recent custom backtests (exploratory)
        </h2>
        <RecentBacktests />
      </section>
    </div>
  );
}
