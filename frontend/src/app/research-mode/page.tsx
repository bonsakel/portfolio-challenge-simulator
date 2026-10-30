import { ResearchMode } from "@/components/ResearchMode";

export default function ResearchModePage() {
  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Research Mode</h1>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          The fixed CSC-475 experiment: reinforcement learning (PPO, DDPG) against Equal Weight, Mean Variance and
          Minimum Variance through three market downturns. Every number is read from the stored research results.
          Historical simulation only; not investment advice.
        </p>
      </header>
      <ResearchMode />
    </div>
  );
}
