import { StrategyLab } from "@/components/StrategyLab";

export default function StrategyLabPage() {
  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Strategy Lab</h1>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          How each strategy works, which trained reinforcement-learning models exist and what they are compatible with, and
          training of new PPO or DDPG models. Historical simulation only; not investment advice.
        </p>
      </header>
      <StrategyLab />
    </div>
  );
}
