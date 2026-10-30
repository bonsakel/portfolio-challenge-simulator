import { DataStatusView } from "@/components/DataStatusView";

export default function SettingsPage() {
  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Settings / Data Status</h1>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          Where market data comes from, what is cached locally, and what failed. Price history is
          downloaded once per symbol and reused; re-downloading is always a deliberate action.
        </p>
      </header>
      <DataStatusView />
    </div>
  );
}
