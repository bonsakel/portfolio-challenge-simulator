"use client";

import { useEffect, useState } from "react";
import { apiGet, type Health } from "@/lib/api";

type State =
  | { kind: "checking" }
  | { kind: "online"; version: string }
  | { kind: "offline"; message: string };

export function ConnectionStatus() {
  const [state, setState] = useState<State>({ kind: "checking" });

  useEffect(() => {
    const controller = new AbortController();
    const check = () =>
      apiGet<Health>("/health", controller.signal)
        .then((h) => setState({ kind: "online", version: h.version }))
        .catch((e: Error) => {
          if (!controller.signal.aborted) setState({ kind: "offline", message: e.message });
        });
    check();
    const timer = setInterval(check, 15000);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, []);

  const text =
    state.kind === "checking"
      ? "Checking backend…"
      : state.kind === "online"
        ? `Backend connected · v${state.version}`
        : "Backend offline";
  const dot =
    state.kind === "online" ? "bg-gain" : state.kind === "offline" ? "bg-loss" : "bg-muted";

  return (
    <div role="status" aria-live="polite" className="text-xs">
      <div className="flex items-center gap-2">
        <span aria-hidden className={`h-2 w-2 rounded-full ${dot}`} />
        <span>{text}</span>
      </div>
      {state.kind === "offline" && <p className="mt-1 text-muted">{state.message}</p>}
    </div>
  );
}
