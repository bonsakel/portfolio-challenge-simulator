import type { Metadata } from "next";
import type { ReactNode } from "react";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "PortfolioLab",
  description: "Historical portfolio research and backtesting simulator",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="flex min-h-screen flex-col md:flex-row">
          <aside className="w-full md:w-56 shrink-0 border-b md:border-b-0 md:border-r border-line bg-panel p-4 flex flex-col gap-6">
            <div>
              <div className="text-lg font-semibold tracking-tight">PortfolioLab</div>
              <div className="text-xs text-muted">Historical research simulator</div>
            </div>
            <Nav />
            <div className="md:mt-auto">
              <ConnectionStatus />
            </div>
          </aside>
          <main className="min-w-0 flex-1 p-4 md:p-8 max-w-6xl">{children}</main>
        </div>
      </body>
    </html>
  );
}
