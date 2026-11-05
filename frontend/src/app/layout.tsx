import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Strategy comparison | PortfolioLab",
  description: "Compare traditional and reinforcement-learning portfolio strategies during market downturns.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-ink">
        <div className="min-h-screen">
          <header className="border-b border-line bg-panel">
            <div className="mx-auto flex max-w-[1440px] items-center justify-between px-5 py-4 md:px-8">
              <div className="text-sm font-semibold tracking-tight">PortfolioLab</div>
              <Nav />
            </div>
          </header>
          <main className="mx-auto w-full max-w-[1440px] px-4 py-6 md:px-8 md:py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
