"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { EXPANDED_MODE } from "@/lib/features";

const CORE_ITEMS = [{ href: "/", name: "Research" }];
const EXPANDED_ITEMS = [
  { href: "/portfolio-builder", name: "Portfolio Builder" },
  { href: "/strategy-lab", name: "Strategy Lab" },
  { href: "/backtests", name: "Backtests" },
  { href: "/risk-analytics", name: "Risk Analytics" },
  { href: "/research-mode", name: "Research Mode" },
  { href: "/settings", name: "Settings / Data Status" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main navigation">
      <ul className="flex items-center gap-2 text-sm">
        {[...CORE_ITEMS, ...(EXPANDED_MODE ? EXPANDED_ITEMS : [])].map((item) => (
          <li key={item.href}>
            <Link
              href={item.href}
              aria-current={pathname === item.href ? "page" : undefined}
              className="rounded px-3 py-2 text-muted hover:bg-bg hover:text-ink aria-[current=page]:bg-bg aria-[current=page]:text-accent"
            >
              {item.name}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
