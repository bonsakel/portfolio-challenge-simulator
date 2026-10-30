"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/", name: "Overview", ready: true },
  { href: "/portfolio-builder", name: "Portfolio Builder", ready: true },
  { href: "/strategy-lab", name: "Strategy Lab", ready: true },
  { href: "/backtests", name: "Backtests", ready: true },
  { href: "/risk-analytics", name: "Risk Analytics", ready: true },
  { href: "/research-mode", name: "Research Mode", ready: true },
  { href: "/settings", name: "Settings / Data Status", ready: true },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main">
      <ul className="flex flex-col gap-1 text-sm">
        {ITEMS.map((item) => (
          <li key={item.href}>
            {item.ready ? (
              <Link
                href={item.href}
                aria-current={pathname === item.href ? "page" : undefined}
                className="block rounded px-3 py-2 hover:bg-line aria-[current=page]:bg-line aria-[current=page]:text-accent"
              >
                {item.name}
              </Link>
            ) : (
              <span
                aria-disabled="true"
                className="flex items-center justify-between rounded px-3 py-2 text-muted"
              >
                {item.name}
                <span className="text-[10px] uppercase tracking-wide border border-line rounded px-1">
                  Planned
                </span>
              </span>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );
}
