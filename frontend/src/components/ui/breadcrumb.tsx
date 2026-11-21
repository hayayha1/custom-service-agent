"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight, Home } from "lucide-react";

const LABELS: Record<string, string> = {
  dashboard: "控制台",
  knowledge: "知识库",
  chat: "客服对话",
  monitoring: "Agent 监测",
  settings: "模型设置",
};

export default function Breadcrumb() {
  const pathname = usePathname();
  const segments = pathname.split("/").filter(Boolean);

  if (segments.length <= 1) {
    return null;
  }

  return (
    <nav className="mb-6 flex items-center space-x-2 text-sm text-muted-foreground">
      <Link
        href="/dashboard"
        className="flex items-center hover:text-foreground transition-colors"
      >
        <Home className="h-4 w-4" />
      </Link>
      {segments.map((segment, index) => {
        const href = "/" + segments.slice(0, index + 1).join("/");
        const isLast = index === segments.length - 1;
        const label = LABELS[segment] || segment;

        return (
          <div key={href} className="flex items-center space-x-2">
            <ChevronRight className="h-4 w-4" />
            {isLast ? (
              <span className="font-medium text-foreground">{label}</span>
            ) : (
              <Link href={href} className="hover:text-foreground transition-colors">
                {label}
              </Link>
            )}
          </div>
        );
      })}
    </nav>
  );
}
