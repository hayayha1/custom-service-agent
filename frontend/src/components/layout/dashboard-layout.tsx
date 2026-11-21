"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  Activity,
  Book,
  LayoutDashboard,
  Menu,
  MessageSquare,
  Settings,
  X,
} from "lucide-react";
import Breadcrumb from "@/components/ui/breadcrumb";

const navigation = [
  { name: "控制台概览", href: "/dashboard", icon: LayoutDashboard, exact: true },
  { name: "知识库", href: "/dashboard/knowledge", icon: Book, exact: false },
  { name: "客服对话", href: "/dashboard/chat", icon: MessageSquare, exact: false },
  { name: "Agent 监测", href: "/dashboard/monitoring", icon: Activity, exact: false },
  { name: "设置", href: "/dashboard/settings", icon: Settings, exact: false },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile menu button */}
      <div className="lg:hidden fixed top-0 left-0 m-4 z-50">
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="p-2 rounded-md bg-primary text-primary-foreground shadow"
          aria-label="切换导航菜单"
        >
          {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Sidebar (adapted from rag-web-ui template) */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 transform bg-card border-r transition-transform duration-200 ease-in-out lg:translate-x-0 ${
          isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Sidebar header */}
          <div className="flex h-16 items-center border-b px-6">
            <Link
              href="/dashboard"
              className="flex items-center gap-3 text-base font-semibold hover:text-primary transition-colors"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/logo.svg"
                alt="Logo"
                className="w-9 h-9 rounded-lg shrink-0"
              />
              <span className="truncate">垂直客服 Agent</span>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-2 px-4 py-6">
            {navigation.map((item) => {
              const isActive = item.exact
                ? pathname === item.href
                : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={`group flex items-center rounded-lg px-4 py-3 text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "bg-gradient-to-r from-primary/10 to-primary/5 text-primary shadow-sm"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground hover:shadow-sm"
                  }`}
                >
                  <item.icon
                    className={`mr-3 h-5 w-5 transition-transform duration-200 ${
                      isActive ? "text-primary scale-110" : "group-hover:scale-110"
                    }`}
                  />
                  <span className="font-medium">{item.name}</span>
                  {isActive && (
                    <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary" />
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Footer info */}
          <div className="border-t px-6 py-4 text-xs text-muted-foreground">
            <div className="font-medium text-foreground">LangGraph · Redis · pgvector</div>
            <div className="mt-1">垂直领域智能客服工作台</div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="lg:pl-64">
        <main className="min-h-screen py-6 px-4 sm:px-6 lg:px-8">
          <Breadcrumb />
          {children}
        </main>
      </div>
    </div>
  );
}
