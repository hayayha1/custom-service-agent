import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "垂直客服 Agent 控制台",
  description: "基于 LangGraph + Redis + PostgreSQL (pgvector) 的垂直客服智能体工作台",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="font-sans">{children}</body>
    </html>
  );
}
