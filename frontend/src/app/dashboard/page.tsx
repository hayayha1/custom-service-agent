"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Book,
  Cpu,
  MessageSquare,
  RefreshCw,
  Settings,
  Wrench,
} from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { api, type SystemOverviewResponse } from "@/lib/api";

export default function DashboardPage() {
  const [overview, setOverview] = useState<SystemOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadOverview = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getOverview();
      setOverview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法连接到后端服务");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadOverview();
  }, []);

  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">控制台概览</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              垂直客服 Agent 运行状态、分层记忆底座与当前模型配置一览
            </p>
          </div>
          <button
            onClick={() => void loadOverview()}
            disabled={loading}
            className="inline-flex items-center justify-center rounded-lg border bg-card px-4 py-2 text-sm font-medium shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            刷新状态
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            后端连接失败：{error}（请确认 FastAPI 服务已在 8000 端口启动）
          </div>
        )}

        {/* Stats Cards (rag-web-ui card grid style) */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border bg-card p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-muted-foreground">服务与 Redis 状态</span>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="mt-3 flex items-center gap-2">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  overview?.status === "ok" ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              <span className="text-2xl font-bold">
                {overview ? (overview.status === "ok" ? "运行正常" : "降级运行") : "--"}
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              存储模式：
              {overview
                ? overview.storage_mode === "postgres+redis"
                  ? "PostgreSQL + Redis"
                  : "本地内存降级模式（未启动 Docker）"
                : "检测中"}
            </p>
          </div>

          <div className="rounded-xl border bg-card p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-muted-foreground">知识库文档数</span>
              <Book className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="mt-3 text-2xl font-bold">
              {overview ? overview.kb_document_count : "--"}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              PostgreSQL + pgvector 向量检索库
            </p>
          </div>

          <div className="rounded-xl border bg-card p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-muted-foreground">已注册客服工具</span>
              <Wrench className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="mt-3 text-2xl font-bold">
              {overview ? overview.registered_tools.length : "--"}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              含知识检索、订单查询、幂等建单与转人工
            </p>
          </div>

          <div className="rounded-xl border bg-card p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-muted-foreground">当前对话模型</span>
              <Cpu className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="mt-3 text-xl font-bold truncate" title={overview?.models.llm.model}>
              {overview?.models.llm.model || "--"}
            </div>
            <p className="mt-1 text-xs text-muted-foreground truncate">
              Reranker：
              {overview?.models.reranker.enabled
                ? `已启用 (${overview.models.reranker.model})`
                : "未启用"}
            </p>
          </div>
        </div>

        {/* Active Models & Tools */}
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl border bg-card p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">当前模型配置</h2>
              <Link
                href="/dashboard/settings"
                className="text-xs font-medium text-primary hover:underline inline-flex items-center"
              >
                前往修改 <ArrowRight className="ml-1 h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between rounded-lg bg-muted/50 px-4 py-3">
                <span className="text-muted-foreground">LLM 对话模型</span>
                <span className="font-mono text-xs font-medium">
                  {overview?.models.llm.model || "--"}
                </span>
              </div>
              <div className="flex items-center justify-between rounded-lg bg-muted/50 px-4 py-3">
                <span className="text-muted-foreground">Embedding 向量模型</span>
                <span className="font-mono text-xs font-medium">
                  {overview
                    ? `${overview.models.embedding.model} (${overview.models.embedding.dim}d)`
                    : "--"}
                </span>
              </div>
              <div className="flex items-center justify-between rounded-lg bg-muted/50 px-4 py-3">
                <span className="text-muted-foreground">Reranker 重排模型</span>
                <span className="font-mono text-xs font-medium">
                  {overview
                    ? overview.models.reranker.enabled
                      ? overview.models.reranker.model
                      : "未启用"
                    : "--"}
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-xl border bg-card p-6 shadow-sm space-y-4">
            <h2 className="text-lg font-semibold">Agent 已挂载工具集</h2>
            <div className="flex flex-wrap gap-2">
              {(overview?.registered_tools || []).map((tool) => (
                <span
                  key={tool}
                  className="inline-flex items-center rounded-md border bg-secondary px-3 py-1.5 font-mono text-xs font-medium text-secondary-foreground"
                >
                  {tool}
                </span>
              ))}
              {!overview && (
                <span className="text-sm text-muted-foreground">等待加载后端工具列表...</span>
              )}
            </div>
            <div className="pt-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
              <Link
                href="/dashboard/chat"
                className="flex items-center justify-center rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                <MessageSquare className="mr-2 h-4 w-4" />
                开始对话测试
              </Link>
              <Link
                href="/dashboard/knowledge"
                className="flex items-center justify-center rounded-lg border bg-card px-4 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
              >
                <Book className="mr-2 h-4 w-4" />
                管理知识库
              </Link>
              <Link
                href="/dashboard/settings"
                className="flex items-center justify-center rounded-lg border bg-card px-4 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
              >
                <Settings className="mr-2 h-4 w-4" />
                配置模型
              </Link>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
