"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  Clock,
  Cpu,
  ExternalLink,
  Gauge,
  RefreshCw,
  Search,
  Wrench,
  Zap,
} from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import {
  api,
  type MetricsSummaryResponse,
  type TraceRecordItem,
} from "@/lib/api";

const SPAN_TYPE_LABELS: Record<string, string> = {
  graph_node: "图节点",
  llm: "LLM 推理",
  tool: "工具调用",
  retrieval_rewrite: "检索1: Query改写",
  retrieval_dense: "检索2A: 向量检索",
  retrieval_sparse: "检索2B: 全文检索",
  retrieval_rrf: "检索3: RRF融合",
  retrieval_rerank: "检索4: Rerank精排",
};

export default function MonitoringPage() {
  const [metrics, setMetrics] = useState<MetricsSummaryResponse | null>(null);
  const [traces, setTraces] = useState<TraceRecordItem[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<TraceRecordItem | null>(null);
  const [statusFilter, setStatusFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [m, t] = await Promise.all([
        api.getMetricsSummary(3600),
        api.listTraces({ status: statusFilter || undefined, limit: 30 }),
      ]);
      setMetrics(m);
      setTraces(t.traces);
      if (t.traces.length > 0) {
        setSelectedTrace((prev) => {
          if (prev) {
            const found = t.traces.find((item) => item.trace_id === prev.trace_id);
            if (found) return found;
          }
          return t.traces[0];
        });
      } else {
        setSelectedTrace(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载监测数据失败");
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header & Langfuse Status */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Agent 全链路监测</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              全链路 Trace 瀑布流（慢在哪、错在哪）、工具调用监控、Token 消耗与双维度 TPS 指标
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {metrics && (
              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium ${
                  metrics.langfuse_connected
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700"
                    : "border-border bg-muted text-muted-foreground"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    metrics.langfuse_connected ? "bg-emerald-500" : "bg-slate-400"
                  }`}
                />
                {metrics.langfuse_connected
                  ? "Langfuse 已连接同步"
                  : "本地实时监测已启用（可在设置中连接 Langfuse）"}
              </span>
            )}
            {metrics?.langfuse_connected && metrics.langfuse_host && (
              <a
                href={metrics.langfuse_host}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center rounded-lg border bg-card px-3 py-1.5 text-xs font-medium hover:bg-accent"
              >
                打开 Langfuse 看板 <ExternalLink className="ml-1.5 h-3.5 w-3.5" />
              </a>
            )}
            <button
              onClick={() => void loadData()}
              disabled={loading}
              className="inline-flex items-center rounded-lg border bg-card px-3.5 py-1.5 text-xs font-medium shadow-sm hover:bg-accent disabled:opacity-50"
            >
              <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              刷新指标
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* 4 Core KPI Cards: Request TPS & Latency, Token & LLM TPS, Tool Monitoring, Retrieval Health */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border bg-card p-5 shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span>系统请求 TPS & 延迟</span>
              <Gauge className="h-4 w-4 text-primary" />
            </div>
            <div className="text-2xl font-bold">
              {metrics ? `${metrics.request_tps} req/s` : "--"}
            </div>
            <div className="text-xs text-muted-foreground space-y-0.5">
              <div>总请求：{metrics?.total_requests ?? 0} 次（近1分钟峰值 {metrics?.peak_minute_tps ?? 0} req/s）</div>
              <div>
                延迟：P50 <span className="font-mono font-medium">{metrics?.p50_latency_ms ?? 0}ms</span>{" "}
                / P95 <span className="font-mono font-medium">{metrics?.p95_latency_ms ?? 0}ms</span>
              </div>
            </div>
          </div>

          <div className="rounded-xl border bg-card p-5 shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span>Token 消耗 & LLM 生成 TPS</span>
              <Zap className="h-4 w-4 text-amber-500" />
            </div>
            <div className="text-2xl font-bold">
              {metrics ? `${metrics.avg_llm_tps} tok/s` : "--"}
            </div>
            <div className="text-xs text-muted-foreground space-y-0.5">
              <div>
                累计 Token：<span className="font-mono font-medium">{metrics?.total_tokens ?? 0}</span>
              </div>
              <div>
                输入 {metrics?.total_prompt_tokens ?? 0} / 输出 {metrics?.total_completion_tokens ?? 0}
              </div>
            </div>
          </div>

          <div className="rounded-xl border bg-card p-5 shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span>工具调用监控</span>
              <Wrench className="h-4 w-4 text-primary" />
            </div>
            <div className="text-2xl font-bold">
              {metrics ? `${Math.round(metrics.tool_success_rate * 100)}% 成功` : "--"}
            </div>
            <div className="text-xs text-muted-foreground space-y-0.5">
              <div>累计调用：{metrics?.tool_total_calls ?? 0} 次</div>
              <div>
                超时率 {Math.round((metrics?.tool_timeout_rate ?? 0) * 100)}% / 错误率{" "}
                {Math.round((metrics?.tool_error_rate ?? 0) * 100)}%
              </div>
            </div>
          </div>

          <div className="rounded-xl border bg-card p-5 shadow-sm space-y-2">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span>混合检索 & 客服健康度</span>
              <Search className="h-4 w-4 text-emerald-600" />
            </div>
            <div className="text-2xl font-bold">
              {metrics ? `Top1: ${metrics.avg_rerank_top1_score}` : "--"}
            </div>
            <div className="text-xs text-muted-foreground space-y-0.5">
              <div>
                空召回率：{Math.round((metrics?.retrieval_zero_hit_rate ?? 0) * 100)}%（共{" "}
                {metrics?.retrieval_total_queries ?? 0} 次检索）
              </div>
              <div>
                转人工率：{Math.round((metrics?.escalation_rate ?? 0) * 100)}% / Redis命中{" "}
                {Math.round((metrics?.redis_cache_hit_rate ?? 0) * 100)}%
              </div>
            </div>
          </div>
        </div>

        {/* Per-Tool Metrics Table */}
        {metrics && metrics.tool_stats.length > 0 && (
          <div className="rounded-xl border bg-card p-5 shadow-sm space-y-3">
            <h2 className="text-sm font-semibold flex items-center gap-2">
              <Wrench className="h-4 w-4 text-primary" />
              各工具调用耗时与成功率分布
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="py-2 pr-4">工具名称</th>
                    <th className="py-2 px-3">调用次数</th>
                    <th className="py-2 px-3">成功（含缓存命中）</th>
                    <th className="py-2 px-3">Redis缓存命中</th>
                    <th className="py-2 px-3">超时</th>
                    <th className="py-2 px-3">异常/校验失败</th>
                    <th className="py-2 pl-3 text-right">平均耗时</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {metrics.tool_stats.map((st) => (
                    <tr key={st.tool_name}>
                      <td className="py-2 pr-4 font-mono font-medium">{st.tool_name}</td>
                      <td className="py-2 px-3">{st.calls}</td>
                      <td className="py-2 px-3 text-emerald-700 font-medium">{st.success_count}</td>
                      <td className="py-2 px-3">{st.cache_hit_count}</td>
                      <td className="py-2 px-3 text-amber-700">{st.timeout_count}</td>
                      <td className="py-2 px-3 text-destructive">{st.error_count}</td>
                      <td className="py-2 pl-3 text-right font-mono">{st.avg_latency_ms} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Trace List & Waterfall Inspector ("慢在哪、错在哪") */}
        <div className="grid gap-4 lg:grid-cols-12">
          {/* Left: Recent Traces List */}
          <div className="lg:col-span-5 rounded-xl border bg-card p-4 shadow-sm flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold flex items-center gap-1.5">
                <Activity className="h-4 w-4 text-primary" />
                近期对话 Trace 列表 ({traces.length})
              </span>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="rounded-md border bg-background px-2 py-1 text-xs"
              >
                <option value="">全部状态</option>
                <option value="ok">正常 (ok)</option>
                <option value="escalated">转人工 (escalated)</option>
                <option value="error">异常 (error)</option>
              </select>
            </div>

            {traces.length === 0 ? (
              <div className="rounded-lg border border-dashed p-8 text-center text-xs text-muted-foreground space-y-2">
                <div>暂无对话 Trace 记录</div>
                <div>
                  前往{" "}
                  <Link href="/dashboard/chat" className="text-primary underline">
                    客服对话
                  </Link>{" "}
                  发送一条消息即可在此实时看到全链路瀑布流与耗时拆解。
                </div>
              </div>
            ) : (
              <div className="space-y-2 max-h-[560px] overflow-y-auto pr-1">
                {traces.map((tr) => {
                  const active = selectedTrace?.trace_id === tr.trace_id;
                  return (
                    <button
                      key={tr.trace_id}
                      onClick={() => setSelectedTrace(tr)}
                      className={`w-full text-left rounded-lg border p-3 transition-colors space-y-1.5 ${
                        active
                          ? "border-primary bg-primary/5"
                          : "bg-background hover:bg-accent/50"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium text-xs line-clamp-1">
                          {tr.user_message}
                        </span>
                        <span
                          className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            tr.status === "error"
                              ? "bg-destructive/15 text-destructive"
                              : tr.status === "escalated"
                              ? "bg-amber-500/15 text-amber-800"
                              : "bg-emerald-500/15 text-emerald-700"
                          }`}
                        >
                          {tr.status}
                        </span>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                        <span className="font-mono">{tr.total_latency_ms}ms</span>
                        <span>·</span>
                        <span>{tr.total_tokens} tok</span>
                        <span>·</span>
                        <span>{tr.llm_tps} tok/s</span>
                        <span>·</span>
                        <span>工具 {tr.tool_call_count} 次</span>
                      </div>

                      {tr.slowest_span_name && (
                        <div className="text-[11px] text-amber-700 flex items-center gap-1">
                          <Clock className="h-3 w-3 shrink-0" />
                          <span className="truncate">
                            最慢环节：{tr.slowest_span_name} ({tr.slowest_span_ms}ms)
                          </span>
                        </div>
                      )}
                      {tr.error_message && (
                        <div className="text-[11px] text-destructive flex items-center gap-1">
                          <AlertCircle className="h-3 w-3 shrink-0" />
                          <span className="truncate">错误：{tr.error_message}</span>
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right: Nested Span Waterfall Inspector */}
          <div className="lg:col-span-7 rounded-xl border bg-card p-5 shadow-sm space-y-4">
            {!selectedTrace ? (
              <div className="h-full flex items-center justify-center text-sm text-muted-foreground py-16">
                点击左侧任意 Trace 查看全链路 Span 瀑布流（定位慢在哪、错在哪）
              </div>
            ) : (
              <>
                <div className="border-b pb-3 space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-sm font-semibold">
                      Trace 详情{" "}
                      <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono">
                        {selectedTrace.trace_id}
                      </code>
                    </div>
                    <div className="text-xs text-muted-foreground font-mono">
                      session={selectedTrace.session_id} | user={selectedTrace.user_id}
                    </div>
                  </div>
                  <div className="text-xs bg-muted/50 rounded-lg p-2.5 space-y-1">
                    <div>
                      <span className="font-medium">用户提问：</span>
                      {selectedTrace.user_message}
                    </div>
                    <div className="line-clamp-2 text-muted-foreground">
                      <span className="font-medium text-foreground">最终回复：</span>
                      {selectedTrace.final_reply}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="rounded-md bg-secondary px-2.5 py-1">
                      总耗时：<strong>{selectedTrace.total_latency_ms} ms</strong>
                    </span>
                    <span className="rounded-md bg-secondary px-2.5 py-1">
                      Token：<strong>{selectedTrace.total_tokens}</strong> (入{" "}
                      {selectedTrace.prompt_tokens} / 出 {selectedTrace.completion_tokens})
                    </span>
                    <span className="rounded-md bg-secondary px-2.5 py-1">
                      LLM 生成 TPS：<strong>{selectedTrace.llm_tps} tok/s</strong>
                    </span>
                    {selectedTrace.slowest_span_name && (
                      <span className="rounded-md bg-amber-500/15 text-amber-800 px-2.5 py-1 font-medium">
                        最慢瓶颈：{selectedTrace.slowest_span_name} ({selectedTrace.slowest_span_ms}
                        ms)
                      </span>
                    )}
                  </div>
                </div>

                {/* Span Waterfall List */}
                <div className="space-y-2.5 max-h-[460px] overflow-y-auto pr-1">
                  {selectedTrace.spans.map((sp) => {
                    const pct = Math.min(
                      100,
                      Math.max(
                        3,
                        Math.round(
                          (sp.latency_ms / Math.max(1, selectedTrace.total_latency_ms)) * 100
                        )
                      )
                    );
                    const isSlowest =
                      `${sp.span_type}:${sp.name}` === selectedTrace.slowest_span_name &&
                      sp.latency_ms > 0;
                    const isErr = sp.status === "error" || sp.status === "timeout";
                    const isSubRetrieval = sp.span_type.startsWith("retrieval_");

                    return (
                      <div
                        key={sp.span_id}
                        className={`rounded-lg border p-3 text-xs space-y-2 ${
                          isSubRetrieval ? "ml-6 bg-muted/20" : "bg-background"
                        } ${
                          isErr
                            ? "border-destructive/50 bg-destructive/5"
                            : isSlowest
                            ? "border-amber-500/50"
                            : ""
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="rounded bg-secondary px-2 py-0.5 text-[10px] font-medium">
                              {SPAN_TYPE_LABELS[sp.span_type] || sp.span_type}
                            </span>
                            <span className="font-mono font-semibold">{sp.name}</span>
                            {isSlowest && (
                              <span className="rounded bg-amber-500/15 text-amber-800 px-1.5 py-0.5 text-[10px] font-medium">
                                耗时最高
                              </span>
                            )}
                            {sp.status === "cache_hit" && (
                              <span className="rounded bg-emerald-500/15 text-emerald-700 px-1.5 py-0.5 text-[10px]">
                                缓存命中
                              </span>
                            )}
                          </div>
                          <span className="font-mono font-medium">
                            {sp.latency_ms} ms ({pct}%)
                          </span>
                        </div>

                        {/* Latency percentage bar */}
                        <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              isErr
                                ? "bg-destructive"
                                : isSlowest
                                ? "bg-amber-500"
                                : "bg-primary"
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>

                        {/* LLM Token / TPS metadata */}
                        {sp.span_type === "llm" && (
                          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
                            <span className="inline-flex items-center gap-1">
                              <Cpu className="h-3 w-3" />
                              Tokens: {String(sp.metadata.total_tokens ?? 0)} (入{" "}
                              {String(sp.metadata.prompt_tokens ?? 0)} / 出{" "}
                              {String(sp.metadata.completion_tokens ?? 0)})
                            </span>
                            <span>生成速度: {String(sp.metadata.llm_tps ?? 0)} tok/s</span>
                          </div>
                        )}

                        {/* Input / Output concise summary */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] text-muted-foreground">
                          {Object.keys(sp.input_summary).length > 0 && (
                            <div className="rounded bg-muted/50 px-2 py-1 font-mono truncate">
                              入参: {JSON.stringify(sp.input_summary)}
                            </div>
                          )}
                          {Object.keys(sp.output_summary).length > 0 && (
                            <div className="rounded bg-muted/50 px-2 py-1 font-mono truncate">
                              输出: {JSON.stringify(sp.output_summary)}
                            </div>
                          )}
                        </div>

                        {sp.error_message && (
                          <div className="rounded bg-destructive/10 px-2.5 py-1.5 text-destructive font-medium">
                            错误详情：{sp.error_message}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
