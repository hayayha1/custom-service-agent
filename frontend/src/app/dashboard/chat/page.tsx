"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  Archive,
  Bot,
  Database,
  Plus,
  RefreshCw,
  Send,
  User,
  Wrench,
} from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { api, type SessionMemoryResponse } from "@/lib/api";

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCallsUsed?: number;
  redisCacheHit?: boolean;
  needsHuman?: boolean;
  escalationReason?: string | null;
}

const QUICK_PROMPTS = [
  "帮我查下订单 20261119001 的状态",
  "7天无理由退换货运费怎么算？",
  "我买的耳机左耳没声音了，帮我建个售后工单",
  "以后快递都帮我发顺丰，我想转人工客服",
];

function generateSessionId() {
  return `web-${Math.random().toString(36).slice(2, 10)}`;
}

export default function ChatPage() {
  const [userId, setUserId] = useState("user_001");
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [archiving, setArchiving] = useState(false);
  const [memory, setMemory] = useState<SessionMemoryResponse | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const refreshMemory = async (sid: string) => {
    try {
      const mem = await api.getSessionMemory(sid);
      setMemory(mem);
    } catch {
      // ignore initial memory fetch error
    }
  };

  const handleSend = async (textOverride?: string) => {
    const content = (textOverride ?? input).trim();
    if (!content || sending) return;

    setInput("");
    setError(null);
    setNotice(null);

    const userTurn: ChatTurn = {
      id: `u-${Date.now()}`,
      role: "user",
      content,
    };
    setTurns((prev) => [...prev, userTurn]);
    setSending(true);

    try {
      const res = await api.chat({
        user_id: userId.trim() || "user_001",
        session_id: sessionId,
        message: content,
        request_id: `req-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      });
      setTurns((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: res.reply,
          toolCallsUsed: res.tool_calls_used,
          redisCacheHit: res.redis_cache_hit,
          needsHuman: res.needs_human,
          escalationReason: res.escalation_reason,
        },
      ]);
      await refreshMemory(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "发送失败，请检查后端服务与模型配置");
    } finally {
      setSending(false);
    }
  };

  const handleNewSession = () => {
    const nextId = generateSessionId();
    setSessionId(nextId);
    setTurns([]);
    setMemory(null);
    setNotice(`已切换至新会话 ${nextId}（同一 user_id 可测试跨会话 pgvector 长期记忆召回）`);
  };

  const handleArchiveSession = async () => {
    setArchiving(true);
    setError(null);
    try {
      const res = await api.archiveSession(sessionId, userId.trim() || "user_001");
      setNotice(
        res.archived
          ? `会话 ${sessionId} 已沉淀至 pgvector 长期记忆`
          : `会话 ${sessionId} 无实质业务信息，已自动跳过长期记忆写入`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "归档长期记忆失败");
    } finally {
      setArchiving(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col gap-4 h-[calc(100vh-8rem)]">
        {/* Top Session Controls */}
        <div className="rounded-xl border bg-card px-4 py-3 shadow-sm flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">用户 ID:</span>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="w-28 rounded-md border bg-background px-2.5 py-1 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">会话 ID:</span>
              <code className="rounded bg-muted px-2 py-1 text-xs">{sessionId}</code>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => void handleArchiveSession()}
              disabled={archiving || turns.length === 0}
              className="inline-flex items-center rounded-lg border bg-card px-3 py-1.5 text-xs font-medium hover:bg-accent disabled:opacity-50"
            >
              <Archive className="mr-1.5 h-3.5 w-3.5" />
              {archiving ? "归档中..." : "归档至长期记忆"}
            </button>
            <button
              onClick={handleNewSession}
              className="inline-flex items-center rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
            >
              <Plus className="mr-1.5 h-3.5 w-3.5" />
              新建会话
            </button>
          </div>
        </div>

        {notice && (
          <div className="rounded-lg border bg-muted/60 px-4 py-2 text-xs text-foreground">
            {notice}
          </div>
        )}
        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        {/* Main Chat + Memory Inspector Grid */}
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Chat Panel */}
          <div className="lg:col-span-8 flex flex-col rounded-xl border bg-card shadow-sm min-h-0">
            <div className="flex-1 overflow-y-auto p-5 space-y-5">
              {turns.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center space-y-5 py-8">
                  <div className="rounded-full bg-primary/10 p-4">
                    <Bot className="h-8 w-8 text-primary" />
                  </div>
                  <div className="space-y-1 max-w-md">
                    <h2 className="text-lg font-semibold">垂直客服 Agent 对话调试</h2>
                    <p className="text-sm text-muted-foreground">
                      支持多轮记忆、活跃业务槽位追踪、知识库检索、订单查询、幂等建单与转人工兜底。
                    </p>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2 max-w-xl">
                    {QUICK_PROMPTS.map((q) => (
                      <button
                        key={q}
                        onClick={() => void handleSend(q)}
                        className="rounded-full border bg-background px-3.5 py-1.5 text-xs text-muted-foreground hover:border-primary hover:text-foreground transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                turns.map((turn) => (
                  <div
                    key={turn.id}
                    className={`flex gap-3 ${turn.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {turn.role === "assistant" && (
                      <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                        <Bot className="h-4 w-4 text-primary" />
                      </div>
                    )}
                    <div
                      className={`max-w-[82%] space-y-2 ${
                        turn.role === "user" ? "items-end" : "items-start"
                      }`}
                    >
                      <div
                        className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                          turn.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted/60 text-foreground border"
                        }`}
                      >
                        {turn.role === "assistant" ? (
                          <div className="prose prose-sm max-w-none dark:prose-invert">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {turn.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <div className="whitespace-pre-wrap">{turn.content}</div>
                        )}
                      </div>

                      {turn.role === "assistant" && (
                        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground px-1">
                          {typeof turn.toolCallsUsed === "number" && (
                            <span className="inline-flex items-center rounded bg-secondary px-2 py-0.5">
                              <Wrench className="mr-1 h-3 w-3" />
                              工具调用 {turn.toolCallsUsed} 次
                            </span>
                          )}
                          {turn.redisCacheHit && (
                            <span className="inline-flex items-center rounded bg-emerald-500/10 text-emerald-700 px-2 py-0.5">
                              Redis 命中
                            </span>
                          )}
                          {turn.needsHuman && (
                            <span className="inline-flex items-center rounded bg-amber-500/15 text-amber-800 px-2 py-0.5 font-medium">
                              <AlertTriangle className="mr-1 h-3 w-3" />
                              已触发转人工{turn.escalationReason ? `：${turn.escalationReason}` : ""}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    {turn.role === "user" && (
                      <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center shrink-0 mt-0.5">
                        <User className="h-4 w-4 text-secondary-foreground" />
                      </div>
                    )}
                  </div>
                ))
              )}
              {sending && (
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary animate-pulse" />
                  </div>
                  正在运行 Agent Loop 推理与工具调用...
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Input Form */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void handleSend();
              }}
              className="border-t p-3 flex items-center gap-2"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={sending}
                placeholder="输入客服咨询问题，按 Enter 发送..."
                className="flex-1 rounded-lg border bg-background px-3.5 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={sending || !input.trim()}
                className="inline-flex items-center justify-center rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Send className="h-4 w-4" />
              </button>
            </form>
          </div>

          {/* Right-side L1 Working Memory & Slots Inspector */}
          <div className="lg:col-span-4 rounded-xl border bg-card p-5 shadow-sm flex flex-col gap-4 overflow-y-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-semibold text-sm">
                <Database className="h-4 w-4 text-primary" />
                实时记忆与活跃槽位 (L1)
              </div>
              <button
                onClick={() => void refreshMemory(sessionId)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-accent"
                title="刷新 Redis 缓存状态"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="rounded-lg bg-muted/50 px-3.5 py-2.5 text-xs flex items-center justify-between">
              <span className="text-muted-foreground">Redis 短期缓存</span>
              <span className="font-medium">
                {memory ? (memory.redis_cached ? "已缓存" : "未命中 / 未创建") : "尚未对话"}
              </span>
            </div>

            <div className="space-y-3">
              <div className="text-xs font-medium text-muted-foreground">
                活跃业务槽位（独立 Hash，支持多订单/多工单跨轮次保留）
              </div>
              {memory && Object.keys(memory.slots).length > 0 ? (
                <div className="space-y-3">
                  {/* Multiple Active Orders */}
                  {Boolean(memory.slots.active_orders) &&
                    typeof memory.slots.active_orders === "object" &&
                    Object.keys(memory.slots.active_orders as Record<string, unknown>).length > 0 && (
                      <div className="space-y-1.5">
                        <div className="text-[11px] font-medium text-primary">
                          进行中的订单（
                          {Object.keys(memory.slots.active_orders as Record<string, unknown>).length}{" "}
                          笔）
                        </div>
                        {Object.entries(
                          memory.slots.active_orders as Record<string, Record<string, string>>
                        ).map(([oid, info]) => (
                          <div
                            key={oid}
                            className="rounded-lg border bg-background p-2.5 text-xs space-y-1"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-mono font-semibold">{oid}</span>
                              {info?.status && (
                                <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px]">
                                  {info.status}
                                </span>
                              )}
                            </div>
                            {info?.product_name && (
                              <div className="text-muted-foreground">{info.product_name}</div>
                            )}
                            {info?.tracking_no && (
                              <div className="font-mono text-[11px] text-muted-foreground">
                                运单: {info.tracking_no}
                              </div>
                            )}
                            {info?.ticket_id && (
                              <div className="font-mono text-[11px] text-emerald-700">
                                关联工单: {info.ticket_id}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                  {/* Multiple Active Tickets */}
                  {Boolean(memory.slots.active_tickets) &&
                    typeof memory.slots.active_tickets === "object" &&
                    Object.keys(memory.slots.active_tickets as Record<string, unknown>).length > 0 && (
                      <div className="space-y-1.5">
                        <div className="text-[11px] font-medium text-emerald-700">
                          已创建工单（
                          {Object.keys(memory.slots.active_tickets as Record<string, unknown>).length}{" "}
                          个）
                        </div>
                        {Object.entries(
                          memory.slots.active_tickets as Record<string, Record<string, string>>
                        ).map(([tid, info]) => (
                          <div
                            key={tid}
                            className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-2.5 text-xs space-y-0.5"
                          >
                            <div className="font-mono font-semibold text-emerald-800">{tid}</div>
                            {info?.order_id && (
                              <div className="text-[11px] text-muted-foreground">
                                订单: {info.order_id}
                              </div>
                            )}
                            {info?.issue_summary && (
                              <div className="text-[11px] text-muted-foreground line-clamp-2">
                                {info.issue_summary}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                  {/* Remaining Scalar Slots */}
                  {Object.entries(memory.slots)
                    .filter(
                      ([k]) =>
                        ![
                          "active_orders",
                          "active_tickets",
                          "order_ids",
                          "tracking_nos",
                        ].includes(k)
                    )
                    .map(([k, v]) => (
                      <div
                        key={k}
                        className="flex items-center justify-between gap-2 rounded-md border bg-background px-3 py-2 text-xs"
                      >
                        <span className="font-mono text-muted-foreground shrink-0">{k}</span>
                        <span className="font-medium text-right break-all">
                          {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </span>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="rounded-lg border border-dashed p-4 text-center text-xs text-muted-foreground">
                  暂无提取到的业务槽位（如 order_id、phone、product_name）
                </div>
              )}
            </div>

            <div className="space-y-2 pt-2 border-t">
              <div className="text-xs font-medium text-muted-foreground">
                Redis 滑动窗口消息数：{memory ? memory.messages.length : 0} 条
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                对话消息按最近 N 轮滑动裁剪，而上方“活跃业务槽位”独立存放，即使超过 N 轮也不会被裁掉。
              </p>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
