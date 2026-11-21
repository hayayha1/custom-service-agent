export interface LLMConfigView {
  base_url: string;
  model: string;
  temperature: number;
  memory_model: string;
  api_key_masked: string;
  has_api_key: boolean;
}

export interface EmbeddingConfigView {
  base_url: string;
  model: string;
  dim: number;
  api_key_masked: string;
  has_api_key: boolean;
}

export interface RerankerConfigView {
  enabled: boolean;
  base_url: string;
  model: string;
  top_n: number;
  min_score: number;
  api_key_masked: string;
  has_api_key: boolean;
}

export interface LangfuseConfigView {
  enabled: boolean;
  host: string;
  public_key_masked: string;
  secret_key_masked: string;
  has_keys: boolean;
}

export interface ModelSettingsResponse {
  llm: LLMConfigView;
  embedding: EmbeddingConfigView;
  reranker: RerankerConfigView;
  langfuse?: LangfuseConfigView;
}

export interface ModelSettingsUpdatePayload {
  llm: {
    base_url: string;
    model: string;
    temperature: number;
    memory_model: string;
    api_key?: string;
  };
  embedding: {
    base_url: string;
    model: string;
    dim: number;
    api_key?: string;
  };
  reranker: {
    enabled: boolean;
    base_url: string;
    model: string;
    top_n: number;
    min_score: number;
    api_key?: string;
  };
  langfuse?: {
    enabled: boolean;
    host: string;
    public_key?: string;
    secret_key?: string;
  };
}

export interface SystemOverviewResponse {
  status: string;
  redis: boolean;
  postgres?: boolean;
  storage_mode?: string;
  kb_document_count: number;
  registered_tools: string[];
  models: ModelSettingsResponse;
  metrics_brief?: {
    total_requests: number;
    request_tps: number;
    avg_llm_tps: number;
    total_tokens: number;
    p95_latency_ms: number;
    langfuse_enabled: boolean;
    langfuse_connected: boolean;
  };
}

export interface ChatResponse {
  reply: string;
  user_id: string;
  session_id: string;
  needs_human: boolean;
  escalation_reason: string | null;
  slots: Record<string, unknown>;
  tool_calls_used: number;
  redis_cache_hit: boolean;
  deduplicated: boolean;
  trace_id?: string | null;
}

export interface SessionMemoryResponse {
  session_id: string;
  redis_cached: boolean;
  slots: Record<string, unknown>;
  messages: Array<{ type: string; content: string }>;
}

export interface KBDocumentItem {
  doc_key: string;
  category: string;
  title: string;
  content: string;
  metadata?: {
    chunk_count?: number;
    content_hash?: string;
    [key: string]: unknown;
  };
}

export interface KBIngestResponse {
  ok: boolean;
  doc_key: string;
  category: string;
  title: string;
  content_hash: string;
  chunk_count: number;
  pii_redacted_count: number;
  noise_removed_count: number;
  skipped_unchanged: boolean;
}

export interface KBSearchMatch {
  chunk_key?: string;
  doc_key: string;
  category: string;
  title: string;
  section_path?: string;
  content: string;
  score: number;
  dense_score?: number;
  sparse_score?: number;
  rrf_score?: number;
  rerank_score?: number | null;
}

export interface KBSearchResponse {
  query: string;
  rewritten_query?: string;
  rewrite_applied?: boolean;
  dense_count?: number;
  sparse_count?: number;
  fused_count?: number;
  reranked?: boolean;
  zero_hit?: boolean;
  timings?: {
    rewrite_ms: number;
    dense_ms: number;
    sparse_ms: number;
    rrf_ms: number;
    rerank_ms: number;
    total_ms: number;
  };
  matches: KBSearchMatch[];
}

export interface ToolMetricStat {
  tool_name: string;
  calls: number;
  success_count: number;
  cache_hit_count: number;
  timeout_count: number;
  error_count: number;
  avg_latency_ms: number;
}

export interface MetricsSummaryResponse {
  window_seconds: number;
  total_requests: number;
  success_requests: number;
  error_requests: number;
  escalated_requests: number;
  request_tps: number;
  peak_minute_tps: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_latency_ms: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  avg_llm_tps: number;
  tool_total_calls: number;
  tool_success_rate: number;
  tool_timeout_rate: number;
  tool_error_rate: number;
  tool_stats: ToolMetricStat[];
  retrieval_total_queries: number;
  retrieval_zero_hit_rate: number;
  avg_rerank_top1_score: number;
  escalation_rate: number;
  redis_cache_hit_rate: number;
  langfuse_enabled: boolean;
  langfuse_connected: boolean;
  langfuse_host: string;
}

export interface SpanRecordItem {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  span_type: string;
  name: string;
  status: string;
  latency_ms: number;
  input_summary: Record<string, unknown>;
  output_summary: Record<string, unknown>;
  metadata: Record<string, unknown>;
  error_message: string | null;
  started_at: string;
}

export interface TraceRecordItem {
  trace_id: string;
  session_id: string;
  user_id: string;
  user_message: string;
  final_reply: string;
  status: string;
  total_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  llm_generation_ms: number;
  llm_tps: number;
  llm_call_count: number;
  tool_call_count: number;
  tool_error_count: number;
  needs_human: boolean;
  escalation_reason: string | null;
  redis_cache_hit: boolean;
  zero_hit: boolean;
  rerank_top1_score: number | null;
  slowest_span_name: string;
  slowest_span_ms: number;
  error_message: string | null;
  spans: SpanRecordItem[];
  created_at: string;
}

const API_BASE = "/api";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // ignore JSON parse error
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  getOverview: () => requestJson<SystemOverviewResponse>("/system/overview"),

  chat: (payload: {
    user_id: string;
    session_id: string;
    message: string;
    request_id?: string;
  }) =>
    requestJson<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getSessionMemory: (sessionId: string) =>
    requestJson<SessionMemoryResponse>(`/sessions/${encodeURIComponent(sessionId)}/memory`),

  archiveSession: (sessionId: string, userId: string) =>
    requestJson<{ session_id: string; user_id: string; archived: boolean }>(
      `/sessions/${encodeURIComponent(sessionId)}/archive`,
      {
        method: "POST",
        body: JSON.stringify({ user_id: userId }),
      }
    ),

  listKBDocuments: (category?: string) => {
    const qs = category ? `?category=${encodeURIComponent(category)}` : "";
    return requestJson<{ documents: KBDocumentItem[] }>(`/kb/documents${qs}`);
  },

  upsertKBDocument: (payload: {
    doc_key?: string;
    category: string;
    title: string;
    content: string;
    chunk_mode?: "auto" | "faq" | "sop";
    redact_pii?: boolean;
    chunk_size?: number;
    chunk_overlap?: number;
    force?: boolean;
  }) =>
    requestJson<KBIngestResponse>("/kb/ingest", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  deleteKBDocument: (docKey: string) =>
    requestJson<{ ok: boolean; doc_key: string }>(`/kb/documents/${encodeURIComponent(docKey)}`, {
      method: "DELETE",
    }),

  searchKB: (payload: {
    query: string;
    category?: string | null;
    top_k?: number;
    slots?: Record<string, unknown>;
  }) =>
    requestJson<KBSearchResponse>("/kb/search", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getMetricsSummary: (windowSeconds = 3600) =>
    requestJson<MetricsSummaryResponse>(`/metrics/summary?window_seconds=${windowSeconds}`),

  listTraces: (params?: { session_id?: string; status?: string; limit?: number }) => {
    const search = new URLSearchParams();
    if (params?.session_id) search.set("session_id", params.session_id);
    if (params?.status) search.set("status", params.status);
    if (params?.limit) search.set("limit", String(params.limit));
    const qs = search.toString() ? `?${search.toString()}` : "";
    return requestJson<{ traces: TraceRecordItem[] }>(`/traces${qs}`);
  },

  getTraceDetail: (traceId: string) =>
    requestJson<TraceRecordItem>(`/traces/${encodeURIComponent(traceId)}`),

  getModelSettings: () => requestJson<ModelSettingsResponse>("/settings/models"),

  updateModelSettings: (payload: ModelSettingsUpdatePayload) =>
    requestJson<ModelSettingsResponse>("/settings/models", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
};
