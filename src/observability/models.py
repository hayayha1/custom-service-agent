"""Data models for full-chain Trace/Span monitoring, tool metrics, token usage, and TPS aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class SpanRecord:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    span_type: str  # graph_node | llm | tool | retrieval_rewrite | retrieval_dense | retrieval_sparse | retrieval_rrf | retrieval_rerank
    name: str
    status: str  # ok | cache_hit | warning | error | timeout
    latency_ms: float
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    session_id: str
    user_id: str
    user_message: str
    final_reply: str = ""
    status: str = "ok"  # ok | escalated | error
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_generation_ms: float = 0.0
    llm_tps: float = 0.0  # Completion tokens per second (Tokens/s)
    llm_call_count: int = 0
    tool_call_count: int = 0
    tool_error_count: int = 0
    needs_human: bool = False
    escalation_reason: str | None = None
    redis_cache_hit: bool = False
    zero_hit: bool = False
    rerank_top1_score: float | None = None
    slowest_span_name: str = ""
    slowest_span_ms: float = 0.0
    error_message: str | None = None
    spans: list[SpanRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolMetricStat:
    tool_name: str
    calls: int = 0
    success_count: int = 0
    cache_hit_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0


@dataclass(slots=True)
class MetricsSummary:
    window_seconds: int
    total_requests: int
    success_requests: int
    error_requests: int
    escalated_requests: int
    request_tps: float  # System request throughput (Req/s in active window)
    peak_minute_tps: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_llm_tps: float  # LLM generation speed (Tokens/s)
    tool_total_calls: int
    tool_success_rate: float
    tool_timeout_rate: float
    tool_error_rate: float
    tool_stats: list[ToolMetricStat]
    retrieval_total_queries: int
    retrieval_zero_hit_rate: float
    avg_rerank_top1_score: float
    escalation_rate: float
    redis_cache_hit_rate: float
    langfuse_enabled: bool
    langfuse_connected: bool
    langfuse_host: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
