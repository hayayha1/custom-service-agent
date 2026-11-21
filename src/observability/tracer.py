"""ContextVar-propagated full-chain tracer with Langfuse SDK export and rolling KPI aggregator."""

from __future__ import annotations

import contextvars
import logging
import math
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from src.observability.models import (
    MetricsSummary,
    SpanRecord,
    ToolMetricStat,
    TraceRecord,
)

logger = logging.getLogger(__name__)

_current_trace_var: contextvars.ContextVar[ActiveTraceContext | None] = contextvars.ContextVar(
    "cs_active_trace", default=None
)
_current_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cs_active_span_id", default=None
)


def get_current_trace() -> ActiveTraceContext | None:
    return _current_trace_var.get()


class ActiveSpanHandle:
    """Mutable handle yielded inside `start_span(...)` so callers can attach output/error metadata."""

    def __init__(
        self,
        *,
        span_id: str,
        trace_id: str,
        parent_span_id: str | None,
        span_type: str,
        name: str,
        input_summary: dict[str, Any] | None = None,
    ) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.span_type = span_type
        self.name = name
        self.status = "ok"
        self.input_summary: dict[str, Any] = dict(input_summary or {})
        self.output_summary: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.error_message: str | None = None
        self.started_at = datetime.now(UTC).isoformat()
        self._start_perf = time.perf_counter()

    def set_output(self, summary: dict[str, Any]) -> None:
        self.output_summary.update(summary)

    def set_metadata(self, meta: dict[str, Any]) -> None:
        self.metadata.update(meta)

    def mark_error(self, error_message: str, status: str = "error") -> None:
        self.status = status
        self.error_message = error_message

    def finalize(self) -> SpanRecord:
        latency_ms = round((time.perf_counter() - self._start_perf) * 1000.0, 2)
        return SpanRecord(
            span_id=self.span_id,
            trace_id=self.trace_id,
            parent_span_id=self.parent_span_id,
            span_type=self.span_type,
            name=self.name,
            status=self.status,
            latency_ms=latency_ms,
            input_summary=self.input_summary,
            output_summary=self.output_summary,
            metadata=self.metadata,
            error_message=self.error_message,
            started_at=self.started_at,
        )


class ActiveTraceContext:
    """Collects nested spans, LLM token/TPS stats, and retrieval metrics for a single turn."""

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        trace_id: str | None = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.session_id = session_id
        self.user_id = user_id
        self.user_message = user_message
        self.created_at = datetime.now(UTC).isoformat()
        self._start_perf = time.perf_counter()
        self.spans: list[SpanRecord] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_generation_ms = 0.0
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.tool_error_count = 0
        self.zero_hit = False
        self.rerank_top1_score: float | None = None
        self.error_message: str | None = None

    @property
    def current_parent_span_id(self) -> str | None:
        return _current_span_id_var.get()

    @contextmanager
    def start_span(
        self,
        *,
        span_type: str,
        name: str,
        input_summary: dict[str, Any] | None = None,
    ) -> Iterator[ActiveSpanHandle]:
        parent_id = _current_span_id_var.get()
        span_id = uuid.uuid4().hex[:16]
        handle = ActiveSpanHandle(
            span_id=span_id,
            trace_id=self.trace_id,
            parent_span_id=parent_id,
            span_type=span_type,
            name=name,
            input_summary=input_summary,
        )
        token = _current_span_id_var.set(span_id)
        try:
            yield handle
        except Exception as exc:
            handle.mark_error(f"{type(exc).__name__}: {exc}", status="error")
            raise
        finally:
            _current_span_id_var.reset(token)
            record = handle.finalize()
            self.spans.append(record)
            if record.span_type == "tool":
                self.tool_call_count += 1
                if record.status in ("error", "timeout"):
                    self.tool_error_count += 1

    def record_llm_generation(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        output_preview: str = "",
        error_message: str | None = None,
    ) -> None:
        self.llm_call_count += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.llm_generation_ms += latency_ms
        tps = (
            round(completion_tokens / (latency_ms / 1000.0), 2)
            if latency_ms > 0 and completion_tokens > 0
            else 0.0
        )
        self.spans.append(
            SpanRecord(
                span_id=uuid.uuid4().hex[:16],
                trace_id=self.trace_id,
                parent_span_id=_current_span_id_var.get(),
                span_type="llm",
                name=f"LLM ({model})",
                status="error" if error_message else "ok",
                latency_ms=round(latency_ms, 2),
                input_summary={"model": model},
                output_summary={"preview": output_preview[:160]},
                metadata={
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "llm_tps": tps,
                },
                error_message=error_message,
            )
        )

    def to_trace_record(
        self,
        *,
        final_reply: str,
        needs_human: bool,
        escalation_reason: str | None,
        redis_cache_hit: bool,
        error_message: str | None = None,
    ) -> TraceRecord:
        total_ms = round((time.perf_counter() - self._start_perf) * 1000.0, 2)
        llm_tps = (
            round(self.completion_tokens / (self.llm_generation_ms / 1000.0), 2)
            if self.llm_generation_ms > 0 and self.completion_tokens > 0
            else 0.0
        )
        err = error_message or self.error_message
        if err:
            status = "error"
        elif needs_human:
            status = "escalated"
        else:
            status = "ok"

        # Identify the bottleneck ("慢在哪") across leaf/operational spans
        slowest_name = ""
        slowest_ms = 0.0
        for sp in self.spans:
            if sp.latency_ms >= slowest_ms:
                slowest_ms = sp.latency_ms
                slowest_name = f"{sp.span_type}:{sp.name}"

        return TraceRecord(
            trace_id=self.trace_id,
            session_id=self.session_id,
            user_id=self.user_id,
            user_message=self.user_message,
            final_reply=final_reply,
            status=status,
            total_latency_ms=total_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            llm_generation_ms=round(self.llm_generation_ms, 2),
            llm_tps=llm_tps,
            llm_call_count=self.llm_call_count,
            tool_call_count=self.tool_call_count,
            tool_error_count=self.tool_error_count,
            needs_human=needs_human,
            escalation_reason=escalation_reason,
            redis_cache_hit=redis_cache_hit,
            zero_hit=self.zero_hit,
            rerank_top1_score=self.rerank_top1_score,
            slowest_span_name=slowest_name,
            slowest_span_ms=slowest_ms,
            error_message=err,
            spans=list(self.spans),
            created_at=self.created_at,
        )


@contextmanager
def trace_turn_context(
    *,
    session_id: str,
    user_id: str,
    user_message: str,
) -> Iterator[ActiveTraceContext]:
    ctx = ActiveTraceContext(
        session_id=session_id,
        user_id=user_id,
        user_message=user_message,
    )
    token = _current_trace_var.set(ctx)
    span_token = _current_span_id_var.set(None)
    try:
        yield ctx
    finally:
        _current_span_id_var.reset(span_token)
        _current_trace_var.reset(token)


@contextmanager
def start_observed_span(
    *,
    span_type: str,
    name: str,
    input_summary: dict[str, Any] | None = None,
) -> Iterator[ActiveSpanHandle | None]:
    """Convenience helper: records a span if inside an active turn trace, else no-op."""
    ctx = _current_trace_var.get()
    if ctx is None:
        yield None
        return
    with ctx.start_span(span_type=span_type, name=name, input_summary=input_summary) as handle:
        yield handle


class AgentObservabilityManager:
    """Manages Langfuse export AND in-memory rolling Trace/Span waterfall & TPS metrics."""

    def __init__(
        self,
        *,
        langfuse_enabled: bool = False,
        langfuse_public_key: str = "",
        langfuse_secret_key: str = "",
        langfuse_host: str = "https://cloud.langfuse.com",
        max_traces: int = 200,
    ) -> None:
        self._traces: deque[tuple[float, TraceRecord]] = deque(maxlen=max_traces)
        self.langfuse_enabled = False
        self.langfuse_host = langfuse_host
        self._langfuse_client: Any | None = None
        self.configure_langfuse(
            enabled=langfuse_enabled,
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host=langfuse_host,
        )

    def configure_langfuse(
        self,
        *,
        enabled: bool,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
    ) -> None:
        self.langfuse_host = host.strip() or "https://cloud.langfuse.com"
        self.langfuse_enabled = bool(enabled and public_key.strip() and secret_key.strip())
        self._langfuse_client = None
        if not self.langfuse_enabled:
            return
        try:
            from langfuse import Langfuse

            self._langfuse_client = Langfuse(
                public_key=public_key.strip(),
                secret_key=secret_key.strip(),
                host=self.langfuse_host,
            )
            logger.info("Langfuse observability enabled (host=%s)", self.langfuse_host)
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse client (falling back to local UI store): %s", exc)
            self._langfuse_client = None

    @property
    def langfuse_connected(self) -> bool:
        return self._langfuse_client is not None

    def record_trace(self, trace: TraceRecord) -> None:
        now_ts = time.time()
        self._traces.append((now_ts, trace))
        if self._langfuse_client is not None:
            self._export_to_langfuse(trace)

    def _export_to_langfuse(self, trace: TraceRecord) -> None:
        """Export completed Trace, nested Spans, Generations, and Scores to Langfuse."""
        client = self._langfuse_client
        if client is None:
            return
        try:
            # Compatible with both Langfuse v2/v3 trace API and v4/v5 start_span/create_score APIs
            if hasattr(client, "trace"):
                lf_trace = client.trace(
                    id=trace.trace_id,
                    name="customer_service_turn",
                    user_id=trace.user_id,
                    session_id=trace.session_id,
                    input=trace.user_message,
                    output=trace.final_reply,
                    metadata={
                        "status": trace.status,
                        "total_latency_ms": trace.total_latency_ms,
                        "slowest_span": trace.slowest_span_name,
                        "needs_human": trace.needs_human,
                        "redis_cache_hit": trace.redis_cache_hit,
                    },
                )
                for sp in trace.spans:
                    if sp.span_type == "llm":
                        lf_trace.generation(
                            name=sp.name,
                            model=str(sp.metadata.get("model") or "llm"),
                            input=sp.input_summary,
                            output=sp.output_summary,
                            usage={
                                "input": int(sp.metadata.get("prompt_tokens") or 0),
                                "output": int(sp.metadata.get("completion_tokens") or 0),
                                "total": int(sp.metadata.get("total_tokens") or 0),
                            },
                            metadata=sp.metadata,
                            level="ERROR" if sp.status == "error" else "DEFAULT",
                            status_message=sp.error_message,
                        )
                    else:
                        lf_trace.span(
                            name=f"{sp.span_type}:{sp.name}",
                            input=sp.input_summary,
                            output=sp.output_summary,
                            metadata={**sp.metadata, "latency_ms": sp.latency_ms, "status": sp.status},
                            level="ERROR" if sp.status in ("error", "timeout") else "DEFAULT",
                            status_message=sp.error_message,
                        )
                lf_trace.score(name="llm_tps", value=trace.llm_tps)
                if trace.rerank_top1_score is not None:
                    lf_trace.score(name="rerank_top1_score", value=trace.rerank_top1_score)
            elif hasattr(client, "start_span"):
                with client.start_as_current_span(
                    name="customer_service_turn",
                    input={"user_id": trace.user_id, "session_id": trace.session_id, "message": trace.user_message},
                    output={"reply": trace.final_reply},
                    metadata={
                        "trace_id": trace.trace_id,
                        "status": trace.status,
                        "total_latency_ms": trace.total_latency_ms,
                        "prompt_tokens": trace.prompt_tokens,
                        "completion_tokens": trace.completion_tokens,
                        "total_tokens": trace.total_tokens,
                        "llm_tps": trace.llm_tps,
                        "slowest_span": trace.slowest_span_name,
                    },
                ):
                    for sp in trace.spans:
                        with client.start_as_current_span(
                            name=f"{sp.span_type}:{sp.name}",
                            input=sp.input_summary,
                            output=sp.output_summary,
                            metadata={**sp.metadata, "latency_ms": sp.latency_ms, "status": sp.status},
                        ):
                            pass
        except Exception as exc:
            logger.debug("Non-fatal Langfuse export warning: %s", exc)

    def list_traces(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for _, tr in reversed(self._traces):
            if session_id and tr.session_id != session_id:
                continue
            if status and tr.status != status:
                continue
            items.append(tr.to_dict())
            if len(items) >= limit:
                break
        return items

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        for _, tr in reversed(self._traces):
            if tr.trace_id == trace_id:
                return tr.to_dict()
        return None

    def get_metrics_summary(self, window_seconds: int = 3600) -> MetricsSummary:
        now_ts = time.time()
        cutoff = now_ts - max(1, window_seconds)
        window_items = [(ts, tr) for ts, tr in self._traces if ts >= cutoff]
        traces = [tr for _, tr in window_items]
        total = len(traces)

        if total == 0:
            return MetricsSummary(
                window_seconds=window_seconds,
                total_requests=0,
                success_requests=0,
                error_requests=0,
                escalated_requests=0,
                request_tps=0.0,
                peak_minute_tps=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                avg_latency_ms=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                avg_llm_tps=0.0,
                tool_total_calls=0,
                tool_success_rate=1.0,
                tool_timeout_rate=0.0,
                tool_error_rate=0.0,
                tool_stats=[],
                retrieval_total_queries=0,
                retrieval_zero_hit_rate=0.0,
                avg_rerank_top1_score=0.0,
                escalation_rate=0.0,
                redis_cache_hit_rate=0.0,
                langfuse_enabled=self.langfuse_enabled,
                langfuse_connected=self.langfuse_connected,
                langfuse_host=self.langfuse_host,
            )

        success_cnt = sum(1 for t in traces if t.status == "ok")
        error_cnt = sum(1 for t in traces if t.status == "error")
        esc_cnt = sum(1 for t in traces if t.needs_human)
        cache_hits = sum(1 for t in traces if t.redis_cache_hit)

        latencies = sorted(t.total_latency_ms for t in traces)
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        avg_lat = round(sum(latencies) / total, 2)

        # Request TPS: active span of requests (or 60s window for recent burst)
        recent_60s = sum(1 for ts, _ in window_items if ts >= now_ts - 60.0)
        span_seconds = max(1.0, now_ts - window_items[0][0]) if len(window_items) > 1 else 1.0
        req_tps = round(total / min(float(window_seconds), max(span_seconds, 1.0)), 3)
        peak_min_tps = round(recent_60s / 60.0, 3)

        prompt_tok = sum(t.prompt_tokens for t in traces)
        comp_tok = sum(t.completion_tokens for t in traces)
        tot_tok = sum(t.total_tokens for t in traces)
        tps_values = [t.llm_tps for t in traces if t.llm_tps > 0]
        avg_llm_tps = round(sum(tps_values) / len(tps_values), 2) if tps_values else 0.0

        # Tool stats aggregation
        tool_map: dict[str, list[SpanRecord]] = {}
        tool_total = 0
        tool_ok = 0
        tool_timeout = 0
        tool_err = 0
        retrieval_queries = 0
        retrieval_zero_hits = 0
        rerank_scores: list[float] = []

        for tr in traces:
            if tr.rerank_top1_score is not None:
                rerank_scores.append(tr.rerank_top1_score)
            for sp in tr.spans:
                if sp.span_type == "tool":
                    tool_total += 1
                    tool_map.setdefault(sp.name, []).append(sp)
                    if sp.status in ("ok", "cache_hit", "success"):
                        tool_ok += 1
                    elif sp.status == "timeout":
                        tool_timeout += 1
                    else:
                        tool_err += 1
                elif sp.span_type == "retrieval_rerank":
                    retrieval_queries += 1
                    if sp.metadata.get("zero_hit"):
                        retrieval_zero_hits += 1

        tool_stats: list[ToolMetricStat] = []
        for tname, sps in sorted(tool_map.items()):
            cnt = len(sps)
            tool_stats.append(
                ToolMetricStat(
                    tool_name=tname,
                    calls=cnt,
                    success_count=sum(1 for s in sps if s.status in ("ok", "success", "cache_hit")),
                    cache_hit_count=sum(1 for s in sps if s.status == "cache_hit"),
                    timeout_count=sum(1 for s in sps if s.status == "timeout"),
                    error_count=sum(1 for s in sps if s.status not in ("ok", "success", "cache_hit", "timeout")),
                    avg_latency_ms=round(sum(s.latency_ms for s in sps) / cnt, 2),
                )
            )

        return MetricsSummary(
            window_seconds=window_seconds,
            total_requests=total,
            success_requests=success_cnt,
            error_requests=error_cnt,
            escalated_requests=esc_cnt,
            request_tps=req_tps,
            peak_minute_tps=peak_min_tps,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            avg_latency_ms=avg_lat,
            total_prompt_tokens=prompt_tok,
            total_completion_tokens=comp_tok,
            total_tokens=tot_tok,
            avg_llm_tps=avg_llm_tps,
            tool_total_calls=tool_total,
            tool_success_rate=round(tool_ok / tool_total, 4) if tool_total else 1.0,
            tool_timeout_rate=round(tool_timeout / tool_total, 4) if tool_total else 0.0,
            tool_error_rate=round(tool_err / tool_total, 4) if tool_total else 0.0,
            tool_stats=tool_stats,
            retrieval_total_queries=retrieval_queries,
            retrieval_zero_hit_rate=(
                round(retrieval_zero_hits / retrieval_queries, 4) if retrieval_queries else 0.0
            ),
            avg_rerank_top1_score=(
                round(sum(rerank_scores) / len(rerank_scores), 4) if rerank_scores else 0.0
            ),
            escalation_rate=round(esc_cnt / total, 4),
            redis_cache_hit_rate=round(cache_hits / total, 4),
            langfuse_enabled=self.langfuse_enabled,
            langfuse_connected=self.langfuse_connected,
            langfuse_host=self.langfuse_host,
        )


def _percentile(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(sorted_values[low], 2)
    weight = rank - low
    return round(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight, 2)
