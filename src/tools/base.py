"""Tool metadata, audit logging, Redis read-cache, and guarded async execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from src.observability.tracer import start_observed_span

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    name: str
    is_write: bool = False
    cacheable: bool = False
    timeout_seconds: float | None = None


@dataclass(slots=True)
class ToolAuditEntry:
    session_id: str
    user_id: str
    tool_name: str
    args: dict[str, Any]
    status: str
    latency_ms: float
    error_message: str | None = None


@dataclass(slots=True)
class ToolExecutionOutcome:
    content: str
    ok: bool
    is_write: bool
    cached: bool = False
    error_message: str | None = None


class ToolAuditLogger(Protocol):
    async def log(self, entry: ToolAuditEntry) -> None: ...


class PostgresToolAuditLogger:
    """Persists every tool invocation into `tool_audit_logs`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def log(self, entry: ToolAuditEntry) -> None:
        sql = """
            INSERT INTO tool_audit_logs (
                session_id, user_id, tool_name, args_json, status, latency_ms, error_message
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
        """
        async with self._pool.connection() as conn:
            await conn.execute(
                sql,
                (
                    entry.session_id,
                    entry.user_id,
                    entry.tool_name,
                    json.dumps(entry.args, ensure_ascii=False, default=str),
                    entry.status,
                    entry.latency_ms,
                    entry.error_message,
                ),
            )


class InMemoryToolAuditLogger:
    """In-memory audit logger for unit tests and local dry runs."""

    def __init__(self) -> None:
        self.entries: list[ToolAuditEntry] = []

    async def log(self, entry: ToolAuditEntry) -> None:
        self.entries.append(entry)


def _tool_cache_key(tool_name: str, args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"cs:tool_cache:{tool_name}:{digest}"


async def safe_execute_tool(
    tool: BaseTool,
    metadata: ToolMetadata,
    args: dict[str, Any],
    *,
    session_id: str,
    user_id: str,
    default_timeout: float = 10.0,
    redis_client: Any | None = None,
    cache_ttl_seconds: int = 60,
    audit_logger: ToolAuditLogger | None = None,
) -> ToolExecutionOutcome:
    """Execute a tool with timeout, optional read-cache, structured error capture, and audit logging."""
    start = time.perf_counter()
    timeout = metadata.timeout_seconds or default_timeout
    cache_key = _tool_cache_key(metadata.name, args)

    with start_observed_span(
        span_type="tool",
        name=metadata.name,
        input_summary={"args": args, "is_write": metadata.is_write},
    ) as tool_span:
        # 1. Optional short-TTL cache for read-only cacheable tools
        if (
            metadata.cacheable
            and not metadata.is_write
            and redis_client is not None
            and cache_ttl_seconds > 0
        ):
            try:
                cached_value = await redis_client.get(cache_key)
                if cached_value is not None:
                    latency_ms = (time.perf_counter() - start) * 1000
                    if tool_span is not None:
                        tool_span.status = "cache_hit"
                        tool_span.set_output({"cached": True, "preview": str(cached_value)[:160]})
                    if audit_logger:
                        await _safe_audit_log(
                            audit_logger,
                            ToolAuditEntry(
                                session_id=session_id,
                                user_id=user_id,
                                tool_name=metadata.name,
                                args=args,
                                status="cache_hit",
                                latency_ms=latency_ms,
                            ),
                        )
                    return ToolExecutionOutcome(
                        content=str(cached_value),
                        ok=True,
                        is_write=metadata.is_write,
                        cached=True,
                    )
            except Exception as exc:
                logger.warning("Tool cache lookup failed for %s: %s", metadata.name, exc)

        # 2. Guarded async invocation
        status = "success"
        error_message: str | None = None
        ok = True
        try:
            raw_result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            content = (
                raw_result
                if isinstance(raw_result, str)
                else json.dumps(raw_result, ensure_ascii=False, default=str)
            )
            # Detect tool-level semantic failure if JSON contains ok=false
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and parsed.get("ok") is False:
                    ok = False
                    status = "tool_error"
                    error_message = str(
                        parsed.get("message") or parsed.get("error") or "tool returned ok=false"
                    )
            except (ValueError, TypeError):
                pass
        except TimeoutError:
            ok = False
            status = "timeout"
            error_message = f"工具 {metadata.name} 执行超时（>{timeout}s），请稍后重试或转人工处理。"
            content = json.dumps(
                {"ok": False, "error_type": "timeout", "message": error_message},
                ensure_ascii=False,
            )
        except ValidationError as exc:
            ok = False
            status = "validation_error"
            error_message = f"工具 {metadata.name} 参数校验失败: {exc.errors()}"
            content = json.dumps(
                {"ok": False, "error_type": "validation_error", "message": error_message},
                ensure_ascii=False,
            )
        except Exception as exc:
            ok = False
            status = "exception"
            error_message = f"工具 {metadata.name} 调用异常: {type(exc).__name__}: {exc}"
            logger.warning("Tool execution exception (%s): %s", metadata.name, exc)
            content = json.dumps(
                {"ok": False, "error_type": "execution_error", "message": error_message},
                ensure_ascii=False,
            )

        if tool_span is not None:
            tool_span.set_output({"ok": ok, "preview": content[:160]})
            if not ok:
                tool_span.mark_error(
                    error_message or "tool execution failed",
                    status="timeout" if status == "timeout" else "error",
                )

        # 3. Populate read-only cache on success
        if (
            ok
            and metadata.cacheable
            and not metadata.is_write
            and redis_client is not None
            and cache_ttl_seconds > 0
        ):
            try:
                await redis_client.set(cache_key, content, ex=cache_ttl_seconds)
            except Exception as exc:
                logger.warning("Tool cache write failed for %s: %s", metadata.name, exc)

        # 4. Write audit log
        latency_ms = (time.perf_counter() - start) * 1000
        if audit_logger:
            await _safe_audit_log(
                audit_logger,
                ToolAuditEntry(
                    session_id=session_id,
                    user_id=user_id,
                    tool_name=metadata.name,
                    args=args,
                    status=status,
                    latency_ms=latency_ms,
                    error_message=error_message,
                ),
            )

        return ToolExecutionOutcome(
            content=content,
            ok=ok,
            is_write=metadata.is_write,
            cached=False,
            error_message=error_message,
        )


async def _safe_audit_log(audit_logger: ToolAuditLogger, entry: ToolAuditEntry) -> None:
    try:
        await audit_logger.log(entry)
    except Exception as exc:
        logger.warning("Failed to write tool audit log: %s", exc)
