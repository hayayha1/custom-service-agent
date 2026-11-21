"""ToolRegistry uniting LangChain tools, metadata (read/write/cache), and guarded execution."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool

from src.memory.long_term import EmbeddingProvider
from src.tools.base import (
    ToolAuditLogger,
    ToolExecutionOutcome,
    ToolMetadata,
    safe_execute_tool,
)
from src.tools.business import BusinessDataAdapter, TicketStore, build_business_tools
from src.tools.escalation import build_escalation_tool
from src.tools.knowledge import KnowledgeBaseStore, build_knowledge_tool


class ToolRegistry:
    """Central registry for all vertical CS tools and their runtime execution policies."""

    def __init__(
        self,
        *,
        default_timeout: float = 10.0,
        redis_client: Any | None = None,
        cache_ttl_seconds: int = 60,
        audit_logger: ToolAuditLogger | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, ToolMetadata] = {}
        self._default_timeout = default_timeout
        self._redis_client = redis_client
        self._cache_ttl_seconds = cache_ttl_seconds
        self._audit_logger = audit_logger

    def register(self, tool: BaseTool, metadata: ToolMetadata | None = None) -> None:
        meta = metadata or ToolMetadata(name=tool.name)
        self._tools[tool.name] = tool
        self._metadata[tool.name] = meta

    @property
    def langchain_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_metadata(self, tool_name: str) -> ToolMetadata | None:
        return self._metadata.get(tool_name)

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        session_id: str,
        user_id: str,
    ) -> ToolExecutionOutcome:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolExecutionOutcome(
                content=json.dumps(
                    {
                        "ok": False,
                        "error_type": "unknown_tool",
                        "message": f"未知工具 '{tool_name}'，请仅调用已注册的工具。",
                    },
                    ensure_ascii=False,
                ),
                ok=False,
                is_write=False,
                error_message=f"Unknown tool: {tool_name}",
            )

        metadata = self._metadata[tool_name]
        return await safe_execute_tool(
            tool=tool,
            metadata=metadata,
            args=args,
            session_id=session_id,
            user_id=user_id,
            default_timeout=self._default_timeout,
            redis_client=self._redis_client,
            cache_ttl_seconds=self._cache_ttl_seconds,
            audit_logger=self._audit_logger,
        )


def build_default_tool_registry(
    *,
    kb_store: KnowledgeBaseStore,
    embeddings: EmbeddingProvider,
    business_adapter: BusinessDataAdapter,
    ticket_store: TicketStore,
    reranker: Any | None = None,
    retrieval_pipeline: Any | None = None,
    default_timeout: float = 10.0,
    redis_client: Any | None = None,
    cache_ttl_seconds: int = 60,
    audit_logger: ToolAuditLogger | None = None,
) -> ToolRegistry:
    """Register the vertical customer service toolset."""
    registry = ToolRegistry(
        default_timeout=default_timeout,
        redis_client=redis_client,
        cache_ttl_seconds=cache_ttl_seconds,
        audit_logger=audit_logger,
    )

    kb_tool = build_knowledge_tool(
        kb_store=kb_store,
        embeddings=embeddings,
        reranker=reranker,
        retrieval_pipeline=retrieval_pipeline,
    )
    registry.register(
        kb_tool,
        ToolMetadata(name=kb_tool.name, is_write=False, cacheable=True),
    )

    for btool in build_business_tools(adapter=business_adapter, ticket_store=ticket_store):
        is_write = btool.name == "create_support_ticket"
        registry.register(
            btool,
            ToolMetadata(
                name=btool.name,
                is_write=is_write,
                cacheable=not is_write,
            ),
        )

    esc_tool = build_escalation_tool()
    registry.register(
        esc_tool,
        ToolMetadata(name=esc_tool.name, is_write=True, cacheable=False),
    )

    return registry
