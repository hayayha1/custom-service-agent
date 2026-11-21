"""LangGraph StateGraph assembly and CustomerServiceAgent service facade with Redis session locking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import AgentNodes
from src.agent.state import AgentState
from src.db.redis import RedisSessionLock
from src.memory.manager import MemoryManager
from src.observability.tracer import AgentObservabilityManager, trace_turn_context
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class ChatResult:
    reply: str
    user_id: str
    session_id: str
    needs_human: bool = False
    escalation_reason: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    tool_calls_used: int = 0
    redis_cache_hit: bool = False
    deduplicated: bool = False
    trace_id: str | None = None


def build_agent_graph(
    *,
    llm: Any,
    memory_manager: MemoryManager,
    tool_registry: ToolRegistry,
    checkpointer: Any,
    max_tool_iterations: int = 5,
    short_term_max_turns: int = 10,
    summary_every_n_turns: int = 10,
) -> Any:
    """Compile the Phase 1 LangGraph StateGraph with Checkpointer persistence."""
    nodes = AgentNodes(
        llm=llm,
        memory_manager=memory_manager,
        tool_registry=tool_registry,
        max_tool_iterations=max_tool_iterations,
        short_term_max_turns=short_term_max_turns,
        summary_every_n_turns=summary_every_n_turns,
    )

    builder = StateGraph(AgentState)
    builder.add_node("load_memory", nodes.load_memory)
    builder.add_node("agent_llm", nodes.agent_llm)
    builder.add_node("tool_executor", nodes.tool_executor)
    builder.add_node("fallback_or_escalate", nodes.fallback_or_escalate)
    builder.add_node("save_memory", nodes.save_memory)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "agent_llm")
    builder.add_conditional_edges(
        "agent_llm",
        nodes.route_after_agent,
        {
            "tool_executor": "tool_executor",
            "fallback_or_escalate": "fallback_or_escalate",
            "save_memory": "save_memory",
        },
    )
    builder.add_conditional_edges(
        "tool_executor",
        nodes.route_after_tool,
        {
            "agent_llm": "agent_llm",
            "fallback_or_escalate": "fallback_or_escalate",
        },
    )
    builder.add_edge("fallback_or_escalate", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile(checkpointer=checkpointer)


class CustomerServiceAgent:
    """High-level runtime facade adding Redis session locking and request idempotency around the graph."""

    def __init__(
        self,
        *,
        graph: Any,
        redis_client: Any,
        memory_manager: MemoryManager,
        observability: AgentObservabilityManager | None = None,
        lock_ttl_seconds: int = 120,
        lock_wait_seconds: float = 5.0,
        idempotency_ttl_seconds: int = 300,
    ) -> None:
        self.graph = graph
        self.redis_client = redis_client
        self.memory_manager = memory_manager
        self.observability = observability or AgentObservabilityManager()
        self._lock_ttl_seconds = lock_ttl_seconds
        self._lock_wait_seconds = lock_wait_seconds
        self._idempotency_ttl_seconds = idempotency_ttl_seconds

    async def chat(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        request_id: str | None = None,
    ) -> ChatResult:
        """Process one customer turn under a per-session Redis distributed lock and full-chain Trace."""
        idem_key = (
            f"cs:req_idem:{session_id}:{request_id}" if request_id else None
        )

        lock = RedisSessionLock(
            redis_client=self.redis_client,
            session_id=session_id,
            ttl_seconds=self._lock_ttl_seconds,
            wait_seconds=self._lock_wait_seconds,
        )
        async with lock:
            if idem_key:
                cached = await self.redis_client.get(idem_key)
                if cached:
                    data = json.loads(cached)
                    return ChatResult(**data, deduplicated=True)

            with trace_turn_context(
                session_id=session_id,
                user_id=user_id,
                user_message=message,
            ) as trace_ctx:
                config = {"configurable": {"thread_id": session_id}}
                final_state = await self.graph.ainvoke(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "messages": [HumanMessage(content=message)],
                    },
                    config=config,
                )

                reply = _extract_final_reply(final_state.get("messages") or [])
                mem_ctx = final_state.get("memory_context") or {}
                needs_human = bool(final_state.get("needs_human", False))
                esc_reason = final_state.get("escalation_reason")
                redis_hit = bool(mem_ctx.get("redis_cache_hit", False))

                trace_record = trace_ctx.to_trace_record(
                    final_reply=reply,
                    needs_human=needs_human,
                    escalation_reason=esc_reason,
                    redis_cache_hit=redis_hit,
                    error_message=final_state.get("error"),
                )
                self.observability.record_trace(trace_record)

            result = ChatResult(
                reply=reply,
                user_id=user_id,
                session_id=session_id,
                needs_human=needs_human,
                escalation_reason=esc_reason,
                slots=dict(final_state.get("slots") or {}),
                tool_calls_used=int(final_state.get("tool_call_count") or 0),
                redis_cache_hit=redis_hit,
                trace_id=trace_record.trace_id,
            )

            if idem_key:
                await self.redis_client.set(
                    idem_key,
                    json.dumps(
                        {
                            "reply": result.reply,
                            "user_id": result.user_id,
                            "session_id": result.session_id,
                            "needs_human": result.needs_human,
                            "escalation_reason": result.escalation_reason,
                            "slots": result.slots,
                            "tool_calls_used": result.tool_calls_used,
                            "redis_cache_hit": result.redis_cache_hit,
                            "trace_id": result.trace_id,
                        },
                        ensure_ascii=False,
                    ),
                    ex=self._idempotency_ttl_seconds,
                )

            return result

    async def get_session_memory(self, session_id: str) -> dict[str, Any]:
        """Inspect current L1 Redis working memory for a session."""
        messages = await self.memory_manager.short_term.get_messages(session_id)
        slots = await self.memory_manager.short_term.get_slots(session_id)
        return {
            "session_id": session_id,
            "redis_cached": messages is not None or bool(slots),
            "slots": slots,
            "messages": [
                {"type": m.type, "content": m.content}
                for m in (messages or [])
            ],
        }

    async def archive_session(self, *, user_id: str, session_id: str) -> bool:
        """Explicitly consolidate a session into L3 pgvector long-term memory."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values if snapshot and snapshot.values else {}
        messages = values.get("messages") or []
        slots = values.get("slots") or {}
        if not messages:
            messages = await self.memory_manager.short_term.get_messages(session_id) or []
            slots = await self.memory_manager.short_term.get_slots(session_id)
        return await self.memory_manager.consolidate_now(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            slots=slots,
        )


def _extract_final_reply(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return "抱歉，暂时无法生成回复，请稍后重试。"
