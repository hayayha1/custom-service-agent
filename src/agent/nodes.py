"""LangGraph node implementations: load_memory, agent_llm, tool_executor, fallback_or_escalate, save_memory."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.memory.manager import MemoryManager
from src.memory.short_term import merge_slots, trim_messages_safely
from src.observability.callbacks import record_llm_call_on_current_trace
from src.observability.tracer import start_observed_span
from src.retrieval.pipeline import reset_active_retrieval_slots, set_active_retrieval_slots
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_FAREWELL_KEYWORDS = ("谢谢", "再见", "拜拜", "没问题了", "解决了", "就这样吧", "结束对话")


def _get_latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _is_farewell(text: str) -> bool:
    cleaned = text.strip()
    return any(kw in cleaned for kw in _FAREWELL_KEYWORDS)


class AgentNodes:
    """Encapsulates dependencies and implements all 5 nodes of the Phase 1 Agent Loop."""

    def __init__(
        self,
        *,
        llm: Any,
        memory_manager: MemoryManager,
        tool_registry: ToolRegistry,
        max_tool_iterations: int = 5,
        short_term_max_turns: int = 10,
        summary_every_n_turns: int = 10,
    ) -> None:
        self._llm = llm
        self._llm_with_tools = llm.bind_tools(tool_registry.langchain_tools)
        self._memory = memory_manager
        self._tools = tool_registry
        self._max_tool_iterations = max_tool_iterations
        self._short_term_max_turns = short_term_max_turns
        self._summary_every_n_turns = summary_every_n_turns

    async def load_memory(self, state: AgentState) -> dict[str, Any]:
        """Node 1: Load L1 working memory (or backfill from L2 Checkpointer) + L3 long-term recall."""
        user_id = state["user_id"]
        session_id = state["session_id"]
        all_messages = list(state.get("messages") or [])
        latest_user_text = _get_latest_human_text(all_messages)
        checkpoint_slots = dict(state.get("slots") or {})

        prior_messages = all_messages[:-1] if len(all_messages) > 1 else []
        with start_observed_span(
            span_type="graph_node",
            name="load_memory",
            input_summary={"session_id": session_id, "user_id": user_id},
        ) as sp:
            mem_ctx = await self._memory.load_for_turn(
                user_id=user_id,
                session_id=session_id,
                latest_user_text=latest_user_text,
                checkpoint_messages=prior_messages,
                checkpoint_slots=checkpoint_slots,
            )
            if sp is not None:
                sp.set_output(
                    {
                        "redis_cache_hit": mem_ctx.redis_cache_hit,
                        "slot_keys": list(mem_ctx.slots.keys()),
                        "facts_count": len(mem_ctx.user_facts),
                        "episodic_count": len(mem_ctx.episodic_memories),
                    }
                )

        next_turn_count = int(state.get("turn_count") or 0) + 1
        should_consolidate = _is_farewell(latest_user_text) or (
            self._summary_every_n_turns > 0
            and next_turn_count % self._summary_every_n_turns == 0
        )

        return {
            "slots": mem_ctx.slots,
            "memory_context": mem_ctx.to_state_dict(),
            "tool_call_count": 0,
            "turn_count": next_turn_count,
            "needs_human": False,
            "escalation_reason": None,
            "should_consolidate_memory": should_consolidate,
            "error": None,
        }

    async def agent_llm(self, state: AgentState) -> dict[str, Any]:
        """Node 2: Assemble System Prompt (pinned slots + L3 memory) + trimmed history and call LLM."""
        system_prompt = build_system_prompt(
            user_id=state["user_id"],
            slots=state.get("slots") or {},
            memory_context=state.get("memory_context") or {},
        )
        trimmed_history = trim_messages_safely(
            state.get("messages") or [],
            max_turns=self._short_term_max_turns,
        )
        prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *trimmed_history]
        model_name = str(getattr(self._llm, "model_name", None) or getattr(self._llm, "model", "llm"))
        t0 = time.perf_counter()

        try:
            response = await self._llm_with_tools.ainvoke(prompt_messages)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            record_llm_call_on_current_trace(
                model=model_name,
                response=response,
                latency_ms=latency_ms,
            )
            return {"messages": [response], "error": None}
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = f"LLM 调用异常: {type(exc).__name__}: {exc}"
            record_llm_call_on_current_trace(
                model=model_name,
                response=None,
                latency_ms=latency_ms,
                error_message=err_msg,
            )
            logger.exception("LLM invocation failed for session=%s", state["session_id"])
            return {"error": err_msg}

    async def tool_executor(self, state: AgentState) -> dict[str, Any]:
        """Node 3: Execute requested tool calls with guardrails, slot updates, and escalation detection."""
        messages = state.get("messages") or []
        last_msg = messages[-1] if messages else None
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {}

        updated_slots = merge_slots({}, state.get("slots") or {})
        tool_messages: list[ToolMessage] = []
        needs_human = bool(state.get("needs_human", False))
        escalation_reason = state.get("escalation_reason")
        should_consolidate = bool(state.get("should_consolidate_memory", False))

        prepared_calls: list[tuple[str, dict[str, Any], str]] = []
        for call in last_msg.tool_calls:
            tool_name = call["name"]
            tool_args = dict(call.get("args") or {})
            tool_call_id = call.get("id") or f"call_{tool_name}"
            if tool_name in ("get_user_orders", "create_support_ticket") and not tool_args.get("user_id"):
                tool_args["user_id"] = state["user_id"]
            prepared_calls.append((tool_name, tool_args, tool_call_id))

        slots_token = set_active_retrieval_slots(updated_slots)
        try:
            # Execute all tool calls in this turn concurrently to minimize multi-tool latency
            outcomes = await asyncio.gather(
                *(
                    self._tools.execute(
                        tool_name,
                        tool_args,
                        session_id=state["session_id"],
                        user_id=state["user_id"],
                    )
                    for tool_name, tool_args, _ in prepared_calls
                )
            )
        finally:
            reset_active_retrieval_slots(slots_token)

        for (tool_name, tool_args, tool_call_id), outcome in zip(
            prepared_calls, outcomes, strict=True
        ):
            tool_messages.append(
                ToolMessage(
                    content=outcome.content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

            # Update active slots & trigger flags based on tool side-effects
            if outcome.ok:
                updated_slots = self._apply_tool_side_effects(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    content=outcome.content,
                    slots=updated_slots,
                )
                if outcome.is_write:
                    should_consolidate = True

                if tool_name == "transfer_to_human":
                    needs_human = True
                    escalation_reason = str(tool_args.get("reason") or "客户请求转接人工客服")
                    should_consolidate = True

        return {
            "messages": tool_messages,
            "slots": updated_slots,
            "tool_call_count": int(state.get("tool_call_count") or 0) + 1,
            "needs_human": needs_human,
            "escalation_reason": escalation_reason,
            "should_consolidate_memory": should_consolidate,
        }

    async def fallback_or_escalate(self, state: AgentState) -> dict[str, Any]:
        """Node 4: Graceful handoff / loop-limit circuit breaker / error fallback."""
        extra_messages: list[BaseMessage] = []
        all_messages = state.get("messages") or []
        last_msg = all_messages[-1] if all_messages else None

        # If we tripped the loop guard while last_msg still has pending tool_calls, close them cleanly
        # so future OpenAI-compatible API turns never fail on dangling tool_call_ids.
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for call in last_msg.tool_calls:
                extra_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "ok": False,
                                "error_type": "loop_limit_exceeded",
                                "message": "已达到本轮最大工具调用次数上限，转为人工兜底。",
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=call.get("id") or f"cancelled_{call['name']}",
                        name=call["name"],
                    )
                )

        reason = state.get("escalation_reason")
        if state.get("error"):
            reply_text = (
                "非常抱歉，当前客服系统遇到临时波动，未能完成您的请求。"
                "我已为您记录上下文并优先转接人工客服为您处理，请稍候。"
            )
            reason = reason or state["error"]
        elif state.get("needs_human"):
            reply_text = (
                f"好的，我已为您登记转接人工客服专员（原因：{reason or '需要人工进一步核实'}），"
                "并同步整理了您当前的订单与沟通记录，请稍候片刻。"
            )
        else:
            reply_text = (
                "抱歉，该问题排查步骤较多，为避免耽误您的时间，我已为您转接人工客服专员继续跟进，"
                "已查询到的信息会自动同步给人工客服。"
            )
            reason = reason or f"达到单轮最大工具调用次数上限({self._max_tool_iterations})"

        extra_messages.append(AIMessage(content=reply_text))
        return {
            "messages": extra_messages,
            "needs_human": True,
            "escalation_reason": reason,
            "should_consolidate_memory": True,
        }

    async def save_memory(self, state: AgentState) -> dict[str, Any]:
        """Node 5: Synchronously persist L1 Redis working memory and trigger async L3 consolidation."""
        should_consolidate = bool(state.get("should_consolidate_memory", False))
        with start_observed_span(
            span_type="graph_node",
            name="save_memory",
            input_summary={"should_consolidate": should_consolidate},
        ):
            await self._memory.save_after_turn(
                user_id=state["user_id"],
                session_id=state["session_id"],
                all_messages=state.get("messages") or [],
                slots=state.get("slots") or {},
                should_consolidate=should_consolidate,
            )
        return {}

    def route_after_agent(self, state: AgentState) -> str:
        """Conditional edge after `agent_llm`."""
        if state.get("error"):
            return "fallback_or_escalate"

        messages = state.get("messages") or []
        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            if int(state.get("tool_call_count") or 0) >= self._max_tool_iterations:
                return "fallback_or_escalate"
            return "tool_executor"

        return "save_memory"

    def route_after_tool(self, state: AgentState) -> str:
        """Conditional edge after `tool_executor`."""
        if state.get("needs_human"):
            return "fallback_or_escalate"
        return "agent_llm"

    @staticmethod
    def _apply_tool_side_effects(
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        content: str,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            payload = {}

        if tool_name == "query_order_status":
            order = payload.get("order") or {}
            oid = str(order.get("order_id") or "").strip()
            if oid:
                order_entry = {
                    "order_id": oid,
                    "product_name": str(order.get("product_name") or ""),
                    "tracking_no": str(order.get("tracking_no") or ""),
                    "status": str(order.get("status") or ""),
                }
                incoming: dict[str, Any] = {
                    "order_id": oid,
                    "order_ids": [oid],
                    "active_orders": {oid: order_entry},
                }
                if order_entry["product_name"]:
                    incoming["product_name"] = order_entry["product_name"]
                if order_entry["tracking_no"]:
                    incoming["tracking_no"] = order_entry["tracking_no"]
                    incoming["tracking_nos"] = [order_entry["tracking_no"]]
                if order_entry["status"]:
                    incoming["order_status"] = order_entry["status"]
                return merge_slots(slots, incoming)

        elif tool_name == "get_user_orders":
            orders = payload.get("orders") or []
            active_orders: dict[str, dict[str, Any]] = {}
            order_ids: list[str] = []
            for order in orders:
                if not isinstance(order, dict):
                    continue
                oid = str(order.get("order_id") or "").strip()
                if not oid:
                    continue
                order_ids.append(oid)
                active_orders[oid] = {
                    "order_id": oid,
                    "product_name": str(order.get("product_name") or ""),
                    "tracking_no": str(order.get("tracking_no") or ""),
                    "status": str(order.get("status") or ""),
                }
            if active_orders:
                return merge_slots(
                    slots,
                    {
                        "order_id": order_ids[-1],
                        "order_ids": order_ids,
                        "active_orders": active_orders,
                    },
                )

        elif tool_name == "create_support_ticket":
            ticket = payload.get("ticket") or {}
            tid = str(ticket.get("ticket_id") or "").strip()
            oid = str(ticket.get("order_id") or tool_args.get("order_id") or "").strip()
            incoming_ticket: dict[str, Any] = {"current_intent": "ticket_created"}
            if tid:
                incoming_ticket["last_ticket_id"] = tid
                incoming_ticket["active_tickets"] = {
                    tid: {
                        "ticket_id": tid,
                        "order_id": oid,
                        "issue_summary": str(
                            ticket.get("issue_summary") or tool_args.get("issue_summary") or ""
                        ),
                        "priority": str(ticket.get("priority") or tool_args.get("priority") or "normal"),
                    }
                }
            if oid:
                incoming_ticket["order_id"] = oid
                incoming_ticket["order_ids"] = [oid]
                incoming_ticket["active_orders"] = {
                    oid: {"order_id": oid, **({"ticket_id": tid} if tid else {})}
                }
            return merge_slots(slots, incoming_ticket)

        return slots
