"""LangGraph AgentState definition for the vertical customer service loop."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Persistent state tracked across graph nodes and LangGraph Checkpointer."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    session_id: str
    slots: dict[str, Any]
    memory_context: dict[str, Any]
    tool_call_count: int
    turn_count: int
    needs_human: bool
    escalation_reason: str | None
    should_consolidate_memory: bool
    error: str | None
