"""Deterministic fakes for LLM, Embeddings, and pgvector stores used in automated tests."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.memory.in_memory import InMemoryKBStore, InMemoryLongTermStore
from src.memory.long_term import (
    ExtractedFact,
    MemoryExtractionResult,
)

__all__ = [
    "HashEmbeddings",
    "InMemoryKBStore",
    "InMemoryLongTermStore",
    "RuleBasedMemoryExtractor",
    "ScriptedLLM",
]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HashEmbeddings:
    """Character-bigram hashing embeddings providing real cosine similarity for Chinese & English."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    async def aembed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        cleaned = re.sub(r"\s+", "", text.lower())
        if not cleaned:
            return vec
        tokens = [cleaned[i : i + 2] for i in range(max(1, len(cleaned) - 1))] + list(cleaned)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class RuleBasedMemoryExtractor:
    """Deterministic MemoryExtractor for testing chitchat filtering, summaries, and fact conflicts."""

    _CHITCHAT = frozenset({"你好", "在吗", "嗨", "hello", "hi", "谢谢", "好的"})

    async def extract(
        self,
        messages: Sequence[BaseMessage],
        slots: dict[str, Any],
    ) -> MemoryExtractionResult:
        human_texts = [
            str(m.content).strip()
            for m in messages
            if isinstance(m, HumanMessage) and str(m.content).strip()
        ]
        if not human_texts:
            return MemoryExtractionResult(worth_saving=False)

        if all(t.lower().rstrip("！!。.") in self._CHITCHAT for t in human_texts) and not slots:
            return MemoryExtractionResult(worth_saving=False)

        joined = " | ".join(human_texts)
        facts: list[ExtractedFact] = []
        if "顺丰" in joined:
            facts.append(ExtractedFact(fact_key="preferred_carrier", fact_value="顺丰"))
        elif "中通" in joined:
            facts.append(ExtractedFact(fact_key="preferred_carrier", fact_value="中通"))

        if "电话联系" in joined:
            facts.append(ExtractedFact(fact_key="preferred_contact", fact_value="电话联系"))

        slot_summary = f"（关联订单:{slots['order_id']}）" if slots.get("order_id") else ""
        ticket_note = f"，已建工单 {slots['last_ticket_id']}" if slots.get("last_ticket_id") else ""
        summary = f"用户咨询：{joined}{slot_summary}{ticket_note}"
        return MemoryExtractionResult(worth_saving=True, summary=summary, facts=facts)


class ScriptedLLM:
    """Fake tool-bindable LLM that delegates decisions to a handler and records received prompts."""

    def __init__(self, handler: Callable[[list[BaseMessage]], AIMessage | Any]) -> None:
        self._handler = handler
        self.bound_tools: list[Any] = []
        self.received_prompts: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list[Any]) -> ScriptedLLM:
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.received_prompts.append(list(messages))
        result = self._handler(messages)
        if hasattr(result, "__await__"):
            result = await result
        return result
