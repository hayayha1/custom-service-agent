"""Unified MemoryManager coordinating L1 (Redis), L2 (Checkpointer recovery), and L3 (pgvector)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from src.memory.long_term import EpisodicMemoryRecord, LongTermMemory, UserFactRecord
from src.memory.short_term import ShortTermMemory, extract_slots_by_rules, merge_slots

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryContext:
    """Memory payload loaded at the start of a turn and injected into the prompt."""

    slots: dict[str, Any] = field(default_factory=dict)
    user_facts: list[UserFactRecord] = field(default_factory=list)
    episodic_memories: list[EpisodicMemoryRecord] = field(default_factory=list)
    redis_cache_hit: bool = False

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "redis_cache_hit": self.redis_cache_hit,
            "user_facts": [
                {"key": f.fact_key, "value": f.fact_value} for f in self.user_facts
            ],
            "episodic_memories": [
                {
                    "session_id": e.session_id,
                    "summary": e.summary,
                    "score": round(e.score, 4),
                }
                for e in self.episodic_memories
            ],
        }


class MemoryManager:
    """Coordinates synchronous L1 Redis updates, L2 Checkpointer backfill, and async L3 consolidation."""

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def load_for_turn(
        self,
        user_id: str,
        session_id: str,
        latest_user_text: str,
        checkpoint_messages: Sequence[BaseMessage],
        checkpoint_slots: dict[str, Any],
    ) -> MemoryContext:
        """Load L1 slots (falling back to L2 Checkpointer on Redis miss) and recall L3 memory."""
        redis_hit = await self.short_term.has_session_cache(session_id)
        merged_slots = merge_slots({}, checkpoint_slots)

        if redis_hit:
            redis_slots = await self.short_term.get_slots(session_id)
            merged_slots = merge_slots(merged_slots, redis_slots)
            await self.short_term.refresh_ttl(session_id)
        else:
            # Redis expired / cache miss -> recover full state from LangGraph Checkpointer & warm Redis
            if checkpoint_messages:
                await self.short_term.replace_messages(session_id, checkpoint_messages)
            if merged_slots:
                await self.short_term.update_slots(session_id, merged_slots)

        # Deterministic rule-based slot extraction on latest user turn (supports multiple orders)
        rule_slots = extract_slots_by_rules(latest_user_text)
        merged_slots = merge_slots(merged_slots, rule_slots)

        # Recall L3 long-term facts + episodic summaries (soft-fail so L3 hiccups never break chat)
        user_facts: list[UserFactRecord] = []
        episodic_memories: list[EpisodicMemoryRecord] = []
        try:
            user_facts, episodic_memories = await self.long_term.recall(
                user_id=user_id,
                query=latest_user_text,
            )
        except Exception as exc:
            logger.warning("L3 memory recall failed for user=%s: %s", user_id, exc)

        return MemoryContext(
            slots=merged_slots,
            user_facts=user_facts,
            episodic_memories=episodic_memories,
            redis_cache_hit=redis_hit,
        )

    async def save_after_turn(
        self,
        user_id: str,
        session_id: str,
        all_messages: Sequence[BaseMessage],
        slots: dict[str, Any],
        should_consolidate: bool = False,
    ) -> None:
        """Synchronously persist L1 working memory and optionally schedule async L3 consolidation."""
        await self.short_term.replace_messages(session_id, all_messages)
        if slots:
            await self.short_term.update_slots(session_id, slots)

        if should_consolidate:
            self.schedule_consolidation(
                user_id=user_id,
                session_id=session_id,
                messages=list(all_messages),
                slots=dict(slots),
            )

    def schedule_consolidation(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[BaseMessage],
        slots: dict[str, Any],
    ) -> asyncio.Task[bool]:
        """Fire-and-forget L3 consolidation task tracked for graceful shutdown."""
        async def _runner() -> bool:
            try:
                return await self.long_term.consolidate(
                    user_id=user_id,
                    session_id=session_id,
                    messages=messages,
                    slots=slots,
                )
            except Exception:
                logger.exception(
                    "Background L3 memory consolidation failed for session=%s",
                    session_id,
                )
                return False

        task = asyncio.create_task(_runner())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def consolidate_now(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[BaseMessage],
        slots: dict[str, Any],
    ) -> bool:
        """Synchronously run L3 consolidation (used by archive API or tests)."""
        return await self.long_term.consolidate(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            slots=slots,
        )

    async def flush_background_tasks(self) -> None:
        """Await all pending background consolidation tasks (graceful shutdown / tests)."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        await asyncio.gather(*pending, return_exceptions=True)
