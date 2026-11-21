"""L3 Long-Term Memory backed by PostgreSQL + pgvector (episodic summaries + structured user facts)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.db.postgres import format_vector_literal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EpisodicMemoryRecord:
    user_id: str
    session_id: str
    summary: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(slots=True)
class UserFactRecord:
    user_id: str
    fact_key: str
    fact_value: str
    source_session_id: str | None = None
    is_active: bool = True


class ExtractedFact(BaseModel):
    fact_key: str = Field(
        description="Canonical short snake_case key, e.g. preferred_carrier, device_model, vip_note"
    )
    fact_value: str = Field(description="Concise factual value valid across future sessions")


class MemoryExtractionResult(BaseModel):
    worth_saving: bool = Field(
        description="False if the conversation is pure greetings/chitchat with no reusable context"
    )
    summary: str = Field(
        default="",
        description="Concise Chinese summary: user issue + key entities + resolution/pending action",
    )
    facts: list[ExtractedFact] = Field(
        default_factory=list,
        description="Stable long-term user attributes or preferences worth remembering across sessions",
    )


class EmbeddingProvider(Protocol):
    async def aembed_query(self, text: str) -> list[float]: ...


class MemoryExtractor(Protocol):
    async def extract(
        self,
        messages: Sequence[BaseMessage],
        slots: dict[str, Any],
    ) -> MemoryExtractionResult: ...


class LongTermMemoryStore(Protocol):
    async def search_episodic(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> list[EpisodicMemoryRecord]: ...

    async def upsert_episodic(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def get_user_facts(self, user_id: str) -> list[UserFactRecord]: ...

    async def upsert_user_facts(
        self,
        user_id: str,
        facts: Sequence[ExtractedFact],
        source_session_id: str,
    ) -> None: ...


class PostgresLongTermStore:
    """PostgreSQL + pgvector implementation of LongTermMemoryStore."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def search_episodic(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> list[EpisodicMemoryRecord]:
        vec = format_vector_literal(query_embedding)
        sql = """
            SELECT
                user_id,
                session_id,
                summary,
                metadata,
                created_at,
                1 - (embedding <=> %s::vector) AS score
            FROM episodic_memories
            WHERE user_id = %s
              AND (1 - (embedding <=> %s::vector)) >= %s
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
        """
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (vec, user_id, vec, min_score, vec, top_k))
            rows = await cur.fetchall()

        return [
            EpisodicMemoryRecord(
                user_id=row["user_id"],
                session_id=row["session_id"],
                summary=row["summary"],
                score=float(row["score"]),
                metadata=row["metadata"] or {},
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def upsert_episodic(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = format_vector_literal(embedding)
        sql = """
            INSERT INTO episodic_memories (user_id, session_id, summary, metadata, embedding)
            VALUES (%s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (session_id)
            DO UPDATE SET
                summary = EXCLUDED.summary,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """
        async with self._pool.connection() as conn:
            await conn.execute(
                sql,
                (
                    user_id,
                    session_id,
                    summary,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    vec,
                ),
            )

    async def get_user_facts(self, user_id: str) -> list[UserFactRecord]:
        sql = """
            SELECT user_id, fact_key, fact_value, source_session_id, is_active
            FROM user_facts
            WHERE user_id = %s AND is_active = TRUE
            ORDER BY updated_at DESC
        """
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (user_id,))
            rows = await cur.fetchall()

        return [
            UserFactRecord(
                user_id=row["user_id"],
                fact_key=row["fact_key"],
                fact_value=row["fact_value"],
                source_session_id=row["source_session_id"],
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    async def upsert_user_facts(
        self,
        user_id: str,
        facts: Sequence[ExtractedFact],
        source_session_id: str,
    ) -> None:
        if not facts:
            return
        sql = """
            INSERT INTO user_facts (user_id, fact_key, fact_value, source_session_id, is_active)
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (user_id, fact_key)
            DO UPDATE SET
                fact_value = EXCLUDED.fact_value,
                source_session_id = EXCLUDED.source_session_id,
                is_active = TRUE,
                updated_at = NOW()
        """
        async with self._pool.connection() as conn:
            for fact in facts:
                await conn.execute(
                    sql,
                    (
                        user_id,
                        fact.fact_key.strip().lower(),
                        fact.fact_value.strip(),
                        source_session_id,
                    ),
                )


class LLMMemoryExtractor:
    """Uses structured LLM output to filter noise and extract episodic summary + user facts."""

    _SYSTEM_PROMPT = (
        "你是客服系统的记忆提炼模块。请分析用户与客服的对话和已知槽位，输出结构化 JSON：\n"
        "1. 如果对话仅为无实质信息的寒暄（如'你好'、'在吗'、'谢谢'），将 worth_saving 设为 false。\n"
        "2. 否则将 worth_saving 设为 true，并用 1~2 句精炼中文写出 summary（包含用户诉求、核心订单/商品/工单实体、处理结果或待跟进事项）。\n"
        "3. 仅提取跨会话仍然有效的长期用户偏好或稳定属性放入 facts（例如 preferred_carrier=顺丰、preferred_contact=电话），不要把一次性订单号放入 facts。"
    )

    def __init__(self, llm: Any) -> None:
        self._structured_llm = llm.with_structured_output(MemoryExtractionResult)

    async def extract(
        self,
        messages: Sequence[BaseMessage],
        slots: dict[str, Any],
    ) -> MemoryExtractionResult:
        transcript_lines: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                transcript_lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                transcript_lines.append(f"客服: {msg.content}")

        if not transcript_lines:
            return MemoryExtractionResult(worth_saving=False)

        prompt_body = (
            f"已知业务槽位: {json.dumps(slots, ensure_ascii=False)}\n\n"
            f"对话记录:\n" + "\n".join(transcript_lines[-20:])
        )
        result = await self._structured_llm.ainvoke(
            [
                SystemMessage(content=self._SYSTEM_PROMPT),
                HumanMessage(content=prompt_body),
            ]
        )
        if isinstance(result, MemoryExtractionResult):
            return result
        return MemoryExtractionResult.model_validate(result)


class LongTermMemory:
    """High-level L3 long-term memory service orchestrating recall and consolidation."""

    def __init__(
        self,
        store: LongTermMemoryStore,
        embeddings: EmbeddingProvider,
        extractor: MemoryExtractor,
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._extractor = extractor
        self._top_k = top_k
        self._min_score = min_score

    async def recall(
        self,
        user_id: str,
        query: str,
    ) -> tuple[list[UserFactRecord], list[EpisodicMemoryRecord]]:
        facts = await self._store.get_user_facts(user_id)
        episodic: list[EpisodicMemoryRecord] = []
        if query.strip():
            query_vec = await self._embeddings.aembed_query(query)
            episodic = await self._store.search_episodic(
                user_id=user_id,
                query_embedding=query_vec,
                top_k=self._top_k,
                min_score=self._min_score,
            )
        return facts, episodic

    async def consolidate(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[BaseMessage],
        slots: dict[str, Any],
    ) -> bool:
        """Extract and persist episodic summary & upserted user facts if worth saving."""
        extraction = await self._extractor.extract(messages, slots)
        if not extraction.worth_saving:
            logger.debug("Skipping L3 consolidation for session=%s (not worth saving)", session_id)
            return False

        saved_any = False
        if extraction.summary.strip():
            summary_vec = await self._embeddings.aembed_query(extraction.summary)
            await self._store.upsert_episodic(
                user_id=user_id,
                session_id=session_id,
                summary=extraction.summary.strip(),
                embedding=summary_vec,
                metadata={"slots": slots},
            )
            saved_any = True

        if extraction.facts:
            await self._store.upsert_user_facts(
                user_id=user_id,
                facts=extraction.facts,
                source_session_id=session_id,
            )
            saved_any = True

        return saved_any
