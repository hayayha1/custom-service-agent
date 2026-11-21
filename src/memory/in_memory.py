"""Lightweight in-memory fallback stores when local Docker Postgres/Redis is unavailable."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from src.kb.tokenizer import tokenize_for_fts, tokenize_terms
from src.memory.long_term import (
    EpisodicMemoryRecord,
    ExtractedFact,
    UserFactRecord,
)
from src.retrieval.models import KBChunkMatch, KBChunkRecord
from src.tools.knowledge import KBDocument


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryLongTermStore:
    """In-memory LongTermMemoryStore fallback for local development without Postgres."""

    def __init__(self) -> None:
        self.episodic: dict[str, tuple[EpisodicMemoryRecord, list[float]]] = {}
        self.facts: dict[tuple[str, str], UserFactRecord] = {}

    async def search_episodic(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> list[EpisodicMemoryRecord]:
        scored: list[EpisodicMemoryRecord] = []
        for record, vec in self.episodic.values():
            if record.user_id != user_id:
                continue
            score = _cosine_similarity(query_embedding, vec)
            if score >= min_score:
                scored.append(
                    EpisodicMemoryRecord(
                        user_id=record.user_id,
                        session_id=record.session_id,
                        summary=record.summary,
                        score=score,
                        metadata=record.metadata,
                        created_at=record.created_at,
                    )
                )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def upsert_episodic(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        rec = EpisodicMemoryRecord(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            score=1.0,
            metadata=metadata or {},
            created_at=datetime.now(UTC),
        )
        self.episodic[session_id] = (rec, list(embedding))

    async def get_user_facts(self, user_id: str) -> list[UserFactRecord]:
        return [
            fact
            for (uid, _), fact in self.facts.items()
            if uid == user_id and fact.is_active
        ]

    async def upsert_user_facts(
        self,
        user_id: str,
        facts: Sequence[ExtractedFact],
        source_session_id: str,
    ) -> None:
        for fact in facts:
            key = fact.fact_key.strip().lower()
            self.facts[(user_id, key)] = UserFactRecord(
                user_id=user_id,
                fact_key=key,
                fact_value=fact.fact_value.strip(),
                source_session_id=source_session_id,
                is_active=True,
            )


class InMemoryKBStore:
    """In-memory KnowledgeBaseStore supporting both document-level and Phase 2 chunk-level Dense+Sparse search."""

    def __init__(self) -> None:
        self._docs: dict[str, tuple[KBDocument, list[float]]] = {}
        self._chunks: dict[str, KBChunkRecord] = {}
        self._hashes: dict[str, str] = {}
        self._chunk_counts: dict[str, int] = {}

    async def get_document_hash(self, doc_key: str) -> str | None:
        return self._hashes.get(doc_key)

    async def get_document_chunk_count(self, doc_key: str) -> int:
        return self._chunk_counts.get(doc_key, 0)

    async def upsert_document(
        self,
        doc_key: str,
        category: str,
        title: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunk = KBChunkRecord(
            chunk_key=f"{doc_key}#c0",
            doc_key=doc_key,
            category=category,
            title=title,
            section_path="",
            chunk_index=0,
            content=content,
            enriched_content=f"【{title}】\n{content}",
            search_tokens=tokenize_for_fts(f"{title} {content}"),
            embedding=list(embedding),
            metadata=metadata or {},
        )
        await self.upsert_document_with_chunks(
            doc_key=doc_key,
            category=category,
            title=title,
            content=content,
            content_hash=content_hash,
            embedding=embedding,
            chunks=[chunk],
            metadata=metadata,
        )

    async def upsert_document_with_chunks(
        self,
        *,
        doc_key: str,
        category: str,
        title: str,
        content: str,
        content_hash: str,
        embedding: list[float],
        chunks: Sequence[KBChunkRecord],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._hashes[doc_key] = content_hash
        self._chunk_counts[doc_key] = len(chunks)
        doc = KBDocument(
            doc_key=doc_key,
            category=category,
            title=title,
            content=content,
            score=1.0,
            metadata={
                **(metadata or {}),
                "content_hash": content_hash,
                "chunk_count": len(chunks),
            },
        )
        self._docs[doc_key] = (doc, list(embedding))
        # Remove previous chunks of this doc_key
        for k in [ck for ck, cr in self._chunks.items() if cr.doc_key == doc_key]:
            self._chunks.pop(k, None)
        for ch in chunks:
            self._chunks[ch.chunk_key] = ch

    async def search(
        self,
        query_embedding: list[float],
        *,
        category: str | None = None,
        top_k: int = 3,
    ) -> list[KBDocument]:
        results: list[KBDocument] = []
        for doc, vec in self._docs.values():
            if category and doc.category != category:
                continue
            score = _cosine_similarity(query_embedding, vec)
            results.append(
                KBDocument(
                    doc_key=doc.doc_key,
                    category=doc.category,
                    title=doc.title,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                )
            )
        results.sort(key=lambda d: d.score, reverse=True)
        return results[:top_k]

    async def search_chunks_dense(
        self,
        query_embedding: list[float],
        *,
        category: str | None = None,
        top_k: int = 15,
    ) -> list[KBChunkMatch]:
        results: list[KBChunkMatch] = []
        for ch in self._chunks.values():
            if category and ch.category != category:
                continue
            score = _cosine_similarity(query_embedding, ch.embedding)
            results.append(
                KBChunkMatch(
                    chunk_key=ch.chunk_key,
                    doc_key=ch.doc_key,
                    category=ch.category,
                    title=ch.title,
                    section_path=ch.section_path,
                    content=ch.content,
                    enriched_content=ch.enriched_content,
                    score=score,
                    dense_score=score,
                    metadata=dict(ch.metadata),
                )
            )
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_k]

    async def search_chunks_sparse(
        self,
        query_text: str,
        *,
        category: str | None = None,
        top_k: int = 15,
    ) -> list[KBChunkMatch]:
        q_terms = set(tokenize_terms(query_text))
        if not q_terms:
            return []
        results: list[KBChunkMatch] = []
        for ch in self._chunks.values():
            if category and ch.category != category:
                continue
            doc_terms = ch.search_tokens.split()
            if not doc_terms:
                continue
            matched = sum(1 for t in q_terms if t in doc_terms)
            if matched <= 0:
                continue
            score = matched / math.sqrt(len(q_terms) * max(1, len(set(doc_terms))))
            results.append(
                KBChunkMatch(
                    chunk_key=ch.chunk_key,
                    doc_key=ch.doc_key,
                    category=ch.category,
                    title=ch.title,
                    section_path=ch.section_path,
                    content=ch.content,
                    enriched_content=ch.enriched_content,
                    score=score,
                    sparse_score=score,
                    metadata=dict(ch.metadata),
                )
            )
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_k]

    async def list_documents(self, category: str | None = None) -> list[KBDocument]:
        docs = [doc for doc, _ in self._docs.values()]
        if category:
            docs = [d for d in docs if d.category == category]
        return docs

    async def delete_document(self, doc_key: str) -> bool:
        existed = self._docs.pop(doc_key, None) is not None
        self._hashes.pop(doc_key, None)
        self._chunk_counts.pop(doc_key, None)
        for k in [ck for ck, cr in self._chunks.items() if cr.doc_key == doc_key]:
            self._chunks.pop(k, None)
        return existed
