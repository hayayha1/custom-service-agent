"""Vertical domain knowledge base store and retrieval tool (Phase 2 Hybrid + RRF + Rerank)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.db.postgres import format_vector_literal
from src.kb.tokenizer import build_postgres_tsquery, tokenize_for_fts
from src.memory.long_term import EmbeddingProvider
from src.retrieval.models import KBChunkMatch, KBChunkRecord
from src.retrieval.pipeline import HybridRetrievalPipeline


@dataclass(slots=True)
class KBDocument:
    doc_key: str
    category: str
    title: str
    content: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeBaseStore(Protocol):
    async def search(
        self,
        query_embedding: list[float],
        *,
        category: str | None = None,
        top_k: int = 3,
    ) -> list[KBDocument]: ...

    async def upsert_document(
        self,
        doc_key: str,
        category: str,
        title: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def list_documents(self, category: str | None = None) -> list[KBDocument]: ...

    async def delete_document(self, doc_key: str) -> bool: ...


class PostgresKnowledgeBaseStore:
    """PostgreSQL + pgvector + GIN tsvector knowledge base store over `kb_documents` & `kb_chunks`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def get_document_hash(self, doc_key: str) -> str | None:
        sql = "SELECT content_hash FROM kb_documents WHERE doc_key = %s"
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (doc_key,))
            row = await cur.fetchone()
        if not row:
            return None
        val = row.get("content_hash")
        return str(val) if val else None

    async def get_document_chunk_count(self, doc_key: str) -> int:
        sql = "SELECT chunk_count FROM kb_documents WHERE doc_key = %s"
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (doc_key,))
            row = await cur.fetchone()
        return int(row["chunk_count"]) if row and row.get("chunk_count") else 0

    async def search(
        self,
        query_embedding: list[float],
        *,
        category: str | None = None,
        top_k: int = 3,
    ) -> list[KBDocument]:
        vec = format_vector_literal(query_embedding)
        if category:
            sql = """
                SELECT doc_key, category, title, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM kb_documents
                WHERE category = %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
            """
            params: tuple[Any, ...] = (vec, category, vec, top_k)
        else:
            sql = """
                SELECT doc_key, category, title, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM kb_documents
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
            """
            params = (vec, vec, top_k)

        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        return [
            KBDocument(
                doc_key=row["doc_key"],
                category=row["category"],
                title=row["title"],
                content=row["content"],
                score=float(row["score"]),
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    async def search_chunks_dense(
        self,
        query_embedding: list[float],
        *,
        category: str | None = None,
        top_k: int = 15,
    ) -> list[KBChunkMatch]:
        vec = format_vector_literal(query_embedding)
        if category:
            sql = """
                SELECT chunk_key, doc_key, category, title, section_path,
                       content, enriched_content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM kb_chunks
                WHERE category = %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
            """
            params: tuple[Any, ...] = (vec, category, vec, top_k)
        else:
            sql = """
                SELECT chunk_key, doc_key, category, title, section_path,
                       content, enriched_content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM kb_chunks
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
            """
            params = (vec, vec, top_k)

        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        if not rows:
            # Backward-compatible fallback if a legacy document has no rows in kb_chunks yet
            legacy_docs = await self.search(query_embedding, category=category, top_k=top_k)
            return [
                KBChunkMatch(
                    chunk_key=f"{d.doc_key}#c0",
                    doc_key=d.doc_key,
                    category=d.category,
                    title=d.title,
                    section_path="",
                    content=d.content,
                    enriched_content=f"【{d.title}】\n{d.content}",
                    score=d.score,
                    dense_score=d.score,
                    metadata=d.metadata,
                )
                for d in legacy_docs
            ]

        return [
            KBChunkMatch(
                chunk_key=row["chunk_key"],
                doc_key=row["doc_key"],
                category=row["category"],
                title=row["title"],
                section_path=row["section_path"] or "",
                content=row["content"],
                enriched_content=row["enriched_content"],
                score=float(row["score"]),
                dense_score=float(row["score"]),
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    async def search_chunks_sparse(
        self,
        query_text: str,
        *,
        category: str | None = None,
        top_k: int = 15,
    ) -> list[KBChunkMatch]:
        tsquery_str = build_postgres_tsquery(query_text)
        if not tsquery_str:
            return []

        if category:
            sql = """
                SELECT chunk_key, doc_key, category, title, section_path,
                       content, enriched_content, metadata,
                       ts_rank_cd(search_tsv, to_tsquery('simple', %s)) AS score
                FROM kb_chunks
                WHERE category = %s
                  AND search_tsv @@ to_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT %s
            """
            params: tuple[Any, ...] = (tsquery_str, category, tsquery_str, top_k)
        else:
            sql = """
                SELECT chunk_key, doc_key, category, title, section_path,
                       content, enriched_content, metadata,
                       ts_rank_cd(search_tsv, to_tsquery('simple', %s)) AS score
                FROM kb_chunks
                WHERE search_tsv @@ to_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT %s
            """
            params = (tsquery_str, tsquery_str, top_k)

        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        return [
            KBChunkMatch(
                chunk_key=row["chunk_key"],
                doc_key=row["doc_key"],
                category=row["category"],
                title=row["title"],
                section_path=row["section_path"] or "",
                content=row["content"],
                enriched_content=row["enriched_content"],
                score=float(row["score"]),
                sparse_score=float(row["score"]),
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

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
        default_chunk = KBChunkRecord(
            chunk_key=f"{doc_key}#c0",
            doc_key=doc_key,
            category=category,
            title=title,
            section_path="",
            chunk_index=0,
            content=content,
            enriched_content=f"【{title}】\n{content}",
            search_tokens=tokenize_for_fts(f"{title} {content}"),
            embedding=embedding,
            metadata=metadata or {},
        )
        await self.upsert_document_with_chunks(
            doc_key=doc_key,
            category=category,
            title=title,
            content=content,
            content_hash=content_hash,
            embedding=embedding,
            chunks=[default_chunk],
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
        doc_vec = format_vector_literal(embedding)
        doc_sql = """
            INSERT INTO kb_documents (
                doc_key, category, title, content, content_hash, chunk_count, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (doc_key)
            DO UPDATE SET
                category = EXCLUDED.category,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                chunk_count = EXCLUDED.chunk_count,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """
        delete_chunks_sql = "DELETE FROM kb_chunks WHERE doc_key = %s"
        insert_chunk_sql = """
            INSERT INTO kb_chunks (
                chunk_key, doc_key, category, title, section_path, chunk_index,
                content, enriched_content, search_tokens, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
        """
        meta_json = json.dumps(
            {**(metadata or {}), "chunk_count": len(chunks), "content_hash": content_hash},
            ensure_ascii=False,
        )
        async with self._pool.connection() as conn:
            await conn.execute(
                doc_sql,
                (
                    doc_key,
                    category,
                    title,
                    content,
                    content_hash,
                    len(chunks),
                    meta_json,
                    doc_vec,
                ),
            )
            await conn.execute(delete_chunks_sql, (doc_key,))
            for ch in chunks:
                await conn.execute(
                    insert_chunk_sql,
                    (
                        ch.chunk_key,
                        ch.doc_key,
                        ch.category,
                        ch.title,
                        ch.section_path,
                        ch.chunk_index,
                        ch.content,
                        ch.enriched_content,
                        ch.search_tokens,
                        json.dumps(ch.metadata, ensure_ascii=False),
                        format_vector_literal(ch.embedding),
                    ),
                )

    async def list_documents(self, category: str | None = None) -> list[KBDocument]:
        if category:
            sql = """
                SELECT doc_key, category, title, content, content_hash, chunk_count, metadata
                FROM kb_documents
                WHERE category = %s
                ORDER BY updated_at DESC
            """
            params: tuple[Any, ...] = (category,)
        else:
            sql = """
                SELECT doc_key, category, title, content, content_hash, chunk_count, metadata
                FROM kb_documents
                ORDER BY updated_at DESC
            """
            params = ()

        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        return [
            KBDocument(
                doc_key=row["doc_key"],
                category=row["category"],
                title=row["title"],
                content=row["content"],
                score=1.0,
                metadata={
                    **(row["metadata"] or {}),
                    "content_hash": row.get("content_hash") or "",
                    "chunk_count": row.get("chunk_count") or 1,
                },
            )
            for row in rows
        ]

    async def delete_document(self, doc_key: str) -> bool:
        sql = "DELETE FROM kb_documents WHERE doc_key = %s"
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, (doc_key,))
            return bool(cur.rowcount and cur.rowcount > 0)


class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(..., min_length=1, description="用户的咨询问题或关键词")
    category: str | None = Field(
        default=None,
        description="可选知识分类：return_policy（退换货）、shipping（物流）、warranty（质保）、invoice（发票）",
    )


def build_knowledge_tool(
    kb_store: KnowledgeBaseStore,
    embeddings: EmbeddingProvider,
    top_k: int = 3,
    reranker: Any | None = None,
    retrieval_pipeline: HybridRetrievalPipeline | None = None,
) -> StructuredTool:
    """Create the `search_knowledge_base` tool backed by the Phase 2 4-stage HybridRetrievalPipeline."""
    pipeline = retrieval_pipeline or HybridRetrievalPipeline(
        kb_store=kb_store,
        embeddings=embeddings,
        reranker=reranker,
    )

    async def _search_knowledge_base(query: str, category: str | None = None) -> str:
        result = await pipeline.retrieve(query, category=category, top_k=top_k)
        if not result.matches:
            return json.dumps(
                {
                    "ok": True,
                    "matches": [],
                    "rewritten_query": result.rewritten_query,
                    "zero_hit": True,
                    "message": "知识库未命中相关政策文档，请谨慎回答或建议转人工核实。",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "ok": True,
                "rewritten_query": result.rewritten_query,
                "reranked": result.reranked,
                "matches": [
                    {
                        "chunk_key": m.chunk_key,
                        "doc_key": m.doc_key,
                        "title": m.title,
                        "section_path": m.section_path,
                        "category": m.category,
                        "score": round(m.score, 4),
                        "dense_score": round(m.dense_score, 4),
                        "sparse_score": round(m.sparse_score, 4),
                        "rrf_score": round(m.rrf_score, 4),
                        "rerank_score": round(m.rerank_score, 4) if m.rerank_score is not None else None,
                        "content": m.content,
                    }
                    for m in result.matches
                ],
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=_search_knowledge_base,
        name="search_knowledge_base",
        description="检索垂直客服知识库（退换货政策、运费险、质保、发货时效、发票规则等 SOP 文档）。回答政策类问题前必须优先调用。",
        args_schema=SearchKnowledgeBaseInput,
    )
