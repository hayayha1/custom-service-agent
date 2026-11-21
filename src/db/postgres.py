"""PostgreSQL + pgvector connection pool, schema migrations, and LangGraph Checkpointer."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


def configure_windows_event_loop() -> None:
    """Psycopg async requires SelectorEventLoop on Windows."""
    if sys.platform == "win32":
        policy = asyncio.get_event_loop_policy()
        if not isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def _configure_connection(conn: Any) -> None:
    """Register pgvector adapter on each newly opened connection."""
    try:
        await register_vector_async(conn)
    except Exception as exc:
        # Extension may not exist yet prior to init_db(); will succeed once created.
        logger.debug("Deferring pgvector registration until init_db(): %s", exc)


async def create_postgres_pool(
    dsn: str,
    min_size: int = 1,
    max_size: int = 10,
) -> AsyncConnectionPool:
    """Create and open an AsyncConnectionPool configured for LangGraph & pgvector."""
    pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        open=False,
        configure=_configure_connection,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    await pool.open()
    return pool


def format_vector_literal(vector: list[float]) -> str:
    """Format a float embedding vector as pgvector text literal `[0.1,0.2,...]`."""
    return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"


async def init_db(pool: AsyncConnectionPool, embedding_dim: int = 1536) -> None:
    """Idempotently initialize pgvector extension and Phase 1 tables/indexes."""
    if embedding_dim <= 0 or embedding_dim > 16000:
        raise ValueError(f"Invalid embedding_dim: {embedding_dim}")

    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        f"""
        CREATE TABLE IF NOT EXISTS kb_documents (
            id BIGSERIAL PRIMARY KEY,
            doc_key TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({embedding_dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_kb_documents_category
        ON kb_documents (category);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_kb_documents_embedding_hnsw
        ON kb_documents USING hnsw (embedding vector_cosine_ops);
        """,
        """
        ALTER TABLE kb_documents
            ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '',
            ADD COLUMN IF NOT EXISTS chunk_count INT NOT NULL DEFAULT 1,
            ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'markdown';
        """,
        f"""
        CREATE TABLE IF NOT EXISTS kb_chunks (
            id BIGSERIAL PRIMARY KEY,
            chunk_key TEXT UNIQUE NOT NULL,
            doc_key TEXT NOT NULL REFERENCES kb_documents(doc_key) ON DELETE CASCADE,
            category TEXT NOT NULL DEFAULT 'general',
            title TEXT NOT NULL,
            section_path TEXT NOT NULL DEFAULT '',
            chunk_index INT NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            enriched_content TEXT NOT NULL,
            search_tokens TEXT NOT NULL DEFAULT '',
            search_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', search_tokens)) STORED,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({embedding_dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_key
        ON kb_chunks (doc_key);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_category
        ON kb_chunks (category);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_search_tsv
        ON kb_chunks USING gin (search_tsv);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding_hnsw
        ON kb_chunks USING hnsw (embedding vector_cosine_ops);
        """,
        f"""
        CREATE TABLE IF NOT EXISTS episodic_memories (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT UNIQUE NOT NULL,
            summary TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({embedding_dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_id
        ON episodic_memories (user_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_episodic_memories_embedding_hnsw
        ON episodic_memories USING hnsw (embedding vector_cosine_ops);
        """,
        """
        CREATE TABLE IF NOT EXISTS user_facts (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            source_session_id TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (user_id, fact_key)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_facts_user_active
        ON user_facts (user_id, is_active);
        """,
        """
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            order_id TEXT,
            issue_summary TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            status TEXT NOT NULL DEFAULT 'open',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS tool_audit_logs (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            args_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL,
            latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            error_message TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_tool_audit_logs_session
        ON tool_audit_logs (session_id, created_at DESC);
        """,
    ]

    async with pool.connection() as conn:
        for stmt in statements:
            await conn.execute(stmt)
        await register_vector_async(conn)


async def create_checkpointer(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    """Create and setup LangGraph AsyncPostgresSaver on the shared connection pool."""
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return checkpointer
