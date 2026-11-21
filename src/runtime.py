"""Application runtime container wiring Postgres, pgvector, Redis, LLM, Embeddings, Retrieval, and Observability."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import psycopg
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import CustomerServiceAgent, build_agent_graph
from src.config import Settings
from src.db.postgres import create_checkpointer, create_postgres_pool, init_db
from src.db.redis import create_redis_client
from src.kb.ingestion import KnowledgeIngestionPipeline
from src.memory.in_memory import InMemoryKBStore, InMemoryLongTermStore
from src.memory.long_term import (
    LLMMemoryExtractor,
    LongTermMemory,
    LongTermMemoryStore,
    PostgresLongTermStore,
)
from src.memory.manager import MemoryManager
from src.memory.short_term import ShortTermMemory
from src.observability.tracer import AgentObservabilityManager
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.retrieval.rewriter import QueryRewriter
from src.settings_store import ModelSettingsStore
from src.tools.base import (
    InMemoryToolAuditLogger,
    PostgresToolAuditLogger,
    ToolAuditLogger,
)
from src.tools.business import (
    InMemoryTicketStore,
    MockEcommerceAdapter,
    PostgresTicketStore,
    TicketStore,
)
from src.tools.knowledge import KnowledgeBaseStore, PostgresKnowledgeBaseStore
from src.tools.registry import ToolRegistry, build_default_tool_registry
from src.tools.reranker import APIReranker

logger = logging.getLogger(__name__)


async def _probe_postgres(dsn: str, timeout: float = 3.0) -> bool:
    try:
        conn = await asyncio.wait_for(
            psycopg.AsyncConnection.connect(dsn, connect_timeout=int(timeout)),
            timeout=timeout + 1.0,
        )
        await conn.close()
        return True
    except Exception as exc:
        logger.warning("PostgreSQL unreachable (%s): %s", dsn, exc)
        return False


async def _probe_redis(redis_url: str, timeout: float = 2.0) -> Any | None:
    client = create_redis_client(redis_url)
    try:
        await asyncio.wait_for(client.ping(), timeout=timeout)
        return client
    except Exception as exc:
        logger.warning("Redis unreachable (%s): %s", redis_url, exc)
        try:
            await client.aclose()
        except Exception as close_exc:
            logger.debug("Ignoring Redis close error during probe fallback: %s", close_exc)
        return None


@dataclass
class AppContainer:
    settings: Settings
    model_settings_store: ModelSettingsStore
    storage_mode: str
    postgres_connected: bool
    redis_connected: bool
    pg_pool: Any | None
    checkpointer: Any
    redis_client: Any
    kb_store: KnowledgeBaseStore
    long_term_store: LongTermMemoryStore
    ticket_store: TicketStore
    audit_logger: ToolAuditLogger
    observability: AgentObservabilityManager
    embeddings: OpenAIEmbeddings
    reranker: APIReranker
    ingestion_pipeline: KnowledgeIngestionPipeline
    retrieval_pipeline: HybridRetrievalPipeline
    tool_registry: ToolRegistry
    memory_manager: MemoryManager
    agent: CustomerServiceAgent

    def rebuild_models_and_agent(self) -> None:
        """Hot-reload LLM, Embeddings, Reranker, RetrievalPipeline, Langfuse, and LangGraph."""
        raw = self.model_settings_store.get_raw()
        llm_cfg = raw["llm"]
        emb_cfg = raw["embedding"]
        rr_cfg = raw["reranker"]
        lf_cfg = raw.get("langfuse") or {}

        self.observability.configure_langfuse(
            enabled=bool(lf_cfg.get("enabled", False)),
            public_key=str(lf_cfg.get("public_key", "")),
            secret_key=str(lf_cfg.get("secret_key", "")),
            host=str(lf_cfg.get("host", "https://cloud.langfuse.com")),
        )

        llm = ChatOpenAI(
            model=llm_cfg["model"],
            api_key=llm_cfg["api_key"] or "not-configured",
            base_url=llm_cfg["base_url"],
            temperature=float(llm_cfg["temperature"]),
            timeout=45.0,
            max_retries=1,
        )
        memory_llm = ChatOpenAI(
            model=llm_cfg.get("memory_model") or llm_cfg["model"],
            api_key=llm_cfg["api_key"] or "not-configured",
            base_url=llm_cfg["base_url"],
            temperature=0.0,
            timeout=30.0,
            max_retries=1,
        )
        emb_kwargs: dict[str, Any] = {
            "model": emb_cfg["model"],
            "api_key": emb_cfg.get("api_key") or llm_cfg["api_key"] or "not-configured",
            "base_url": emb_cfg["base_url"],
            "check_embedding_ctx_length": False,
        }
        if str(emb_cfg["model"]).startswith("text-embedding-3"):
            emb_kwargs["dimensions"] = int(emb_cfg["dim"])
        self.embeddings = OpenAIEmbeddings(**emb_kwargs)

        self.reranker = APIReranker(
            enabled=bool(rr_cfg.get("enabled", False)),
            base_url=str(rr_cfg.get("base_url", "")),
            api_key=str(rr_cfg.get("api_key", "")),
            model=str(rr_cfg.get("model", "")),
            top_n=int(rr_cfg.get("top_n", 4)),
            min_score=float(rr_cfg.get("min_score", 0.25)),
        )

        self.ingestion_pipeline = KnowledgeIngestionPipeline(
            kb_store=self.kb_store,
            embeddings=self.embeddings,
            chunk_size=self.settings.kb_chunk_size,
            chunk_overlap=self.settings.kb_chunk_overlap,
            redact_pii=self.settings.kb_redact_pii,
        )
        self.retrieval_pipeline = HybridRetrievalPipeline(
            kb_store=self.kb_store,
            embeddings=self.embeddings,
            rewriter=QueryRewriter(enabled=self.settings.retrieval_rewrite_enabled),
            reranker=self.reranker,
            dense_top_k=self.settings.retrieval_dense_top_k,
            sparse_top_k=self.settings.retrieval_sparse_top_k,
            rrf_k=self.settings.retrieval_rrf_k,
            rrf_top_k=self.settings.retrieval_rrf_top_k,
            dense_weight=self.settings.retrieval_dense_weight,
            sparse_weight=self.settings.retrieval_sparse_weight,
        )

        short_term = ShortTermMemory(
            redis_client=self.redis_client,
            max_turns=self.settings.short_term_max_turns,
            ttl_seconds=self.settings.session_ttl_seconds,
        )
        long_term = LongTermMemory(
            store=self.long_term_store,
            embeddings=self.embeddings,
            extractor=LLMMemoryExtractor(memory_llm),
            top_k=self.settings.long_term_top_k,
            min_score=self.settings.long_term_min_score,
        )
        self.memory_manager = MemoryManager(short_term=short_term, long_term=long_term)
        self.tool_registry = build_default_tool_registry(
            kb_store=self.kb_store,
            embeddings=self.embeddings,
            business_adapter=MockEcommerceAdapter(),
            ticket_store=self.ticket_store,
            reranker=self.reranker,
            retrieval_pipeline=self.retrieval_pipeline,
            default_timeout=self.settings.tool_timeout_seconds,
            redis_client=self.redis_client,
            cache_ttl_seconds=self.settings.tool_cache_ttl_seconds,
            audit_logger=self.audit_logger,
        )
        graph = build_agent_graph(
            llm=llm,
            memory_manager=self.memory_manager,
            tool_registry=self.tool_registry,
            checkpointer=self.checkpointer,
            max_tool_iterations=self.settings.max_tool_iterations,
            short_term_max_turns=self.settings.short_term_max_turns,
            summary_every_n_turns=self.settings.long_term_summary_every_n_turns,
        )
        self.agent = CustomerServiceAgent(
            graph=graph,
            redis_client=self.redis_client,
            memory_manager=self.memory_manager,
            observability=self.observability,
            lock_ttl_seconds=self.settings.session_lock_ttl_seconds,
            lock_wait_seconds=self.settings.session_lock_wait_seconds,
        )

    async def close(self) -> None:
        await self.memory_manager.flush_background_tasks()
        await self.redis_client.aclose()
        if self.pg_pool is not None:
            await self.pg_pool.close()


async def create_app_container(settings: Settings) -> AppContainer:
    """Initialize infrastructure (with optional graceful in-memory fallback for local dev)."""
    model_settings_store = ModelSettingsStore(settings)
    raw = model_settings_store.get_raw()

    # 1. PostgreSQL + pgvector
    pg_ok = await _probe_postgres(settings.postgres_dsn)
    pg_pool: Any | None = None
    if pg_ok:
        pg_pool = await create_postgres_pool(
            dsn=settings.postgres_dsn,
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size,
        )
        await init_db(pg_pool, embedding_dim=int(raw["embedding"]["dim"]))
        checkpointer: Any = await create_checkpointer(pg_pool)
        kb_store: KnowledgeBaseStore = PostgresKnowledgeBaseStore(pg_pool)
        long_term_store: LongTermMemoryStore = PostgresLongTermStore(pg_pool)
        ticket_store: TicketStore = PostgresTicketStore(pg_pool)
        audit_logger: ToolAuditLogger = PostgresToolAuditLogger(pg_pool)
    elif settings.local_dev_fallback:
        logger.warning(
            "Starting in LOCAL MEMORY FALLBACK mode for Postgres (start Docker Desktop & run "
            "'docker compose up -d' for persistent PostgreSQL + pgvector)."
        )
        checkpointer = MemorySaver()
        kb_store = InMemoryKBStore()
        long_term_store = InMemoryLongTermStore()
        ticket_store = InMemoryTicketStore()
        audit_logger = InMemoryToolAuditLogger()
    else:
        raise RuntimeError(
            f"Unable to connect to PostgreSQL at {settings.postgres_dsn} and LOCAL_DEV_FALLBACK=false."
        )

    # 2. Redis
    redis_client = await _probe_redis(settings.redis_url)
    redis_ok = redis_client is not None
    if redis_client is None:
        if settings.local_dev_fallback:
            import fakeredis.aioredis

            logger.warning(
                "Starting in LOCAL MEMORY FALLBACK mode for Redis (using fakeredis)."
            )
            redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        else:
            raise RuntimeError(
                f"Unable to connect to Redis at {settings.redis_url} and LOCAL_DEV_FALLBACK=false."
            )

    storage_mode = "postgres+redis" if (pg_ok and redis_ok) else "local_memory_fallback"
    observability = AgentObservabilityManager(
        langfuse_enabled=settings.langfuse_enabled,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        langfuse_host=settings.langfuse_host,
    )

    container = AppContainer(
        settings=settings,
        model_settings_store=model_settings_store,
        storage_mode=storage_mode,
        postgres_connected=pg_ok,
        redis_connected=redis_ok,
        pg_pool=pg_pool,
        checkpointer=checkpointer,
        redis_client=redis_client,
        kb_store=kb_store,
        long_term_store=long_term_store,
        ticket_store=ticket_store,
        audit_logger=audit_logger,
        observability=observability,
        embeddings=None,  # type: ignore[arg-type]
        reranker=None,  # type: ignore[arg-type]
        ingestion_pipeline=None,  # type: ignore[arg-type]
        retrieval_pipeline=None,  # type: ignore[arg-type]
        tool_registry=None,  # type: ignore[arg-type]
        memory_manager=None,  # type: ignore[arg-type]
        agent=None,  # type: ignore[arg-type]
    )
    container.rebuild_models_and_agent()
    return container
