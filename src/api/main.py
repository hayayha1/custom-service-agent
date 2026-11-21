"""FastAPI entrypoint exposing chat, memory, Phase 2 KB ingestion/hybrid search, observability, and settings APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.graph import CustomerServiceAgent
from src.config import Settings, get_settings
from src.db.postgres import configure_windows_event_loop
from src.db.redis import SessionBusyError
from src.kb.ingestion import KnowledgeIngestionPipeline
from src.memory.long_term import EmbeddingProvider
from src.observability.tracer import AgentObservabilityManager
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.runtime import create_app_container
from src.settings_store import (
    ModelSettingsResponse,
    ModelSettingsStore,
    ModelSettingsUpdateRequest,
)
from src.tools.knowledge import KnowledgeBaseStore

configure_windows_event_loop()


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="唯一用户 ID")
    session_id: str = Field(..., min_length=1, description="当前会话 ID（映射 LangGraph thread_id）")
    message: str = Field(..., min_length=1, description="用户输入消息")
    request_id: str | None = Field(default=None, description="可选客户端幂等请求 ID")


class ChatResponse(BaseModel):
    reply: str
    user_id: str
    session_id: str
    needs_human: bool
    escalation_reason: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    tool_calls_used: int = 0
    redis_cache_hit: bool = False
    deduplicated: bool = False
    trace_id: str | None = None


class ArchiveRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


class KBDocumentUpsertRequest(BaseModel):
    doc_key: str = Field(default="", description="唯一文档标识，留空时自动生成")
    category: str = Field(default="general", min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    chunk_mode: Literal["auto", "faq", "sop"] = Field(default="auto")
    redact_pii: bool = Field(default=True)
    chunk_size: int = Field(default=450, ge=100, le=4000)
    chunk_overlap: int = Field(default=70, ge=0, le=1000)
    force: bool = Field(default=False)


class KBSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: str | None = None
    top_k: int = Field(default=4, ge=1, le=20)
    slots: dict[str, Any] | None = None


def create_app(
    *,
    agent: CustomerServiceAgent | None = None,
    settings: Settings | None = None,
    kb_store: KnowledgeBaseStore | None = None,
    embeddings: EmbeddingProvider | None = None,
    reranker: Any | None = None,
    model_settings_store: ModelSettingsStore | None = None,
    observability: AgentObservabilityManager | None = None,
    registered_tools: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI app. Inject dependencies directly in tests to bypass live DB startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        if agent is not None:
            app.state.agent = agent
            app.state.container = None
            app.state.kb_store = kb_store
            app.state.embeddings = embeddings
            app.state.reranker = reranker
            app.state.observability = observability or getattr(agent, "observability", None) or AgentObservabilityManager()
            app.state.ingestion_pipeline = (
                KnowledgeIngestionPipeline(kb_store=kb_store, embeddings=embeddings)
                if (kb_store is not None and embeddings is not None)
                else None
            )
            app.state.retrieval_pipeline = (
                HybridRetrievalPipeline(kb_store=kb_store, embeddings=embeddings, reranker=reranker)
                if (kb_store is not None and embeddings is not None)
                else None
            )
            app.state.model_settings_store = model_settings_store or ModelSettingsStore(cfg)
            app.state.registered_tools = registered_tools or [
                "search_knowledge_base",
                "query_order_status",
                "get_user_orders",
                "create_support_ticket",
                "transfer_to_human",
            ]
            yield
            return

        container = await create_app_container(cfg)
        app.state.container = container
        app.state.agent = container.agent
        app.state.kb_store = container.kb_store
        app.state.embeddings = container.embeddings
        app.state.reranker = container.reranker
        app.state.observability = container.observability
        app.state.ingestion_pipeline = container.ingestion_pipeline
        app.state.retrieval_pipeline = container.retrieval_pipeline
        app.state.model_settings_store = container.model_settings_store
        app.state.registered_tools = [t.name for t in container.tool_registry.langchain_tools]
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(
        title="Vertical Customer Service Agent (Phase 2)",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        redis_ok = True
        try:
            await request.app.state.agent.redis_client.ping()
        except Exception:
            redis_ok = False
        return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok}

    @app.get("/system/overview")
    async def system_overview(request: Request) -> dict[str, Any]:
        redis_ok = True
        try:
            await request.app.state.agent.redis_client.ping()
        except Exception:
            redis_ok = False

        kb_count = 0
        store: KnowledgeBaseStore | None = request.app.state.kb_store
        if store is not None:
            try:
                docs = await store.list_documents()
                kb_count = len(docs)
            except Exception:
                kb_count = 0

        settings_store: ModelSettingsStore = request.app.state.model_settings_store
        models = settings_store.get_masked()
        container = request.app.state.container
        storage_mode = container.storage_mode if container is not None else "local_memory_fallback"
        postgres_ok = container.postgres_connected if container is not None else False
        obs: AgentObservabilityManager = request.app.state.observability
        metrics = obs.get_metrics_summary(window_seconds=3600)
        return {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "postgres": postgres_ok,
            "storage_mode": storage_mode,
            "kb_document_count": kb_count,
            "registered_tools": list(request.app.state.registered_tools),
            "models": models.model_dump(),
            "metrics_brief": {
                "total_requests": metrics.total_requests,
                "request_tps": metrics.request_tps,
                "avg_llm_tps": metrics.avg_llm_tps,
                "total_tokens": metrics.total_tokens,
                "p95_latency_ms": metrics.p95_latency_ms,
                "langfuse_enabled": metrics.langfuse_enabled,
                "langfuse_connected": metrics.langfuse_connected,
            },
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        cs_agent: CustomerServiceAgent = request.app.state.agent
        try:
            result = await cs_agent.chat(
                user_id=payload.user_id,
                session_id=payload.session_id,
                message=payload.message,
                request_id=payload.request_id,
            )
        except SessionBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc

        return ChatResponse(
            reply=result.reply,
            user_id=result.user_id,
            session_id=result.session_id,
            needs_human=result.needs_human,
            escalation_reason=result.escalation_reason,
            slots=result.slots,
            tool_calls_used=result.tool_calls_used,
            redis_cache_hit=result.redis_cache_hit,
            deduplicated=result.deduplicated,
            trace_id=result.trace_id,
        )

    @app.get("/sessions/{session_id}/memory")
    async def get_session_memory(session_id: str, request: Request) -> dict[str, Any]:
        cs_agent: CustomerServiceAgent = request.app.state.agent
        return await cs_agent.get_session_memory(session_id)

    @app.post("/sessions/{session_id}/archive")
    async def archive_session(
        session_id: str,
        payload: ArchiveRequest,
        request: Request,
    ) -> dict[str, Any]:
        cs_agent: CustomerServiceAgent = request.app.state.agent
        saved = await cs_agent.archive_session(
            user_id=payload.user_id,
            session_id=session_id,
        )
        return {"session_id": session_id, "user_id": payload.user_id, "archived": saved}

    @app.get("/kb/documents")
    async def list_kb_documents(
        request: Request,
        category: str | None = Query(default=None),
    ) -> dict[str, Any]:
        store: KnowledgeBaseStore | None = request.app.state.kb_store
        if store is None:
            return {"documents": []}
        docs = await store.list_documents(category=category or None)
        return {
            "documents": [
                {
                    "doc_key": d.doc_key,
                    "category": d.category,
                    "title": d.title,
                    "content": d.content,
                    "metadata": d.metadata,
                }
                for d in docs
            ]
        }

    async def _ingest_doc_handler(
        payload: KBDocumentUpsertRequest,
        request: Request,
    ) -> dict[str, Any]:
        pipeline: KnowledgeIngestionPipeline | None = request.app.state.ingestion_pipeline
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Knowledge ingestion pipeline is not initialized.",
            )
        ing = await pipeline.ingest_document(
            doc_key=payload.doc_key,
            category=payload.category,
            title=payload.title,
            raw_content=payload.content,
            chunk_mode=payload.chunk_mode,
            redact_pii=payload.redact_pii,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            force=payload.force,
            metadata={"source": "web_ui"},
        )
        return {
            "ok": True,
            "doc_key": ing.doc_key,
            "category": ing.category,
            "title": ing.title,
            "content_hash": ing.content_hash,
            "chunk_count": ing.chunk_count,
            "pii_redacted_count": ing.pii_redacted_count,
            "noise_removed_count": ing.noise_removed_count,
            "skipped_unchanged": ing.skipped_unchanged,
        }

    @app.post("/kb/documents")
    async def upsert_kb_document(
        payload: KBDocumentUpsertRequest,
        request: Request,
    ) -> dict[str, Any]:
        return await _ingest_doc_handler(payload, request)

    @app.post("/kb/ingest")
    async def ingest_kb_document(
        payload: KBDocumentUpsertRequest,
        request: Request,
    ) -> dict[str, Any]:
        return await _ingest_doc_handler(payload, request)

    @app.delete("/kb/documents/{doc_key}")
    async def delete_kb_document(doc_key: str, request: Request) -> dict[str, Any]:
        store: KnowledgeBaseStore | None = request.app.state.kb_store
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Knowledge base store is not initialized.",
            )
        deleted = await store.delete_document(doc_key)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_key}' not found.",
            )
        return {"ok": True, "doc_key": doc_key}

    @app.post("/kb/search")
    async def search_kb_documents(
        payload: KBSearchRequest,
        request: Request,
    ) -> dict[str, Any]:
        pipeline: HybridRetrievalPipeline | None = request.app.state.retrieval_pipeline
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Hybrid retrieval pipeline is not initialized.",
            )
        res = await pipeline.retrieve(
            payload.query,
            category=payload.category or None,
            top_k=payload.top_k,
            slots=payload.slots,
        )
        return {
            "query": res.original_query,
            "rewritten_query": res.rewritten_query,
            "rewrite_applied": res.rewrite_applied,
            "dense_count": res.dense_count,
            "sparse_count": res.sparse_count,
            "fused_count": res.fused_count,
            "reranked": res.reranked,
            "zero_hit": res.zero_hit,
            "timings": asdict(res.timings),
            "matches": [
                {
                    "chunk_key": m.chunk_key,
                    "doc_key": m.doc_key,
                    "category": m.category,
                    "title": m.title,
                    "section_path": m.section_path,
                    "content": m.content,
                    "score": round(m.score, 4),
                    "dense_score": round(m.dense_score, 4),
                    "sparse_score": round(m.sparse_score, 4),
                    "rrf_score": round(m.rrf_score, 4),
                    "rerank_score": round(m.rerank_score, 4) if m.rerank_score is not None else None,
                }
                for m in res.matches
            ],
        }

    @app.get("/metrics/summary")
    async def get_metrics_summary(
        request: Request,
        window_seconds: int = Query(default=3600, ge=10, le=604800),
    ) -> dict[str, Any]:
        obs: AgentObservabilityManager = request.app.state.observability
        return obs.get_metrics_summary(window_seconds=window_seconds).to_dict()

    @app.get("/traces")
    async def list_traces(
        request: Request,
        session_id: str | None = Query(default=None),
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = Query(default=30, ge=1, le=200),
    ) -> dict[str, Any]:
        obs: AgentObservabilityManager = request.app.state.observability
        return {
            "traces": obs.list_traces(
                session_id=session_id or None,
                status=status_filter or None,
                limit=limit,
            )
        }

    @app.get("/traces/{trace_id}")
    async def get_trace_detail(trace_id: str, request: Request) -> dict[str, Any]:
        obs: AgentObservabilityManager = request.app.state.observability
        detail = obs.get_trace(trace_id)
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trace '{trace_id}' not found.",
            )
        return detail

    @app.get("/settings/models", response_model=ModelSettingsResponse)
    async def get_model_settings(request: Request) -> ModelSettingsResponse:
        settings_store: ModelSettingsStore = request.app.state.model_settings_store
        return settings_store.get_masked()

    @app.put("/settings/models", response_model=ModelSettingsResponse)
    async def update_model_settings(
        payload: ModelSettingsUpdateRequest,
        request: Request,
    ) -> ModelSettingsResponse:
        settings_store: ModelSettingsStore = request.app.state.model_settings_store
        updated = settings_store.update(payload)
        container = request.app.state.container
        if container is not None:
            container.rebuild_models_and_agent()
            request.app.state.agent = container.agent
            request.app.state.embeddings = container.embeddings
            request.app.state.reranker = container.reranker
            request.app.state.ingestion_pipeline = container.ingestion_pipeline
            request.app.state.retrieval_pipeline = container.retrieval_pipeline
        elif payload.langfuse is not None:
            raw_lf = settings_store.get_raw().get("langfuse") or {}
            obs: AgentObservabilityManager = request.app.state.observability
            obs.configure_langfuse(
                enabled=bool(raw_lf.get("enabled", False)),
                public_key=str(raw_lf.get("public_key", "")),
                secret_key=str(raw_lf.get("secret_key", "")),
                host=str(raw_lf.get("host", "https://cloud.langfuse.com")),
            )
        return updated

    return app


app = create_app()
