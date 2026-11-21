"""Tests for Phase 2 Agent Observability: Trace/Span waterfall, bottleneck/error detection, Token & TPS."""

from __future__ import annotations

import fakeredis.aioredis
import httpx
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import CustomerServiceAgent, build_agent_graph
from src.api.main import create_app
from src.kb.ingestion import KnowledgeIngestionPipeline
from src.memory.in_memory import InMemoryKBStore, InMemoryLongTermStore
from src.memory.long_term import LongTermMemory
from src.memory.manager import MemoryManager
from src.memory.short_term import ShortTermMemory
from src.observability.tracer import AgentObservabilityManager
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.tools.base import InMemoryToolAuditLogger
from src.tools.business import InMemoryTicketStore, MockEcommerceAdapter
from src.tools.registry import build_default_tool_registry
from tests.fakes import HashEmbeddings, RuleBasedMemoryExtractor, ScriptedLLM


async def test_full_chain_trace_waterfall_tokens_tps_and_monitoring_apis() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    embeddings = HashEmbeddings(dim=64)
    kb_store = InMemoryKBStore()
    lt_store = InMemoryLongTermStore()
    observability = AgentObservabilityManager(langfuse_enabled=False)

    ingestion = KnowledgeIngestionPipeline(kb_store=kb_store, embeddings=embeddings)
    await ingestion.ingest_document(
        doc_key="policy_return_7day",
        category="return_policy",
        title="7天无理由退换货政策",
        raw_content="# 退换货\n签收7日内支持无理由退货，质量问题运费平台承担。",
    )
    retrieval_pipeline = HybridRetrievalPipeline(kb_store=kb_store, embeddings=embeddings)

    short_term = ShortTermMemory(redis_client=redis)
    long_term = LongTermMemory(
        store=lt_store,
        embeddings=embeddings,
        extractor=RuleBasedMemoryExtractor(),
    )
    memory_manager = MemoryManager(short_term=short_term, long_term=long_term)
    registry = build_default_tool_registry(
        kb_store=kb_store,
        embeddings=embeddings,
        business_adapter=MockEcommerceAdapter(),
        ticket_store=InMemoryTicketStore(),
        retrieval_pipeline=retrieval_pipeline,
        redis_client=redis,
        audit_logger=InMemoryToolAuditLogger(),
    )

    def handler(messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(
                content="根据7天无理由退换货政策，质量问题退货运费由平台承担。",
                usage_metadata={"input_tokens": 180, "output_tokens": 45, "total_tokens": 225},
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "退钱邮费谁掏"},
                    "id": "call_kb_1",
                }
            ],
            usage_metadata={"input_tokens": 120, "output_tokens": 25, "total_tokens": 145},
        )

    llm = ScriptedLLM(handler)
    graph = build_agent_graph(
        llm=llm,
        memory_manager=memory_manager,
        tool_registry=registry,
        checkpointer=MemorySaver(),
    )
    agent = CustomerServiceAgent(
        graph=graph,
        redis_client=redis,
        memory_manager=memory_manager,
        observability=observability,
    )
    app = create_app(
        agent=agent,
        kb_store=kb_store,
        embeddings=embeddings,
        observability=observability,
    )

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        # 1. Trigger a full chat turn with KB retrieval
        chat_resp = await client.post(
            "/chat",
            json={
                "user_id": "user_001",
                "session_id": "sess-obs-1",
                "message": "东西不想要了退钱邮费谁掏？",
            },
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        trace_id = chat_data["trace_id"]
        assert trace_id

        # 2. Verify Trace detail contains full Span waterfall (load_memory, llm, tool, retrieval 4 stages, save_memory)
        trace_resp = await client.get(f"/traces/{trace_id}")
        assert trace_resp.status_code == 200
        tr = trace_resp.json()
        assert tr["prompt_tokens"] == 300
        assert tr["completion_tokens"] == 70
        assert tr["total_tokens"] == 370
        assert tr["llm_tps"] > 0
        assert tr["tool_call_count"] == 1
        assert tr["slowest_span_name"] != ""

        span_types = {sp["span_type"] for sp in tr["spans"]}
        assert "graph_node" in span_types
        assert "llm" in span_types
        assert "tool" in span_types
        assert "retrieval_rewrite" in span_types
        assert "retrieval_dense" in span_types
        assert "retrieval_sparse" in span_types
        assert "retrieval_rrf" in span_types
        assert "retrieval_rerank" in span_types

        # 3. Verify /metrics/summary aggregates TPS, Tokens, Latency, Tool & Retrieval KPIs
        metrics_resp = await client.get("/metrics/summary", params={"window_seconds": 3600})
        assert metrics_resp.status_code == 200
        m = metrics_resp.json()
        assert m["total_requests"] == 1
        assert m["total_tokens"] == 370
        assert m["avg_llm_tps"] > 0
        assert m["request_tps"] > 0
        assert m["tool_total_calls"] == 1
        assert m["tool_success_rate"] == 1.0
        assert m["retrieval_total_queries"] == 1
        assert m["retrieval_zero_hit_rate"] == 0.0

        # 4. Verify /kb/ingest and /kb/search Phase 2 diagnostics
        ing_resp = await client.post(
            "/kb/ingest",
            json={
                "doc_key": "faq_shipping",
                "category": "shipping",
                "title": "物流FAQ",
                "content": "Q1: 手机号 13811112222 怎么查快递？\nA1: 订单发货48小时内可在详情页查看顺丰运单号。",
                "chunk_mode": "faq",
                "redact_pii": True,
            },
        )
        assert ing_resp.status_code == 200
        assert ing_resp.json()["pii_redacted_count"] == 1
        assert ing_resp.json()["chunk_count"] == 1

        search_resp = await client.post(
            "/kb/search",
            json={"query": "它怎么退钱，邮费谁掏", "slots": {"product_name": "声学降噪耳机 Pro"}},
        )
        assert search_resp.status_code == 200
        sdata = search_resp.json()
        assert sdata["rewrite_applied"] is True
        assert "声学降噪耳机 Pro" in sdata["rewritten_query"]
        assert "timings" in sdata
        assert len(sdata["matches"]) >= 1

    await redis.aclose()
