"""Tests for Web UI backend APIs: system overview, KB CRUD/search, and LLM/Embedding/Reranker settings."""

from __future__ import annotations

from pathlib import Path

import fakeredis.aioredis
import httpx
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import CustomerServiceAgent, build_agent_graph
from src.api.main import create_app
from src.config import Settings
from src.memory.long_term import LongTermMemory
from src.memory.manager import MemoryManager
from src.memory.short_term import ShortTermMemory
from src.settings_store import ModelSettingsStore
from src.tools.base import InMemoryToolAuditLogger
from src.tools.business import InMemoryTicketStore, MockEcommerceAdapter
from src.tools.registry import build_default_tool_registry
from tests.fakes import (
    HashEmbeddings,
    InMemoryKBStore,
    InMemoryLongTermStore,
    RuleBasedMemoryExtractor,
    ScriptedLLM,
)


async def test_web_ui_kb_and_model_settings_apis(tmp_path: Path) -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    embeddings = HashEmbeddings()
    kb_store = InMemoryKBStore()
    lt_store = InMemoryLongTermStore()

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
        redis_client=redis,
        audit_logger=InMemoryToolAuditLogger(),
    )
    llm = ScriptedLLM(lambda _msgs: AIMessage(content="ok"))
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
    )
    settings_path = tmp_path / "model_settings.json"
    settings_store = ModelSettingsStore(Settings(), storage_path=settings_path)

    app = create_app(
        agent=agent,
        kb_store=kb_store,
        embeddings=embeddings,
        model_settings_store=settings_store,
    )

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        # 1. Upsert KB doc
        create_resp = await client.post(
            "/kb/documents",
            json={
                "doc_key": "policy_shipping",
                "category": "shipping",
                "title": "发货与物流异常规则",
                "content": "常规订单48小时内发货，72小时无更新支持补发或退款。",
            },
        )
        assert create_resp.status_code == 200
        assert create_resp.json()["doc_key"] == "policy_shipping"

        # 2. List & search KB docs
        list_resp = await client.get("/kb/documents", params={"category": "shipping"})
        assert list_resp.status_code == 200
        assert len(list_resp.json()["documents"]) == 1

        search_resp = await client.post(
            "/kb/search",
            json={"query": "发货后72小时物流没更新怎么办", "category": "shipping", "top_k": 3},
        )
        assert search_resp.status_code == 200
        assert search_resp.json()["matches"][0]["doc_key"] == "policy_shipping"

        # 3. Model settings GET & PUT (LLM, Embedding, Reranker) with secret masking
        put_resp = await client.put(
            "/settings/models",
            json={
                "llm": {
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                    "temperature": 0.1,
                    "memory_model": "deepseek-chat",
                    "api_key": "sk-test-llm-secret-9988",
                },
                "embedding": {
                    "base_url": "https://api.siliconflow.cn/v1",
                    "model": "BAAI/bge-m3",
                    "dim": 1024,
                    "api_key": "sk-test-emb-secret-1122",
                },
                "reranker": {
                    "enabled": True,
                    "base_url": "https://api.siliconflow.cn/v1/rerank",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "top_n": 4,
                    "min_score": 0.4,
                    "api_key": "sk-test-rr-secret-3344",
                },
            },
        )
        assert put_resp.status_code == 200
        body = put_resp.json()
        assert body["llm"]["model"] == "deepseek-chat"
        assert body["llm"]["has_api_key"] is True
        assert body["llm"]["api_key_masked"] == "sk-t...9988"
        assert body["embedding"]["model"] == "BAAI/bge-m3"
        assert body["reranker"]["enabled"] is True
        assert body["reranker"]["model"] == "BAAI/bge-reranker-v2-m3"
        assert settings_path.exists()

        # Updating without API key preserves existing secrets
        put_resp2 = await client.put(
            "/settings/models",
            json={
                "llm": {
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-reasoner",
                    "temperature": 0.2,
                    "memory_model": "",
                    "api_key": "",
                },
                "embedding": {
                    "base_url": "https://api.siliconflow.cn/v1",
                    "model": "BAAI/bge-m3",
                    "dim": 1024,
                },
                "reranker": {
                    "enabled": False,
                    "base_url": "https://api.siliconflow.cn/v1/rerank",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "top_n": 3,
                    "min_score": 0.35,
                },
            },
        )
        assert put_resp2.status_code == 200
        assert put_resp2.json()["llm"]["api_key_masked"] == "sk-t...9988"
        assert put_resp2.json()["reranker"]["enabled"] is False

        # 4. System overview reflects KB count & active model settings
        overview = await client.get("/system/overview")
        assert overview.status_code == 200
        ov = overview.json()
        assert ov["status"] == "ok"
        assert ov["kb_document_count"] == 1
        assert ov["models"]["llm"]["model"] == "deepseek-reasoner"

        # 5. Delete KB doc
        del_resp = await client.delete("/kb/documents/policy_shipping")
        assert del_resp.status_code == 200
        assert (await client.get("/kb/documents")).json()["documents"] == []

    await redis.aclose()
