"""Tests for tool guardrails: timeout, exceptions, validation, read cache, idempotent tickets, and audit."""

from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.tools.base import InMemoryToolAuditLogger, ToolMetadata
from src.tools.business import InMemoryTicketStore, MockEcommerceAdapter
from src.tools.registry import ToolRegistry, build_default_tool_registry
from tests.fakes import HashEmbeddings, InMemoryKBStore


async def test_tool_timeout_and_exception_never_crash_caller() -> None:
    audit = InMemoryToolAuditLogger()
    registry = ToolRegistry(default_timeout=0.05, audit_logger=audit)

    async def slow_tool(x: str) -> str:
        await asyncio.sleep(0.3)
        return x

    async def broken_tool(x: str) -> str:
        raise RuntimeError("upstream OMS 503")

    class SingleArg(BaseModel):
        x: str = Field(..., min_length=2)

    registry.register(
        StructuredTool.from_function(coroutine=slow_tool, name="slow_tool", description="slow", args_schema=SingleArg),
        ToolMetadata(name="slow_tool", timeout_seconds=0.05),
    )
    registry.register(
        StructuredTool.from_function(coroutine=broken_tool, name="broken_tool", description="broken", args_schema=SingleArg),
        ToolMetadata(name="broken_tool"),
    )

    # 1. Timeout returns structured JSON
    res_timeout = await registry.execute("slow_tool", {"x": "ok"}, session_id="s1", user_id="u1")
    assert res_timeout.ok is False
    assert json.loads(res_timeout.content)["error_type"] == "timeout"

    # 2. Runtime exception returns structured JSON
    res_err = await registry.execute("broken_tool", {"x": "ok"}, session_id="s1", user_id="u1")
    assert res_err.ok is False
    assert json.loads(res_err.content)["error_type"] == "execution_error"

    # 3. Pydantic validation failure returns structured JSON
    res_val = await registry.execute("broken_tool", {"x": ""}, session_id="s1", user_id="u1")
    assert res_val.ok is False
    assert json.loads(res_val.content)["error_type"] == "validation_error"

    # 4. Audit log captured all 3 statuses
    assert [e.status for e in audit.entries] == ["timeout", "exception", "validation_error"]


async def test_read_tool_cache_and_idempotent_ticket_creation() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    embeddings = HashEmbeddings()
    kb_store = InMemoryKBStore()
    doc_vec = await embeddings.aembed_query("7天无理由退换货政策 运费险 定制商品不可退")
    await kb_store.upsert_document(
        doc_key="policy_return",
        category="return_policy",
        title="7天无理由退换货政策",
        content="商品完好支持7天无理由退换货，质量问题运费平台承担。",
        embedding=doc_vec,
    )

    audit = InMemoryToolAuditLogger()
    registry = build_default_tool_registry(
        kb_store=kb_store,
        embeddings=embeddings,
        business_adapter=MockEcommerceAdapter(),
        ticket_store=InMemoryTicketStore(),
        redis_client=redis,
        cache_ttl_seconds=60,
        audit_logger=audit,
    )

    # 1. Read tool hits cache on 2nd identical invocation
    first = await registry.execute(
        "query_order_status",
        {"order_id": "20261119001"},
        session_id="s1",
        user_id="user_001",
    )
    second = await registry.execute(
        "query_order_status",
        {"order_id": "20261119001"},
        session_id="s1",
        user_id="user_001",
    )
    assert first.ok is True and first.cached is False
    assert second.ok is True and second.cached is True
    assert json.loads(second.content)["order"]["product_name"] == "声学降噪耳机 Pro"

    # 2. Knowledge base search returns matching policy
    kb_res = await registry.execute(
        "search_knowledge_base",
        {"query": "退换货和运费险怎么算", "category": "return_policy"},
        session_id="s1",
        user_id="user_001",
    )
    assert kb_res.ok is True
    matches = json.loads(kb_res.content)["matches"]
    assert len(matches) == 1
    assert matches[0]["title"] == "7天无理由退换货政策"

    # 3. Write tool `create_support_ticket` is idempotent (never creates duplicates)
    t1 = await registry.execute(
        "create_support_ticket",
        {
            "user_id": "user_001",
            "order_id": "20261119001",
            "issue_summary": "耳机左耳无声申请质检换新",
            "priority": "high",
        },
        session_id="s1",
        user_id="user_001",
    )
    t2 = await registry.execute(
        "create_support_ticket",
        {
            "user_id": "user_001",
            "order_id": "20261119001",
            "issue_summary": "耳机左耳无声申请质检换新",
            "priority": "high",
        },
        session_id="s1",
        user_id="user_001",
    )
    p1 = json.loads(t1.content)["ticket"]
    p2 = json.loads(t2.content)["ticket"]
    assert p1["ticket_id"] == p2["ticket_id"]
    assert p1["already_existed"] is False
    assert p2["already_existed"] is True

    await redis.aclose()
