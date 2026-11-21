"""End-to-end tests for LangGraph Agent Loop, layered memory, loop guard, escalation, and FastAPI."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import fakeredis.aioredis
import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import CustomerServiceAgent, build_agent_graph
from src.api.main import create_app
from src.memory.long_term import LongTermMemory
from src.memory.manager import MemoryManager
from src.memory.short_term import ShortTermMemory
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


async def _make_test_stack(
    llm_handler: Any,
    *,
    max_tool_iterations: int = 3,
    short_term_max_turns: int = 2,
    lock_wait_seconds: float = 1.0,
) -> tuple[CustomerServiceAgent, ScriptedLLM, Any, InMemoryLongTermStore]:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    embeddings = HashEmbeddings()
    lt_store = InMemoryLongTermStore()
    kb_store = InMemoryKBStore()

    doc_vec = await embeddings.aembed_query("7天无理由退换货政策 耳机质保1年")
    await kb_store.upsert_document(
        doc_key="policy_1",
        category="warranty",
        title="数码声学产品质保",
        content="耳机享有1年质保，15天内单耳无声支持免费换新或全额退款。",
        embedding=doc_vec,
    )

    short_term = ShortTermMemory(
        redis_client=redis,
        max_turns=short_term_max_turns,
        ttl_seconds=600,
    )
    long_term = LongTermMemory(
        store=lt_store,
        embeddings=embeddings,
        extractor=RuleBasedMemoryExtractor(),
        top_k=3,
        min_score=0.1,
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
    llm = ScriptedLLM(llm_handler)
    graph = build_agent_graph(
        llm=llm,
        memory_manager=memory_manager,
        tool_registry=registry,
        checkpointer=MemorySaver(),
        max_tool_iterations=max_tool_iterations,
        short_term_max_turns=short_term_max_turns,
        summary_every_n_turns=10,
    )
    agent = CustomerServiceAgent(
        graph=graph,
        redis_client=redis,
        memory_manager=memory_manager,
        lock_ttl_seconds=30,
        lock_wait_seconds=lock_wait_seconds,
    )
    return agent, llm, redis, lt_store


async def test_agent_loop_tool_call_and_slot_survives_beyond_n_turns() -> None:
    def handler(messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            data = json.loads(str(last.content))
            order = data.get("order") or {}
            return AIMessage(
                content=f"已查到订单 {order.get('order_id')}（{order.get('product_name')}），状态：{order.get('status')}。"
            )
        text = str(last.content)
        if "20261119001" in text and "查" in text:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_order_status",
                        "args": {"order_id": "20261119001"},
                        "id": "call_order_1",
                    }
                ],
            )
        return AIMessage(content=f"收到您的问题：{text}")

    agent, llm, redis, _ = await _make_test_stack(handler, short_term_max_turns=2)

    # Turn 1: triggers tool call -> populates slots(order_id, product_name, tracking_no)
    r1 = await agent.chat(
        user_id="user_001",
        session_id="sess-loop-1",
        message="帮我查订单 20261119001",
    )
    assert r1.tool_calls_used == 1
    assert "声学降噪耳机 Pro" in r1.reply
    assert r1.slots["order_id"] == "20261119001"
    assert r1.slots["product_name"] == "声学降噪耳机 Pro"

    # Turns 2, 3, 4 push Turn 1 out of the 2-turn sliding window
    await agent.chat(user_id="user_001", session_id="sess-loop-1", message="顺便问下运费险怎么用？")
    await agent.chat(user_id="user_001", session_id="sess-loop-1", message="包装盒有点压扁了行吗？")
    r4 = await agent.chat(user_id="user_001", session_id="sess-loop-1", message="那帮我退这个订单吧")

    assert r4.redis_cache_hit is True
    assert r4.slots["order_id"] == "20261119001"

    # Verify the prompt sent to LLM on Turn 4:
    # 1) History trimmed to last 2 human turns (Turn 1 text is gone)
    # 2) System Prompt STILL contains pinned `order_id: 20261119001`!
    turn4_prompt = llm.received_prompts[-1]
    assert isinstance(turn4_prompt[0], SystemMessage)
    assert "order_id: 20261119001" in str(turn4_prompt[0].content)
    assert "product_name: 声学降噪耳机 Pro" in str(turn4_prompt[0].content)
    human_msgs_in_prompt = [m for m in turn4_prompt[1:] if isinstance(m, HumanMessage)]
    assert len(human_msgs_in_prompt) == 2
    assert all("20261119001" not in str(m.content) for m in human_msgs_in_prompt)

    # Now simulate Redis expiration -> Checkpointer must seamlessly restore state & backfill Redis
    await agent.memory_manager.short_term.clear_session("sess-loop-1")
    r5 = await agent.chat(user_id="user_001", session_id="sess-loop-1", message="继续处理")
    assert r5.redis_cache_hit is False
    assert r5.slots["order_id"] == "20261119001"

    await redis.aclose()


async def test_loop_guard_trips_cleanly_and_subsequent_turn_remains_valid() -> None:
    call_seq = {"n": 0}

    def endless_tool_handler(messages: list[BaseMessage]) -> AIMessage:
        # On turn 2, reply normally to prove dangling tool_calls were cleanly closed
        if any(isinstance(m, HumanMessage) and "换个问题" in str(m.content) for m in messages):
            return AIMessage(content="好的，请问您有什么新问题？")
        call_seq["n"] += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_order_status",
                    "args": {"order_id": "20261119001"},
                    "id": f"loop_call_{call_seq['n']}",
                }
            ],
        )

    agent, _, redis, _ = await _make_test_stack(endless_tool_handler, max_tool_iterations=2)

    r1 = await agent.chat(
        user_id="user_001",
        session_id="sess-guard",
        message="一直查这个订单",
    )
    assert r1.tool_calls_used == 2
    assert r1.needs_human is True
    assert "人工客服" in r1.reply

    # Flush async consolidation triggered by escalation
    await agent.memory_manager.flush_background_tasks()

    # Verify next turn works cleanly with no protocol mismatch
    r2 = await agent.chat(
        user_id="user_001",
        session_id="sess-guard",
        message="换个问题",
    )
    assert r2.needs_human is False
    assert "新问题" in r2.reply

    await redis.aclose()


async def test_cross_session_long_term_memory_recall_and_escalation() -> None:
    def handler(messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage) and "转人工" in str(last.content):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "transfer_to_human",
                        "args": {
                            "reason": "客户要求人工跟进耳机故障",
                            "context_summary": "订单20261119001耳机左耳无声，客户偏好顺丰",
                        },
                        "id": "call_esc_1",
                    }
                ],
            )
        return AIMessage(content="已记录您的诉求，我们会按您的偏好优先处理。")

    agent, llm, redis, lt_store = await _make_test_stack(handler)

    # Session 1: mentions headset issue & preference for 顺丰 + triggers transfer_to_human -> consolidates L3
    r1 = await agent.chat(
        user_id="user_001",
        session_id="sess-cross-1",
        message="订单 20261119001 耳机左耳无声，以后都发顺丰，帮我转人工",
    )
    assert r1.needs_human is True
    await agent.memory_manager.flush_background_tasks()
    assert "sess-cross-1" in lt_store.episodic

    # Session 2: brand new session_id for same user_001 -> must recall L3 facts & episodic memory!
    await agent.chat(
        user_id="user_001",
        session_id="sess-cross-2",
        message="我之前的耳机左耳无声售后进展怎么样？",
    )
    sys_prompt = str(llm.received_prompts[-1][0].content)
    assert "preferred_carrier: 顺丰" in sys_prompt
    assert "sess-cross-1" in sys_prompt
    assert "耳机左耳无声" in sys_prompt

    await redis.aclose()


async def test_fastapi_endpoints_chat_idempotency_and_lock_429() -> None:
    async def slow_handler(messages: list[BaseMessage]) -> AIMessage:
        if any(isinstance(m, HumanMessage) and "慢查询" in str(m.content) for m in messages):
            await asyncio.sleep(0.25)
        return AIMessage(content="您好，已为您处理完成。")

    agent, _, redis, _ = await _make_test_stack(slow_handler, lock_wait_seconds=0.05)
    app = create_app(agent=agent)

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        # 1. Health check
        h = await client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        # 2. Chat + request_id deduplication
        payload = {
            "user_id": "user_001",
            "session_id": "sess-api-1",
            "message": "帮我看下订单 20261119001",
            "request_id": "req-abc-123",
        }
        resp1 = await client.post("/chat", json=payload)
        assert resp1.status_code == 200
        assert resp1.json()["deduplicated"] is False
        assert resp1.json()["slots"]["order_id"] == "20261119001"

        resp2 = await client.post("/chat", json=payload)
        assert resp2.status_code == 200
        assert resp2.json()["deduplicated"] is True

        # 3. Inspect Redis session memory endpoint
        mem = await client.get("/sessions/sess-api-1/memory")
        assert mem.status_code == 200
        assert mem.json()["redis_cached"] is True
        assert mem.json()["slots"]["order_id"] == "20261119001"

        # 4. Concurrent call on same session while slow call holds lock -> 429 Too Many Requests
        c1, c2 = await asyncio.gather(
            client.post(
                "/chat",
                json={"user_id": "user_001", "session_id": "sess-busy", "message": "慢查询"},
            ),
            client.post(
                "/chat",
                json={"user_id": "user_001", "session_id": "sess-busy", "message": "并发请求"},
            ),
        )
        statuses = sorted([c1.status_code, c2.status_code])
        assert statuses == [200, 429]

        # 5. Archive session into L3 memory
        arch = await client.post("/sessions/sess-api-1/archive", json={"user_id": "user_001"})
        assert arch.status_code == 200
        assert arch.json()["archived"] is True

    await redis.aclose()


async def test_multi_round_tool_calling_accumulates_multiple_orders_and_tickets() -> None:
    round_state = {"step": 0}

    def multi_round_handler(messages: list[BaseMessage]) -> AIMessage:
        if any(isinstance(m, HumanMessage) and "第二个订单呢" in str(m.content) for m in messages):
            return AIMessage(content="您的两个订单都在活跃槽位中已记录。")

        round_state["step"] += 1
        if round_state["step"] == 1:
            # Round 1: parallel get_user_orders + search_knowledge_base
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_user_orders", "args": {"user_id": "user_001"}, "id": "r1_orders"},
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "退货周期"},
                        "id": "r1_kb",
                    },
                ],
            )
        if round_state["step"] == 2:
            # Round 2: create ticket for order 20261119001
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_support_ticket",
                        "args": {
                            "user_id": "user_001",
                            "order_id": "20261119001",
                            "issue_summary": "我要退货",
                            "priority": "normal",
                        },
                        "id": "r2_ticket",
                    }
                ],
            )
        return AIMessage(content="已查到您的2笔订单、核对退货周期，并为20261119001创建退货工单。")

    agent, llm, redis, _ = await _make_test_stack(multi_round_handler, max_tool_iterations=5)

    r1 = await agent.chat(
        user_id="user_001",
        session_id="sess-multi-round",
        message="帮我查我的订单，并检索知识库这类产品退货周期，最后再创建一个工单，内容:我要退货",
    )
    assert r1.tool_calls_used == 2
    assert r1.needs_human is False
    active_orders = r1.slots["active_orders"]
    assert set(active_orders.keys()) == {"20261119001", "20261119002"}
    assert active_orders["20261119001"]["product_name"] == "声学降噪耳机 Pro"
    assert active_orders["20261119002"]["product_name"] == "机械键盘 87键 静音轴"
    assert active_orders["20261119001"]["ticket_id"].startswith("TK-")
    assert len(r1.slots["active_tickets"]) == 1

    # Turn 2: verify System Prompt sees BOTH ongoing orders!
    await agent.chat(user_id="user_001", session_id="sess-multi-round", message="第二个订单呢？")
    sys_prompt = str(llm.received_prompts[-1][0].content)
    assert "20261119001" in sys_prompt
    assert "20261119002" in sys_prompt
    assert "声学降噪耳机 Pro" in sys_prompt
    assert "机械键盘 87键 静音轴" in sys_prompt

    await redis.aclose()
