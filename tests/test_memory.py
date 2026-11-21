"""Tests for L1 Redis working memory, L2 fallback, L3 pgvector long-term memory, and Redis Lua lock."""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.db.redis import RedisSessionLock, SessionBusyError
from src.memory.long_term import LongTermMemory
from src.memory.manager import MemoryManager
from src.memory.short_term import (
    ShortTermMemory,
    extract_slots_by_rules,
    merge_slots,
    trim_messages_safely,
)
from tests.fakes import HashEmbeddings, InMemoryLongTermStore, RuleBasedMemoryExtractor


def test_extract_slots_by_rules_ignores_phone_as_order_id() -> None:
    slots = extract_slots_by_rules("我手机号是13812345678，运单号 SF1234567890")
    assert slots["phone"] == "13812345678"
    assert slots["tracking_no"] == "SF1234567890"
    assert "order_id" not in slots

    slots2 = extract_slots_by_rules("帮我查一下订单号 20261119001 的状态")
    assert slots2["order_id"] == "20261119001"
    assert slots2["order_ids"] == ["20261119001"]


async def test_multi_entity_active_slots_preserve_multiple_orders_in_redis() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    stm = ShortTermMemory(redis_client=redis, max_turns=2, ttl_seconds=600)

    extracted = extract_slots_by_rules("我有两个订单 20261119001 和 20261119002 都在进行中")
    assert extracted["order_ids"] == ["20261119001", "20261119002"]
    assert set(extracted["active_orders"].keys()) == {"20261119001", "20261119002"}

    # Merge detail for order 1 then order 2 — neither should overwrite the other
    merged = merge_slots(
        extracted,
        {
            "active_orders": {
                "20261119001": {"order_id": "20261119001", "product_name": "声学降噪耳机 Pro"}
            }
        },
    )
    merged = merge_slots(
        merged,
        {
            "active_orders": {
                "20261119002": {"order_id": "20261119002", "product_name": "机械键盘 87键 静音轴"}
            }
        },
    )
    await stm.update_slots("sess-multi-slots", merged)

    restored = await stm.get_slots("sess-multi-slots")
    assert isinstance(restored["active_orders"], dict)
    assert restored["active_orders"]["20261119001"]["product_name"] == "声学降噪耳机 Pro"
    assert restored["active_orders"]["20261119002"]["product_name"] == "机械键盘 87键 静音轴"
    assert restored["order_ids"] == ["20261119001", "20261119002"]

    await redis.aclose()


def test_trim_messages_safely_keeps_tool_pairs_intact() -> None:
    msgs = [
        HumanMessage(content="turn 1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "query_order_status", "args": {"order_id": "20261119001"}, "id": "c1"}],
        ),
        ToolMessage(content='{"ok": true}', tool_call_id="c1", name="query_order_status"),
        AIMessage(content="answer 1"),
        HumanMessage(content="turn 2"),
        AIMessage(content="answer 2"),
    ]
    trimmed = trim_messages_safely(msgs, max_turns=1)
    assert len(trimmed) == 2
    assert isinstance(trimmed[0], HumanMessage)
    assert trimmed[0].content == "turn 2"

    # Even if raw list starts with orphan ToolMessage
    orphan_trimmed = trim_messages_safely([ToolMessage(content="x", tool_call_id="c0"), HumanMessage(content="hi")], max_turns=1)
    assert isinstance(orphan_trimmed[0], HumanMessage)


async def test_l1_messages_window_and_slots_physically_separated() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    stm = ShortTermMemory(redis_client=redis, max_turns=2, ttl_seconds=600)
    session_id = "sess-sep-1"

    # Turn 1 sets an active slot `order_id`
    await stm.update_slots(session_id, {"order_id": "20261119001", "phone": "13800138000"})

    # Simulate 6 full turns (far exceeding max_turns=2)
    all_msgs = []
    for i in range(1, 7):
        all_msgs.append(HumanMessage(content=f"第{i}轮用户消息"))
        all_msgs.append(AIMessage(content=f"第{i}轮客服回复"))
    await stm.replace_messages(session_id, all_msgs)

    # Messages List only keeps last 2 turns (4 messages)
    cached_msgs = await stm.get_messages(session_id)
    assert cached_msgs is not None
    assert len(cached_msgs) == 4
    assert cached_msgs[0].content == "第5轮用户消息"
    assert cached_msgs[-1].content == "第6轮客服回复"

    # Independent Slots Hash STILL retains turn-1 order_id & phone!
    cached_slots = await stm.get_slots(session_id)
    assert cached_slots["order_id"] == "20261119001"
    assert cached_slots["phone"] == "13800138000"

    await redis.aclose()


async def test_redis_cache_miss_recovers_from_checkpointer_and_backfills() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    stm = ShortTermMemory(redis_client=redis, max_turns=3, ttl_seconds=600)
    ltm = LongTermMemory(
        store=InMemoryLongTermStore(),
        embeddings=HashEmbeddings(),
        extractor=RuleBasedMemoryExtractor(),
    )
    mgr = MemoryManager(short_term=stm, long_term=ltm)

    # Redis is empty (expired), but LangGraph Checkpointer holds prior state
    checkpoint_msgs = [
        HumanMessage(content="之前问的订单 20261119001"),
        AIMessage(content="已为您查到订单 20261119001"),
    ]
    checkpoint_slots = {"order_id": "20261119001", "product_name": "声学降噪耳机 Pro"}

    ctx = await mgr.load_for_turn(
        user_id="user_001",
        session_id="sess-expired",
        latest_user_text="它现在能退吗？",
        checkpoint_messages=checkpoint_msgs,
        checkpoint_slots=checkpoint_slots,
    )

    assert ctx.redis_cache_hit is False
    assert ctx.slots["order_id"] == "20261119001"
    assert ctx.slots["product_name"] == "声学降噪耳机 Pro"

    # Redis should now be warmed / backfilled
    assert await stm.has_session_cache("sess-expired") is True
    assert (await stm.get_slots("sess-expired"))["order_id"] == "20261119001"
    backfilled_msgs = await stm.get_messages("sess-expired")
    assert backfilled_msgs is not None and len(backfilled_msgs) == 2

    await redis.aclose()


async def test_l3_long_term_memory_chitchat_filter_and_fact_conflict_upsert() -> None:
    store = InMemoryLongTermStore()
    ltm = LongTermMemory(
        store=store,
        embeddings=HashEmbeddings(),
        extractor=RuleBasedMemoryExtractor(),
        top_k=3,
        min_score=0.1,
    )

    # 1. Pure chitchat should be skipped
    saved_chitchat = await ltm.consolidate(
        user_id="user_001",
        session_id="s-chat",
        messages=[HumanMessage(content="你好"), AIMessage(content="您好，请问有什么可以帮您？")],
        slots={},
    )
    assert saved_chitchat is False
    assert len(store.episodic) == 0

    # 2. First meaningful session saves carrier=中通 and episodic summary
    saved_s1 = await ltm.consolidate(
        user_id="user_001",
        session_id="s-1",
        messages=[HumanMessage(content="我的耳机左耳无声，以前都发中通，帮我处理")],
        slots={"order_id": "20261119001"},
    )
    assert saved_s1 is True

    # 3. Second session updates preference to 顺丰 -> should Upsert (no conflicting duplicates)
    saved_s2 = await ltm.consolidate(
        user_id="user_001",
        session_id="s-2",
        messages=[HumanMessage(content="以后快递全部改发顺丰，并且电话联系我")],
        slots={},
    )
    assert saved_s2 is True

    facts, episodes = await ltm.recall(user_id="user_001", query="耳机左耳无声售后")
    fact_map = {f.fact_key: f.fact_value for f in facts}
    assert fact_map["preferred_carrier"] == "顺丰"
    assert fact_map["preferred_contact"] == "电话联系"
    assert len([f for f in facts if f.fact_key == "preferred_carrier"]) == 1
    assert any("耳机左耳无声" in ep.summary for ep in episodes)

    # 4. User isolation: another user cannot recall user_001's memories
    other_facts, other_eps = await ltm.recall(user_id="user_999", query="耳机左耳无声售后")
    assert other_facts == []
    assert other_eps == []


async def test_redis_session_lock_mutual_exclusion_and_safe_lua_release() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    lock1 = RedisSessionLock(redis, "sess-lock", ttl_seconds=30, wait_seconds=0.1)
    lock2 = RedisSessionLock(redis, "sess-lock", ttl_seconds=30, wait_seconds=0.15)

    async with lock1:
        with pytest.raises(SessionBusyError):
            async with lock2:
                pass

    # After lock1 exits, lock3 can acquire immediately
    lock3 = RedisSessionLock(redis, "sess-lock", ttl_seconds=30, wait_seconds=0.1)
    assert await lock3.acquire() is True

    # Simulate token mismatch (e.g. lock1 expired and lock3 owns the key): wrong token must NOT delete key
    lock_rogue = RedisSessionLock(redis, "sess-lock", ttl_seconds=30, wait_seconds=0.1)
    lock_rogue._acquired = True  # force attempted release with wrong token
    assert await lock_rogue.release() is False
    assert await redis.get(lock3.key) == lock3.token

    assert await lock3.release() is True
    assert await redis.get(lock3.key) is None
    await redis.aclose()


async def test_redis_session_lock_serializes_concurrent_requests() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    events: list[str] = []

    async def worker(name: str) -> None:
        async with RedisSessionLock(redis, "sess-serial", ttl_seconds=30, wait_seconds=2.0):
            events.append(f"{name}:enter")
            await asyncio.sleep(0.08)
            events.append(f"{name}:exit")

    await asyncio.gather(worker("A"), worker("B"))
    # Strict non-interleaving execution order
    assert events in (
        ["A:enter", "A:exit", "B:enter", "B:exit"],
        ["B:enter", "B:exit", "A:enter", "A:exit"],
    )
    await redis.aclose()
