"""L1 Working Memory backed by Redis (sliding-window messages List + independent multi-entity slots Hash)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

_ORDER_PATTERNS = [
    re.compile(
        r"(?<![运快物])(?:订单号?|单号|order[_\s-]?id)\s*[:：#]?\s*([A-Za-z0-9_-]{6,32})",
        re.IGNORECASE,
    ),
    re.compile(r"\b(ORD[-_]?[A-Za-z0-9]{4,24})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{9,16})\b"),
]
_PHONE_PATTERN = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_TRACKING_PATTERN = re.compile(
    r"(?:运单号?|快递单号|物流单号|tracking[_\s-]?no)\s*[:：#]?\s*([A-Za-z0-9]{8,24})",
    re.IGNORECASE,
)


def merge_slots(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge active slots without losing multi-entity state (multiple orders, tickets, tracking_nos)."""
    merged: dict[str, Any] = {}
    for k, v in existing.items():
        if isinstance(v, dict):
            merged[k] = {ik: (dict(iv) if isinstance(iv, dict) else iv) for ik, iv in v.items()}
        elif isinstance(v, list):
            merged[k] = list(v)
        elif v is not None:
            merged[k] = v

    for key, val in incoming.items():
        if val is None or (isinstance(val, str) and not val.strip()):
            continue

        if key in ("active_orders", "active_tickets") and isinstance(val, dict):
            bucket = merged.get(key)
            if not isinstance(bucket, dict):
                bucket = {}
            for entity_id, entity_data in val.items():
                eid = str(entity_id)
                prev = bucket.get(eid)
                if isinstance(prev, dict) and isinstance(entity_data, dict):
                    updated = dict(prev)
                    for fk, fv in entity_data.items():
                        if fv is not None and str(fv).strip() != "":
                            updated[fk] = fv
                    bucket[eid] = updated
                elif isinstance(entity_data, dict):
                    bucket[eid] = {
                        fk: fv
                        for fk, fv in entity_data.items()
                        if fv is not None and str(fv).strip() != ""
                    }
                else:
                    bucket[eid] = entity_data
            merged[key] = bucket
        elif key in ("order_ids", "tracking_nos") and isinstance(val, list):
            prev_list = merged.get(key)
            combined: list[str] = list(prev_list) if isinstance(prev_list, list) else []
            for item in val:
                s = str(item).strip()
                if s and s not in combined:
                    combined.append(s)
            merged[key] = combined
        else:
            merged[key] = val

    # Keep `order_ids` synchronized with `active_orders`
    if isinstance(merged.get("active_orders"), dict) and merged["active_orders"]:
        order_ids = list(merged.get("order_ids") or [])
        for oid in merged["active_orders"]:
            if oid not in order_ids:
                order_ids.append(oid)
        merged["order_ids"] = order_ids
        if not merged.get("order_id") and order_ids:
            merged["order_id"] = order_ids[-1]

    return merged


def extract_slots_by_rules(text: str) -> dict[str, Any]:
    """Deterministically extract all high-confidence business entities (supporting multiple orders)."""
    slots: dict[str, Any] = {}
    if not text:
        return slots

    phone_match = _PHONE_PATTERN.search(text)
    if phone_match:
        slots["phone"] = phone_match.group(1)

    tracking_nos: list[str] = []
    for match in _TRACKING_PATTERN.finditer(text):
        tno = match.group(1)
        if tno not in tracking_nos:
            tracking_nos.append(tno)
    if tracking_nos:
        slots["tracking_no"] = tracking_nos[-1]
        slots["tracking_nos"] = tracking_nos

    tracking_set = set(tracking_nos)
    order_ids: list[str] = []
    for pattern in _ORDER_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(1)
            # Avoid misclassifying a China mobile number or matched tracking_no as order_id
            if (
                not _PHONE_PATTERN.fullmatch(candidate)
                and candidate not in tracking_set
                and candidate not in order_ids
            ):
                order_ids.append(candidate)

    if order_ids:
        slots["order_id"] = order_ids[-1]
        slots["order_ids"] = order_ids
        slots["active_orders"] = {oid: {"order_id": oid} for oid in order_ids}

    return slots


def _encode_slot_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _decode_slot_value(raw: str) -> Any:
    stripped = raw.strip()
    if stripped and stripped[0] in ("{", "["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                return parsed
        except ValueError:
            pass
    return raw


def trim_messages_safely(
    messages: Sequence[BaseMessage],
    max_turns: int,
) -> list[BaseMessage]:
    """Keep the most recent `max_turns` user turns without splitting tool-call pairs."""
    if max_turns <= 0 or not messages:
        return []

    human_indices = [
        idx for idx, msg in enumerate(messages) if isinstance(msg, HumanMessage)
    ]
    if not human_indices:
        trimmed = list(messages[-(max_turns * 2) :])
        while trimmed and isinstance(trimmed[0], ToolMessage):
            trimmed.pop(0)
        return trimmed

    if len(human_indices) <= max_turns:
        start_idx = human_indices[0]
    else:
        start_idx = human_indices[-max_turns]

    trimmed = list(messages[start_idx:])
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed.pop(0)
    return trimmed


def serialize_message(message: BaseMessage) -> str:
    """Serialize a LangChain message to compact JSON for Redis List storage."""
    payload: dict[str, Any] = {
        "type": message.type,
        "content": (
            message.content
            if isinstance(message.content, str)
            else json.dumps(message.content, ensure_ascii=False)
        ),
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
    return json.dumps(payload, ensure_ascii=False)


def deserialize_message(raw: str) -> BaseMessage:
    """Deserialize a Redis JSON payload back into a LangChain BaseMessage."""
    data = json.loads(raw)
    msg_type = data.get("type", "human")
    content = data.get("content", "")
    if msg_type == "human":
        return HumanMessage(content=content)
    if msg_type == "ai":
        return AIMessage(content=content, tool_calls=data.get("tool_calls", []))
    if msg_type == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name"),
        )
    if msg_type == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


class ShortTermMemory:
    """Redis-backed L1 working memory with physically separated messages List and slots Hash."""

    def __init__(
        self,
        redis_client: Any,
        max_turns: int = 10,
        ttl_seconds: int = 1800,
        key_prefix: str = "cs:session",
    ) -> None:
        self._redis = redis_client
        self._max_turns = max_turns
        self._max_raw_messages = max_turns * 6
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    def messages_key(self, session_id: str) -> str:
        return f"{self._key_prefix}:{session_id}:messages"

    def slots_key(self, session_id: str) -> str:
        return f"{self._key_prefix}:{session_id}:slots"

    async def has_session_cache(self, session_id: str) -> bool:
        msg_exists = await self._redis.exists(self.messages_key(session_id))
        slots_exists = await self._redis.exists(self.slots_key(session_id))
        return bool(msg_exists or slots_exists)

    async def get_messages(self, session_id: str) -> list[BaseMessage] | None:
        key = self.messages_key(session_id)
        exists = await self._redis.exists(key)
        if not exists:
            return None
        raw_items = await self._redis.lrange(key, 0, -1)
        messages = [deserialize_message(item) for item in raw_items]
        return trim_messages_safely(messages, self._max_turns)

    async def get_slots(self, session_id: str) -> dict[str, Any]:
        raw = await self._redis.hgetall(self.slots_key(session_id))
        return {str(k): _decode_slot_value(str(v)) for k, v in raw.items()}

    async def replace_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        """Replace the sliding-window message list and refresh TTL."""
        key = self.messages_key(session_id)
        trimmed = trim_messages_safely(messages, self._max_turns)
        serialized = [serialize_message(m) for m in trimmed]
        pipe = self._redis.pipeline()
        pipe.delete(key)
        if serialized:
            pipe.rpush(key, *serialized)
            pipe.ltrim(key, -self._max_raw_messages, -1)
            pipe.expire(key, self._ttl_seconds)
        await pipe.execute()

    async def append_messages(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        """Append new messages to Redis List with LTRIM window bound and TTL refresh."""
        if not messages:
            return
        key = self.messages_key(session_id)
        serialized = [serialize_message(m) for m in messages]
        pipe = self._redis.pipeline()
        pipe.rpush(key, *serialized)
        pipe.ltrim(key, -self._max_raw_messages, -1)
        pipe.expire(key, self._ttl_seconds)
        await pipe.execute()

    async def update_slots(self, session_id: str, slots: dict[str, Any]) -> None:
        """Upsert active slots (scalars & JSON-encoded multi-entity dicts/lists) in Redis Hash."""
        clean_slots: dict[str, str] = {}
        for k, v in slots.items():
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            if isinstance(v, (dict, list)) and not v:
                continue
            clean_slots[str(k)] = _encode_slot_value(v)

        if not clean_slots:
            return
        key = self.slots_key(session_id)
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=clean_slots)
        pipe.expire(key, self._ttl_seconds)
        await pipe.execute()

    async def delete_slots(self, session_id: str, keys: Sequence[str]) -> None:
        if not keys:
            return
        await self._redis.hdel(self.slots_key(session_id), *keys)

    async def refresh_ttl(self, session_id: str) -> None:
        pipe = self._redis.pipeline()
        pipe.expire(self.messages_key(session_id), self._ttl_seconds)
        pipe.expire(self.slots_key(session_id), self._ttl_seconds)
        await pipe.execute()

    async def clear_session(self, session_id: str) -> None:
        await self._redis.delete(self.messages_key(session_id), self.slots_key(session_id))
