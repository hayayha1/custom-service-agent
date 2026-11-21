"""Async Redis client factory and session-scoped distributed lock."""

from __future__ import annotations

import asyncio
import time
import uuid
from types import TracebackType
from typing import Any, Self

import redis.asyncio as aioredis

RELEASE_LOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class SessionBusyError(RuntimeError):
    """Raised when a concurrent request fails to acquire the session lock within timeout."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session '{session_id}' is currently processing another request. Please retry shortly."
        )


def create_redis_client(redis_url: str) -> aioredis.Redis:
    """Create an async Redis client with UTF-8 string decoding enabled."""
    return aioredis.from_url(redis_url, decode_responses=True)


class RedisSessionLock:
    """Per-session distributed mutex backed by Redis SET NX EX + atomic Lua release."""

    def __init__(
        self,
        redis_client: Any,
        session_id: str,
        ttl_seconds: int = 60,
        wait_seconds: float = 5.0,
        retry_interval: float = 0.05,
        key_prefix: str = "cs:lock:session",
    ) -> None:
        self._redis = redis_client
        self._session_id = session_id
        self._key = f"{key_prefix}:{session_id}"
        self._ttl_seconds = ttl_seconds
        self._wait_seconds = wait_seconds
        self._retry_interval = retry_interval
        self._token = str(uuid.uuid4())
        self._acquired = False

    @property
    def key(self) -> str:
        return self._key

    @property
    def token(self) -> str:
        return self._token

    async def acquire(self) -> bool:
        deadline = time.monotonic() + self._wait_seconds
        while True:
            acquired = await self._redis.set(
                self._key,
                self._token,
                nx=True,
                ex=self._ttl_seconds,
            )
            if acquired:
                self._acquired = True
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._retry_interval)

    async def release(self) -> bool:
        if not self._acquired:
            return False
        self._acquired = False
        released = await self._redis.eval(RELEASE_LOCK_LUA, 1, self._key, self._token)
        return bool(released)

    async def __aenter__(self) -> Self:
        ok = await self.acquire()
        if not ok:
            raise SessionBusyError(self._session_id)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.release()
