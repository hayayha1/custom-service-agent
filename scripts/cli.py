"""Interactive CLI for manual multi-turn testing against live Redis + Postgres + LLM."""

from __future__ import annotations

import asyncio
import uuid

from src.config import get_settings
from src.db.postgres import configure_windows_event_loop
from src.runtime import create_app_container


async def main() -> None:
    settings = get_settings()
    container = await create_app_container(settings)
    user_id = "user_001"
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    print("=" * 60)
    print("垂直客服 Agent Phase 1 CLI 调试终端")
    print(f"user_id={user_id} | session_id={session_id}")
    print("命令: /new 开启新会话(测试跨会话长期记忆) | /mem 查看缓存 | /archive 归档长期记忆 | /quit 退出")
    print("=" * 60)

    try:
        while True:
            text = (await asyncio.to_thread(input, "\n用户> ")).strip()
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/new":
                await container.agent.archive_session(user_id=user_id, session_id=session_id)
                session_id = f"cli-{uuid.uuid4().hex[:8]}"
                print(f"[系统] 已归档旧会话并切换新会话: session_id={session_id}")
                continue
            if text == "/mem":
                mem = await container.agent.get_session_memory(session_id)
                print(f"[Redis 缓存] {mem}")
                continue
            if text == "/archive":
                saved = await container.agent.archive_session(user_id=user_id, session_id=session_id)
                print(f"[长期记忆归档] saved={saved}")
                continue

            result = await container.agent.chat(
                user_id=user_id,
                session_id=session_id,
                message=text,
            )
            print(f"客服> {result.reply}")
            print(
                f"      [tools={result.tool_calls_used} | cache_hit={result.redis_cache_hit} "
                f"| needs_human={result.needs_human} | slots={result.slots}]"
            )
    finally:
        await container.close()


if __name__ == "__main__":
    configure_windows_event_loop()
    asyncio.run(main())
