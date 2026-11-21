"""Windows-safe Uvicorn launcher ensuring SelectorEventLoop is active before psycopg async runs."""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

from src.db.postgres import configure_windows_event_loop


def main() -> None:
    configure_windows_event_loop()
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))

    if sys.platform == "win32":
        # Running uvicorn.Server inside asyncio.run() guarantees SelectorEventLoop on Windows
        # (uvicorn's default CLI --reload spawns ProactorEventLoop which psycopg async rejects).
        config = uvicorn.Config(
            "src.api.main:app",
            host=host,
            port=port,
            loop="asyncio",
            log_level="info",
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    else:
        uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
