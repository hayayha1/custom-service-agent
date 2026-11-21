"""Cross-Encoder HTTP Reranker client (SiliconFlow / Jina / Cohere / DashScope /rerank compatible)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from src.retrieval.models import KBChunkMatch
from src.tools.knowledge import KBDocument

logger = logging.getLogger(__name__)


class APIReranker:
    """Calls a standard `/rerank` HTTP API (e.g. SiliconFlow BAAI/bge-reranker-v2-m3) with graceful fallback."""

    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        api_key: str,
        model: str,
        top_n: int = 4,
        min_score: float = 0.0,
        timeout: float = 8.0,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url.strip()
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.top_n = top_n
        self.min_score = min_score
        self.timeout = timeout

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.base_url and self.api_key and self.model)

    async def rerank(
        self,
        query: str,
        documents: Sequence[KBDocument],
        *,
        top_n: int | None = None,
    ) -> list[KBDocument]:
        if not documents:
            return []
        limit = top_n or self.top_n
        if not self.is_ready:
            return list(documents[:limit])

        payload = {
            "model": self.model,
            "query": query,
            "documents": [f"{d.title}\n{d.content}" for d in documents],
            "top_n": min(limit, len(documents)),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results") or []
            reranked: list[KBDocument] = []
            for item in results:
                idx = item.get("index")
                score = float(item.get("relevance_score", 0.0))
                if not isinstance(idx, int) or idx < 0 or idx >= len(documents):
                    continue
                if score < self.min_score:
                    continue
                orig = documents[idx]
                reranked.append(
                    KBDocument(
                        doc_key=orig.doc_key,
                        category=orig.category,
                        title=orig.title,
                        content=orig.content,
                        score=score,
                        metadata={**orig.metadata, "reranked": True, "reranker_model": self.model},
                    )
                )
            return reranked
        except Exception as exc:
            logger.warning("Reranker API failed (%s); falling back to initial order: %s", self.model, exc)
            return list(documents[:limit])

    async def rerank_chunks(
        self,
        query: str,
        chunks: Sequence[KBChunkMatch],
        *,
        top_n: int | None = None,
    ) -> tuple[list[KBChunkMatch], bool]:
        """Rerank fused KBChunkMatch candidates and filter by `min_score`."""
        if not chunks:
            return [], False
        limit = top_n or self.top_n
        if not self.is_ready:
            return list(chunks[:limit]), False

        payload = {
            "model": self.model,
            "query": query,
            "documents": [c.enriched_content or f"{c.title}\n{c.content}" for c in chunks],
            "top_n": min(limit, len(chunks)),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results") or []
            reranked: list[KBChunkMatch] = []
            for item in results:
                idx = item.get("index")
                score = float(item.get("relevance_score", 0.0))
                if not isinstance(idx, int) or idx < 0 or idx >= len(chunks):
                    continue
                if score < self.min_score:
                    continue
                orig = chunks[idx]
                reranked.append(
                    KBChunkMatch(
                        chunk_key=orig.chunk_key,
                        doc_key=orig.doc_key,
                        category=orig.category,
                        title=orig.title,
                        section_path=orig.section_path,
                        content=orig.content,
                        enriched_content=orig.enriched_content,
                        score=score,
                        dense_score=orig.dense_score,
                        sparse_score=orig.sparse_score,
                        rrf_score=orig.rrf_score,
                        rerank_score=score,
                        metadata={**orig.metadata, "reranked": True, "reranker_model": self.model},
                    )
                )
            return reranked, True
        except Exception as exc:
            logger.warning(
                "Chunk reranker API failed (%s); falling back to RRF ranking: %s",
                self.model,
                exc,
            )
            return list(chunks[:limit]), False
