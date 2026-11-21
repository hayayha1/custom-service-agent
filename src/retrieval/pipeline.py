"""4-stage Hybrid Retrieval Pipeline: Query Rewrite -> Dense+Sparse -> RRF Fusion -> Rerank."""

from __future__ import annotations

import asyncio
import contextvars
import time
from typing import Any

from src.memory.long_term import EmbeddingProvider
from src.observability.tracer import get_current_trace, start_observed_span
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import (
    KBChunkMatch,
    RetrievalResult,
    RetrievalStageTimings,
)
from src.retrieval.rewriter import QueryRewriter

_active_slots_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "cs_active_retrieval_slots", default=None
)


def set_active_retrieval_slots(slots: dict[str, Any] | None) -> contextvars.Token[dict[str, Any] | None]:
    return _active_slots_var.set(slots)


def reset_active_retrieval_slots(token: contextvars.Token[dict[str, Any] | None]) -> None:
    _active_slots_var.reset(token)


def get_active_retrieval_slots() -> dict[str, Any]:
    return _active_slots_var.get() or {}


class HybridRetrievalPipeline:
    """Executes Query Rewrite, parallel Dense+Sparse search, RRF fusion, and Cross-Encoder Rerank."""

    def __init__(
        self,
        *,
        kb_store: Any,
        embeddings: EmbeddingProvider,
        rewriter: QueryRewriter | None = None,
        reranker: Any | None = None,
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        rrf_k: int = 60,
        rrf_top_k: int = 10,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self._store = kb_store
        self._embeddings = embeddings
        self._rewriter = rewriter or QueryRewriter(enabled=True)
        self._reranker = reranker
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._rrf_k = rrf_k
        self._rrf_top_k = rrf_top_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    async def retrieve(
        self,
        query: str,
        *,
        category: str | None = None,
        top_k: int = 4,
        slots: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        t_start = time.perf_counter()
        effective_slots = slots if slots is not None else get_active_retrieval_slots()

        # Stage 1: Context-aware Query Rewriting
        t0 = time.perf_counter()
        with start_observed_span(
            span_type="retrieval_rewrite",
            name="query_rewrite",
            input_summary={"query": query, "has_slots": bool(effective_slots)},
        ) as sp_rw:
            rewritten_query, rewrite_applied = await self._rewriter.rewrite(
                query, slots=effective_slots
            )
            if sp_rw is not None:
                sp_rw.set_output(
                    {"rewritten_query": rewritten_query, "rewrite_applied": rewrite_applied}
                )
        rewrite_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 2: Parallel Dense (pgvector) + Sparse (Postgres FTS / jieba) Retrieval
        async def _run_dense() -> tuple[list[KBChunkMatch], float]:
            td = time.perf_counter()
            with start_observed_span(
                span_type="retrieval_dense",
                name="pgvector_dense_search",
                input_summary={"query": rewritten_query, "category": category, "top_k": self._dense_top_k},
            ) as sp_d:
                query_vec = await self._embeddings.aembed_query(rewritten_query)
                if hasattr(self._store, "search_chunks_dense"):
                    res = await self._store.search_chunks_dense(
                        query_vec, category=category, top_k=self._dense_top_k
                    )
                else:
                    docs = await self._store.search(
                        query_vec, category=category, top_k=self._dense_top_k
                    )
                    res = [
                        KBChunkMatch(
                            chunk_key=f"{d.doc_key}#c0",
                            doc_key=d.doc_key,
                            category=d.category,
                            title=d.title,
                            section_path="",
                            content=d.content,
                            enriched_content=f"【{d.title}】\n{d.content}",
                            score=d.score,
                            dense_score=d.score,
                            metadata=d.metadata,
                        )
                        for d in docs
                    ]
                if sp_d is not None:
                    sp_d.set_output({"count": len(res)})
            return res, round((time.perf_counter() - td) * 1000.0, 2)

        async def _run_sparse() -> tuple[list[KBChunkMatch], float]:
            ts = time.perf_counter()
            with start_observed_span(
                span_type="retrieval_sparse",
                name="postgres_fts_sparse_search",
                input_summary={"query": rewritten_query, "category": category, "top_k": self._sparse_top_k},
            ) as sp_s:
                if hasattr(self._store, "search_chunks_sparse"):
                    res = await self._store.search_chunks_sparse(
                        rewritten_query, category=category, top_k=self._sparse_top_k
                    )
                else:
                    res = []
                if sp_s is not None:
                    sp_s.set_output({"count": len(res)})
            return res, round((time.perf_counter() - ts) * 1000.0, 2)

        (dense_matches, dense_ms), (sparse_matches, sparse_ms) = await asyncio.gather(
            _run_dense(), _run_sparse()
        )

        # Stage 3: Reciprocal Rank Fusion (RRF)
        t_rrf = time.perf_counter()
        with start_observed_span(
            span_type="retrieval_rrf",
            name="rrf_fusion",
            input_summary={"dense_count": len(dense_matches), "sparse_count": len(sparse_matches)},
        ) as sp_rrf:
            fused = reciprocal_rank_fusion(
                dense_matches,
                sparse_matches,
                k=self._rrf_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                top_k=max(self._rrf_top_k, top_k),
            )
            if sp_rrf is not None:
                sp_rrf.set_output({"fused_count": len(fused)})
        rrf_ms = round((time.perf_counter() - t_rrf) * 1000.0, 2)

        # Stage 4: Cross-Encoder Rerank + Low-Confidence Threshold Filter
        t_rerank = time.perf_counter()
        did_rerank = False
        with start_observed_span(
            span_type="retrieval_rerank",
            name="cross_encoder_rerank",
            input_summary={"candidates": len(fused), "top_k": top_k},
        ) as sp_rk:
            if fused and self._reranker is not None and hasattr(self._reranker, "rerank_chunks"):
                final_matches, did_rerank = await self._reranker.rerank_chunks(
                    rewritten_query, fused, top_n=top_k
                )
            else:
                final_matches = fused[:top_k]

            zero_hit = len(final_matches) == 0
            top1_score = final_matches[0].score if final_matches else 0.0

            if sp_rk is not None:
                sp_rk.set_output(
                    {
                        "final_count": len(final_matches),
                        "reranked": did_rerank,
                        "top1_score": round(top1_score, 4),
                    }
                )
                sp_rk.set_metadata(
                    {"zero_hit": zero_hit, "top1_score": round(top1_score, 4), "reranked": did_rerank}
                )

            trace_ctx = get_current_trace()
            if trace_ctx is not None:
                trace_ctx.zero_hit = trace_ctx.zero_hit or zero_hit
                if final_matches:
                    trace_ctx.rerank_top1_score = round(top1_score, 4)

        rerank_ms = round((time.perf_counter() - t_rerank) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return RetrievalResult(
            original_query=query,
            rewritten_query=rewritten_query,
            rewrite_applied=rewrite_applied,
            matches=final_matches,
            dense_count=len(dense_matches),
            sparse_count=len(sparse_matches),
            fused_count=len(fused),
            reranked=did_rerank,
            zero_hit=zero_hit,
            timings=RetrievalStageTimings(
                rewrite_ms=rewrite_ms,
                dense_ms=dense_ms,
                sparse_ms=sparse_ms,
                rrf_ms=rrf_ms,
                rerank_ms=rerank_ms,
                total_ms=total_ms,
            ),
        )
