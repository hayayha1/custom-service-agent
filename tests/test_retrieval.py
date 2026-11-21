"""Tests for Phase 2 4-stage Hybrid Retrieval: Query Rewriting, Dense+Sparse, RRF, and Rerank."""

from __future__ import annotations

from collections.abc import Sequence

from src.kb.ingestion import KnowledgeIngestionPipeline
from src.memory.in_memory import InMemoryKBStore
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import KBChunkMatch
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.retrieval.rewriter import QueryRewriter
from tests.fakes import HashEmbeddings


async def test_query_rewriter_resolves_slot_pronouns_and_colloquialisms() -> None:
    rewriter = QueryRewriter(enabled=True)
    slots = {
        "active_orders": {
            "20261119001": {
                "order_id": "20261119001",
                "product_name": "声学降噪耳机 Pro",
            }
        }
    }
    rewritten, applied = await rewriter.rewrite("它没声音坏了怎么退钱，邮费谁掏？", slots=slots)
    assert applied is True
    assert "声学降噪耳机 Pro" in rewritten
    assert "质保" in rewritten
    assert "运费险" in rewritten


def test_reciprocal_rank_fusion_combines_dense_and_sparse_ranks() -> None:
    d1 = KBChunkMatch(
        chunk_key="doc1#c0",
        doc_key="doc1",
        category="warranty",
        title="Doc 1",
        section_path="S1",
        content="c1",
        enriched_content="c1",
        score=0.9,
    )
    d2 = KBChunkMatch(
        chunk_key="doc2#c0",
        doc_key="doc2",
        category="warranty",
        title="Doc 2",
        section_path="S2",
        content="c2",
        enriched_content="c2",
        score=0.8,
    )
    s1 = KBChunkMatch(
        chunk_key="doc2#c0",
        doc_key="doc2",
        category="warranty",
        title="Doc 2",
        section_path="S2",
        content="c2",
        enriched_content="c2",
        score=3.5,
    )
    s2 = KBChunkMatch(
        chunk_key="doc3#c0",
        doc_key="doc3",
        category="warranty",
        title="Doc 3",
        section_path="S3",
        content="c3",
        enriched_content="c3",
        score=2.1,
    )

    fused = reciprocal_rank_fusion([d1, d2], [s1, s2], k=60, top_k=3)
    # doc2 appears in BOTH dense (rank 2) and sparse (rank 1), so its RRF score is highest
    assert fused[0].chunk_key == "doc2#c0"
    expected_doc2_rrf = 1.0 / (60 + 2) + 1.0 / (60 + 1)
    assert abs(fused[0].rrf_score - expected_doc2_rrf) < 1e-8
    assert fused[0].dense_score == 0.8
    assert fused[0].sparse_score == 3.5


class _StubReranker:
    def __init__(self, *, min_score: float = 0.5, fail: bool = False) -> None:
        self.enabled = True
        self.min_score = min_score
        self.fail = fail

    async def rerank_chunks(
        self,
        query: str,
        chunks: Sequence[KBChunkMatch],
        *,
        top_n: int | None = None,
    ) -> tuple[list[KBChunkMatch], bool]:
        if self.fail:
            # Simulate APIReranker graceful fallback on network/timeout error
            return list(chunks[: (top_n or 4)]), False

        scored: list[KBChunkMatch] = []
        for ch in chunks:
            score = 0.92 if "刻字保温杯" in ch.content or "单耳无声" in ch.content else 0.15
            if score >= self.min_score:
                ch.rerank_score = score
                ch.score = score
                scored.append(ch)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: (top_n or 4)], True


async def test_hybrid_retrieval_pipeline_end_to_end_and_rerank_threshold() -> None:
    store = InMemoryKBStore()
    embeddings = HashEmbeddings(dim=64)
    ingestion = KnowledgeIngestionPipeline(kb_store=store, embeddings=embeddings, chunk_size=200)

    await ingestion.ingest_document(
        doc_key="policy_return",
        category="return_policy",
        title="7天无理由退换货政策",
        raw_content="# 退货规则\n普通商品支持7天无理由退货。\n## 定制限制\n定制刻字保温杯非质量问题不支持7天无理由退货。",
    )
    await ingestion.ingest_document(
        doc_key="policy_audio",
        category="warranty",
        title="声学产品质保",
        raw_content="# 质保\n声学降噪耳机 Pro 签收15天内单耳无声支持免费换新。",
    )

    pipeline = HybridRetrievalPipeline(
        kb_store=store,
        embeddings=embeddings,
        rewriter=QueryRewriter(enabled=True),
        reranker=_StubReranker(min_score=0.5),
    )

    res = await pipeline.retrieve(
        "它单耳无声坏了能换新吗",
        top_k=3,
        slots={"product_name": "声学降噪耳机 Pro"},
    )
    assert res.rewrite_applied is True
    assert "声学降噪耳机 Pro" in res.rewritten_query
    assert res.dense_count >= 1
    assert res.sparse_count >= 1
    assert res.reranked is True
    assert res.zero_hit is False
    assert res.matches[0].doc_key == "policy_audio"
    assert res.matches[0].rerank_score == 0.92

    # Low-confidence out-of-domain query filtered out by min_score=0.99 -> zero_hit=True
    strict_pipeline = HybridRetrievalPipeline(
        kb_store=store,
        embeddings=embeddings,
        reranker=_StubReranker(min_score=0.99),
    )
    zero_res = await strict_pipeline.retrieve("今天天气怎么样", top_k=3)
    assert zero_res.zero_hit is True
    assert zero_res.matches == []
