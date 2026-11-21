"""Stage 3 Reciprocal Rank Fusion (RRF) combining Dense vector and Sparse FTS ranked lists."""

from __future__ import annotations

from collections.abc import Sequence

from src.retrieval.models import KBChunkMatch


def reciprocal_rank_fusion(
    dense_matches: Sequence[KBChunkMatch],
    sparse_matches: Sequence[KBChunkMatch],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 10,
) -> list[KBChunkMatch]:
    """Fuse Dense and Sparse retrieval lists via weighted RRF:

    score_RRF(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))
    where rank is 1-based.
    """
    by_key: dict[str, KBChunkMatch] = {}
    rrf_scores: dict[str, float] = {}

    for rank_idx, item in enumerate(dense_matches, start=1):
        key = item.chunk_key
        if key not in by_key:
            by_key[key] = KBChunkMatch(
                chunk_key=item.chunk_key,
                doc_key=item.doc_key,
                category=item.category,
                title=item.title,
                section_path=item.section_path,
                content=item.content,
                enriched_content=item.enriched_content,
                dense_score=item.score,
                sparse_score=0.0,
                metadata=dict(item.metadata),
            )
        else:
            by_key[key].dense_score = max(by_key[key].dense_score, item.score)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (dense_weight / (k + rank_idx))

    for rank_idx, item in enumerate(sparse_matches, start=1):
        key = item.chunk_key
        if key not in by_key:
            by_key[key] = KBChunkMatch(
                chunk_key=item.chunk_key,
                doc_key=item.doc_key,
                category=item.category,
                title=item.title,
                section_path=item.section_path,
                content=item.content,
                enriched_content=item.enriched_content,
                dense_score=0.0,
                sparse_score=item.score,
                metadata=dict(item.metadata),
            )
        else:
            by_key[key].sparse_score = max(by_key[key].sparse_score, item.score)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (sparse_weight / (k + rank_idx))

    fused: list[KBChunkMatch] = []
    for key, match in by_key.items():
        r_score = rrf_scores[key]
        match.rrf_score = r_score
        match.score = r_score
        fused.append(match)

    fused.sort(
        key=lambda m: (m.rrf_score, m.dense_score, m.sparse_score),
        reverse=True,
    )
    return fused[:top_k]
