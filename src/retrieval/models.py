"""Models for Phase 2 chunk storage and 4-stage hybrid retrieval diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KBChunkRecord:
    chunk_key: str
    doc_key: str
    category: str
    title: str
    section_path: str
    chunk_index: int
    content: str
    enriched_content: str
    search_tokens: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KBChunkMatch:
    chunk_key: str
    doc_key: str
    category: str
    title: str
    section_path: str
    content: str
    enriched_content: str
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalStageTimings:
    rewrite_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class RetrievalResult:
    original_query: str
    rewritten_query: str
    rewrite_applied: bool
    matches: list[KBChunkMatch]
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    reranked: bool = False
    zero_hit: bool = False
    timings: RetrievalStageTimings = field(default_factory=RetrievalStageTimings)
