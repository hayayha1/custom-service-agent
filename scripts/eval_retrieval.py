"""Offline retrieval evaluation script comparing Phase 1 Dense-only vs Phase 2 Rewrite + Hybrid + RRF."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from scripts.seed_kb import SAMPLE_KB_DOCS
from src.kb.ingestion import KnowledgeIngestionPipeline
from src.memory.in_memory import InMemoryKBStore
from src.retrieval.pipeline import HybridRetrievalPipeline
from src.retrieval.rewriter import QueryRewriter
from tests.fakes import HashEmbeddings


async def run_evaluation() -> dict[str, float]:
    embeddings = HashEmbeddings(dim=128)
    store = InMemoryKBStore()
    ingestion = KnowledgeIngestionPipeline(
        kb_store=store,
        embeddings=embeddings,
        chunk_size=260,
        chunk_overlap=40,
    )
    for doc in SAMPLE_KB_DOCS:
        await ingestion.ingest_document(
            doc_key=doc["doc_key"],
            category=doc["category"],
            title=doc["title"],
            raw_content=doc["content"],
            chunk_mode=doc["chunk_mode"],  # type: ignore[arg-type]
        )

    hybrid_pipeline = HybridRetrievalPipeline(
        kb_store=store,
        embeddings=embeddings,
        rewriter=QueryRewriter(enabled=True),
    )

    dataset_path = Path(__file__).resolve().parent.parent / "evals" / "golden_retrieval.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    dense_hits = 0
    hybrid_hits = 0
    hybrid_mrr = 0.0

    for case in cases:
        query = case["query"]
        expected = case["expected_doc_key"]
        slots = case.get("slots") or {}

        # Phase 1 baseline: pure dense vector on raw query
        q_vec = await embeddings.aembed_query(query)
        dense_res = await store.search_chunks_dense(q_vec, top_k=3)
        if any(m.doc_key == expected for m in dense_res[:1]):
            dense_hits += 1

        # Phase 2: Query Rewrite + Dense + Sparse + RRF
        hyb_res = await hybrid_pipeline.retrieve(query, top_k=3, slots=slots)
        for rank, m in enumerate(hyb_res.matches, start=1):
            if m.doc_key == expected:
                if rank == 1:
                    hybrid_hits += 1
                hybrid_mrr += 1.0 / rank
                break

    total = len(cases)
    metrics = {
        "total_cases": float(total),
        "phase1_dense_top1_acc": round(dense_hits / total, 4),
        "phase2_hybrid_top1_acc": round(hybrid_hits / total, 4),
        "phase2_hybrid_mrr_at_3": round(hybrid_mrr / total, 4),
    }
    print("=== Retrieval Offline Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


if __name__ == "__main__":
    asyncio.run(run_evaluation())
