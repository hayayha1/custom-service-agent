"""Knowledge base ingestion pipeline: Clean -> Hash Check -> Chunk -> Tokenize -> Embed -> Upsert."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from src.kb.chunker import chunk_document
from src.kb.cleaner import clean_document_text
from src.kb.tokenizer import tokenize_for_fts
from src.memory.long_term import EmbeddingProvider
from src.retrieval.models import KBChunkRecord


@dataclass(slots=True)
class IngestionResult:
    doc_key: str
    title: str
    category: str
    content_hash: str
    chunk_count: int
    pii_redacted_count: int
    noise_removed_count: int
    skipped_unchanged: bool = False


class KnowledgeIngestionPipeline:
    """End-to-end document cleaning, chunking, FTS tokenization, embedding, and incremental storage."""

    def __init__(
        self,
        *,
        kb_store: Any,
        embeddings: EmbeddingProvider,
        chunk_size: int = 450,
        chunk_overlap: int = 70,
        redact_pii: bool = True,
    ) -> None:
        self._store = kb_store
        self._embeddings = embeddings
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._redact_pii = redact_pii

    async def ingest_document(
        self,
        *,
        title: str,
        raw_content: str,
        category: str = "general",
        doc_key: str = "",
        chunk_mode: Literal["auto", "faq", "sop"] = "auto",
        redact_pii: bool | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        force: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        effective_key = doc_key.strip() or f"doc_{uuid.uuid4().hex[:10]}"
        clean_title = title.strip()
        clean_category = category.strip() or "general"
        use_redact = self._redact_pii if redact_pii is None else redact_pii
        use_size = chunk_size or self._chunk_size
        use_overlap = self._chunk_overlap if chunk_overlap is None else chunk_overlap

        # 1. Clean text, strip HTML/watermarks, redact PII, and compute SHA-256 fingerprint
        cleaned = clean_document_text(raw_content, redact_pii=use_redact)

        # 2. Check existing content_hash for idempotent skip
        if not force and hasattr(self._store, "get_document_hash"):
            existing_hash = await self._store.get_document_hash(effective_key)
            if existing_hash and existing_hash == cleaned.content_hash:
                existing_chunks = 1
                if hasattr(self._store, "get_document_chunk_count"):
                    existing_chunks = await self._store.get_document_chunk_count(effective_key) or 1
                return IngestionResult(
                    doc_key=effective_key,
                    title=clean_title,
                    category=clean_category,
                    content_hash=cleaned.content_hash,
                    chunk_count=existing_chunks,
                    pii_redacted_count=cleaned.pii_redacted_count,
                    noise_removed_count=cleaned.noise_removed_count,
                    skipped_unchanged=True,
                )

        # 3. Structure-aware chunking (FAQ atomic pairs vs SOP Markdown heading hierarchy)
        drafts = chunk_document(
            title=clean_title,
            cleaned_text=cleaned.text,
            mode=chunk_mode,
            chunk_size=use_size,
            chunk_overlap=use_overlap,
        )
        if not drafts:
            from src.kb.chunker import KBChunkDraft

            drafts = [
                KBChunkDraft(
                    chunk_index=0,
                    section_path="",
                    content=cleaned.text,
                    enriched_content=f"【{clean_title}】\n{cleaned.text}",
                )
            ]

        # 4. Compute embeddings & FTS tokens for each chunk
        chunk_records: list[KBChunkRecord] = []
        for draft in drafts:
            vec = await self._embeddings.aembed_query(draft.enriched_content)
            tokens = tokenize_for_fts(f"{clean_title} {draft.section_path} {draft.content}")
            chunk_records.append(
                KBChunkRecord(
                    chunk_key=f"{effective_key}#c{draft.chunk_index}",
                    doc_key=effective_key,
                    category=clean_category,
                    title=clean_title,
                    section_path=draft.section_path,
                    chunk_index=draft.chunk_index,
                    content=draft.content,
                    enriched_content=draft.enriched_content,
                    search_tokens=tokens,
                    embedding=vec,
                    metadata={**(metadata or {}), **draft.metadata},
                )
            )

        doc_embedding = chunk_records[0].embedding

        # 5. Persist document & chunks atomically
        if hasattr(self._store, "upsert_document_with_chunks"):
            await self._store.upsert_document_with_chunks(
                doc_key=effective_key,
                category=clean_category,
                title=clean_title,
                content=cleaned.text,
                content_hash=cleaned.content_hash,
                embedding=doc_embedding,
                chunks=chunk_records,
                metadata={
                    **(metadata or {}),
                    "content_hash": cleaned.content_hash,
                    "chunk_count": len(chunk_records),
                    "pii_redacted_count": cleaned.pii_redacted_count,
                },
            )
        else:
            await self._store.upsert_document(
                doc_key=effective_key,
                category=clean_category,
                title=clean_title,
                content=cleaned.text,
                embedding=doc_embedding,
                metadata=metadata,
            )

        return IngestionResult(
            doc_key=effective_key,
            title=clean_title,
            category=clean_category,
            content_hash=cleaned.content_hash,
            chunk_count=len(chunk_records),
            pii_redacted_count=cleaned.pii_redacted_count,
            noise_removed_count=cleaned.noise_removed_count,
            skipped_unchanged=False,
        )
