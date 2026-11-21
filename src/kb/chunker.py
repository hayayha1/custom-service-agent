"""Structure-aware KB chunker supporting Atomic FAQ pairs and Markdown Heading + Recursive SOP chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_FAQ_SPLIT_RE = re.compile(
    r"(?:^|\n)(?=(?:Q\d*[:：]|问[:：]|【问】|###\s*Q\d*[:：]?))",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*|\n+")


@dataclass(slots=True)
class KBChunkDraft:
    chunk_index: int
    section_path: str
    content: str
    enriched_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _looks_like_faq(text: str) -> bool:
    q_markers = len(re.findall(r"(?:^|\n)\s*(?:Q\d*[:：]|问[:：]|【问】)", text, re.IGNORECASE))
    a_markers = len(re.findall(r"(?:^|\n)\s*(?:A\d*[:：]|答[:：]|【答】)", text, re.IGNORECASE))
    return q_markers >= 1 and a_markers >= 1


def _format_enriched(title: str, section_path: str, content: str) -> str:
    header = f"【{title} | {section_path}】" if section_path else f"【{title}】"
    return f"{header}\n{content.strip()}"


def _split_long_block(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively split a text block along Chinese sentence boundaries with overlap."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s and s.strip()]
    if not sentences:
        sentences = [cleaned]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sent in sentences:
        # Hard-slice single ultra-long sentence without punctuation
        if len(sent) > chunk_size:
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_len = 0
            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, len(sent), step):
                piece = sent[i : i + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if i + chunk_size >= len(sent):
                    break
            continue

        add_len = len(sent) + (1 if current_parts else 0)
        if current_len + add_len <= chunk_size:
            current_parts.append(sent)
            current_len += add_len
        else:
            if current_parts:
                chunks.append(" ".join(current_parts))
            # Build overlap tail from previous parts
            overlap_parts: list[str] = []
            overlap_len = 0
            if chunk_overlap > 0:
                for prev_sent in reversed(current_parts):
                    if overlap_len + len(prev_sent) <= chunk_overlap or not overlap_parts:
                        overlap_parts.insert(0, prev_sent)
                        overlap_len += len(prev_sent) + 1
                    if overlap_len >= chunk_overlap:
                        break
            current_parts = [*overlap_parts, sent]
            current_len = sum(len(p) for p in current_parts) + max(0, len(current_parts) - 1)

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def chunk_document(
    *,
    title: str,
    cleaned_text: str,
    mode: Literal["auto", "faq", "sop"] = "auto",
    chunk_size: int = 450,
    chunk_overlap: int = 70,
) -> list[KBChunkDraft]:
    """Split a cleaned document into context-enriched chunks for hybrid retrieval."""
    text = cleaned_text.strip()
    if not text:
        return []

    effective_mode = mode
    if effective_mode == "auto":
        effective_mode = "faq" if _looks_like_faq(text) else "sop"

    drafts: list[KBChunkDraft] = []

    if effective_mode == "faq":
        raw_blocks = [b.strip() for b in _FAQ_SPLIT_RE.split(text) if b and b.strip()]
        for block in raw_blocks:
            first_line = block.split("\n", 1)[0].strip().lstrip("#").strip()
            section_path = f"FAQ > {first_line[:40]}"
            sub_pieces = (
                [block]
                if len(block) <= chunk_size * 1.5
                else _split_long_block(block, chunk_size, chunk_overlap)
            )
            for piece in sub_pieces:
                idx = len(drafts)
                drafts.append(
                    KBChunkDraft(
                        chunk_index=idx,
                        section_path=section_path,
                        content=piece,
                        enriched_content=_format_enriched(title, section_path, piece),
                        metadata={"chunk_mode": "faq"},
                    )
                )
        return drafts

    # SOP / Markdown heading hierarchy mode
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    current_lines: list[str] = []

    def _current_breadcrumb() -> str:
        return " > ".join(h[1] for h in heading_stack)

    for line in text.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            if current_lines:
                sections.append((_current_breadcrumb(), current_lines))
                current_lines = []
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((_current_breadcrumb(), current_lines))

    for breadcrumb, lines in sections:
        block_text = "\n".join(lines).strip()
        if not block_text:
            continue
        pieces = _split_long_block(block_text, chunk_size, chunk_overlap)
        for piece in pieces:
            idx = len(drafts)
            drafts.append(
                KBChunkDraft(
                    chunk_index=idx,
                    section_path=breadcrumb,
                    content=piece,
                    enriched_content=_format_enriched(title, breadcrumb, piece),
                    metadata={"chunk_mode": "sop"},
                )
            )

    return drafts
