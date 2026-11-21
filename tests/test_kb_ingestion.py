"""Tests for Phase 2 KB data cleaning, PII redaction, structure-aware chunking, and incremental ingestion."""

from __future__ import annotations

from src.kb.chunker import chunk_document
from src.kb.cleaner import clean_document_text
from src.kb.ingestion import KnowledgeIngestionPipeline
from src.memory.in_memory import InMemoryKBStore
from tests.fakes import HashEmbeddings


def test_clean_document_text_strips_html_watermarks_and_redacts_pii() -> None:
    raw = (
        "<script>alert(1)</script>\n"
        "<div><h1>售后规则</h1><p>测试联系人手机：13812345678，邮箱：support_test@example.com</p></div>\n"
        "第 1 页 / 共 5 页\n"
        "====================\n"
        "身份证号：11010119900307123X，退货地址说明。\n\n\n\n"
        "内部资料 请勿外传"
    )
    cleaned = clean_document_text(raw, redact_pii=True)
    assert "alert(1)" not in cleaned.text
    assert "<div>" not in cleaned.text
    assert "第 1 页" not in cleaned.text
    assert "内部资料" not in cleaned.text
    assert "13812345678" not in cleaned.text
    assert "138****5678" in cleaned.text
    assert "11010119900307123X" not in cleaned.text
    assert "s***@example.com" in cleaned.text
    assert cleaned.pii_redacted_count >= 3
    assert cleaned.noise_removed_count >= 3
    assert len(cleaned.content_hash) == 64


def test_chunk_document_faq_and_sop_modes() -> None:
    faq_text = (
        "Q1: 什么时候可以开发票？\n"
        "A1: 确认收货后在订单详情页申请电子发票。\n\n"
        "Q2: 抬头填错了怎么处理？\n"
        "A2: 联系客服创建红冲重开工单。"
    )
    faq_chunks = chunk_document(title="发票FAQ", cleaned_text=faq_text, mode="auto")
    assert len(faq_chunks) == 2
    assert faq_chunks[0].metadata["chunk_mode"] == "faq"
    assert "Q1:" in faq_chunks[0].content and "A1:" in faq_chunks[0].content
    assert "【发票FAQ | FAQ >" in faq_chunks[0].enriched_content

    sop_text = (
        "# 退换货总则\n"
        "签收7日内商品完好支持无理由退货。\n\n"
        "## 运费规则\n"
        "质量问题运费由平台承担。非质量问题若有运费险可抵扣首重运费，其余情况由买家承担。"
    )
    sop_chunks = chunk_document(
        title="退换货政策",
        cleaned_text=sop_text,
        mode="sop",
        chunk_size=60,
        chunk_overlap=15,
    )
    assert len(sop_chunks) >= 2
    assert any("退换货总则 > 运费规则" in c.section_path for c in sop_chunks)
    assert all(c.enriched_content.startswith("【退换货政策") for c in sop_chunks)


async def test_ingestion_pipeline_incremental_hash_skip_and_chunk_storage() -> None:
    store = InMemoryKBStore()
    embeddings = HashEmbeddings(dim=64)
    pipeline = KnowledgeIngestionPipeline(
        kb_store=store,
        embeddings=embeddings,
        chunk_size=120,
        chunk_overlap=20,
    )

    raw_sop = (
        "# 质保规则\n"
        "耳机享有1年质保，客服测试电话 13987654321。\n\n"
        "## 15天换新\n"
        "签收15天内出现单耳无声或充不进电支持免费换新。"
    )
    first = await pipeline.ingest_document(
        doc_key="policy_audio",
        category="warranty",
        title="声学产品质保",
        raw_content=raw_sop,
        chunk_mode="sop",
    )
    assert first.skipped_unchanged is False
    assert first.chunk_count == 2
    assert first.pii_redacted_count == 1

    # Re-ingesting identical content skips re-embedding via content_hash match
    second = await pipeline.ingest_document(
        doc_key="policy_audio",
        category="warranty",
        title="声学产品质保",
        raw_content=raw_sop,
        chunk_mode="sop",
    )
    assert second.skipped_unchanged is True
    assert second.content_hash == first.content_hash
    assert second.chunk_count == 2
