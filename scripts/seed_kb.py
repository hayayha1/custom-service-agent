"""Seed vertical e-commerce SOP & FAQ documents through the Phase 2 KnowledgeIngestionPipeline."""

from __future__ import annotations

import asyncio

from langchain_openai import OpenAIEmbeddings

from src.config import get_settings
from src.db.postgres import configure_windows_event_loop, create_postgres_pool, init_db
from src.kb.ingestion import KnowledgeIngestionPipeline
from src.tools.knowledge import PostgresKnowledgeBaseStore

SAMPLE_KB_DOCS = [
    {
        "doc_key": "policy_return_7day",
        "category": "return_policy",
        "title": "7天无理由退换货政策",
        "chunk_mode": "sop",
        "content": (
            "# 退换货总则\n"
            "自签收次日起7日内，商品完好、配件齐全且不影响二次销售的，支持7天无理由退换货。\n\n"
            "## 不支持7天无理由的商品\n"
            "私人定制类商品（如定制刻字保温杯、定制尺寸商品）、拆封后影响人身卫生的贴身商品，非质量问题不支持7天无理由退货。\n\n"
            "## 退换货运费承担规则\n"
            "1. 质量问题或发错货：退换货运费由平台全额承担。\n"
            "2. 非质量问题：若订单赠送运费险，退货时可由运费险自动抵扣首重运费；无运费险则由买家承担寄回运费。"
        ),
    },
    {
        "doc_key": "policy_warranty_audio",
        "category": "warranty",
        "title": "数码声学产品质保与故障处理",
        "chunk_mode": "sop",
        "content": (
            "# 官方质保周期\n"
            "声学降噪耳机 Pro、蓝牙音箱、机械键盘 87键 静音轴等数码外设均享有1年官方质保（型号 X100 / Pro 系列通用）。\n\n"
            "## 15天内硬件故障换新\n"
            "签收15天内出现单耳无声、杂音断连、充不进电、按键失灵等非人为硬件故障，支持免费换新或全额退款，往返运费由平台承担。\n\n"
            "## 15天至1年内质保维修\n"
            "签收15天以上至1年内出现非人为故障，提供免费寄修或换良服务。客服核实订单号后需创建售后质检工单，专员将在24小时内联系安排顺丰上门取件。"
        ),
    },
    {
        "doc_key": "policy_shipping_delay",
        "category": "shipping",
        "title": "发货时效与物流异常赔付规则",
        "chunk_mode": "sop",
        "content": (
            "# 发货时效承诺\n"
            "常规现货订单付款后48小时内发货，默认合作顺丰速运、中通快递与京东物流。\n\n"
            "## 物流停滞与丢件破损处理\n"
            "发货后若物流轨迹超过72小时未更新、或签收时发现包裹外包装破损/缺件/丢件，客服应立即为用户创建高优先级（high）物流异常工单。\n"
            "经核实确认丢件或破损的，支持24小时内免费补发或全额退款，并额外补偿20元无门槛优惠券。"
        ),
    },
    {
        "doc_key": "policy_invoice_rules",
        "category": "invoice",
        "title": "电子发票开具与换开FAQ",
        "chunk_mode": "faq",
        "content": (
            "Q1: 什么时候可以申请开具发票？\n"
            "A1: 订单确认收货后，支持在App「订单详情-申请开票」自助申请增值税电子普通发票（支持个人或企业抬头），一般10分钟内发送至邮箱。\n\n"
            "Q2: 需要开具增值税专用发票或者发票抬头填错了怎么换开？\n"
            "A2: 如需开具增值税专用发票，或已开发票抬头/税号填错需要红冲重开，请提供订单号与正确开票信息，由客服创建工单转财务团队在3个工作日内处理。"
        ),
    },
]


async def main() -> None:
    settings = get_settings()
    pool = await create_postgres_pool(settings.postgres_dsn)
    try:
        await init_db(pool, embedding_dim=settings.embedding_dim)
        emb_kwargs = {
            "model": settings.embedding_model,
            "api_key": settings.effective_embedding_api_key,
            "base_url": settings.embedding_base_url,
            "check_embedding_ctx_length": False,
        }
        if settings.embedding_model.startswith("text-embedding-3"):
            emb_kwargs["dimensions"] = settings.embedding_dim
        embeddings = OpenAIEmbeddings(**emb_kwargs)
        store = PostgresKnowledgeBaseStore(pool)
        pipeline = KnowledgeIngestionPipeline(
            kb_store=store,
            embeddings=embeddings,
            chunk_size=settings.kb_chunk_size,
            chunk_overlap=settings.kb_chunk_overlap,
            redact_pii=settings.kb_redact_pii,
        )
        for doc in SAMPLE_KB_DOCS:
            res = await pipeline.ingest_document(
                doc_key=doc["doc_key"],
                category=doc["category"],
                title=doc["title"],
                raw_content=doc["content"],
                chunk_mode=doc["chunk_mode"],  # type: ignore[arg-type]
                force=True,
                metadata={"source": "seed_kb"},
            )
            print(
                f"[OK] Ingested {res.doc_key} ({res.title}): "
                f"chunks={res.chunk_count}, hash={res.content_hash[:10]}"
            )
    finally:
        await pool.close()


if __name__ == "__main__":
    configure_windows_event_loop()
    asyncio.run(main())
