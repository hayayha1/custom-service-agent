"""Stage 1 Query Rewriter: pronoun resolution from active slots + colloquial synonym expansion."""

from __future__ import annotations

import re
from typing import Any

_PRONOUN_RE = re.compile(r"(?:它|这个|这件|那件|这款|那款|该商品|这个商品|这个订单)")

# Domain colloquialisms -> standard retrieval keywords
_COLLOQUIAL_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:退钱|退掉|不想要了|怎么退)"), "7天无理由退换货政策 退款"),
    (re.compile(r"(?:邮费|快递费|路费|谁掏运费|运费谁出)"), "运费险 退换货运费承担"),
    (re.compile(r"(?:坏了|没声音|无声|充不进电|失灵|修一下|换新)"), "质保 硬件故障 15天免费换新 1年质保"),
    (re.compile(r"(?:一直没动|卡住了|没物流|丢了|破损)"), "发货时效 物流异常 72小时未更新 丢件赔付"),
    (re.compile(r"(?:开票|发票抬头|专票|普票|红冲)"), "增值税电子发票 换开 红冲重开"),
)


class QueryRewriter:
    """Context-aware query rewriter combining active session slots and domain terminology normalization."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    async def rewrite(
        self,
        query: str,
        *,
        slots: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Return `(rewritten_query, rewrite_applied)`."""
        cleaned = query.strip()
        if not self.enabled or not cleaned:
            return cleaned, False

        slots = slots or {}
        additions: list[str] = []

        # 1. Pronoun / implicit entity resolution from active slots
        product_names = _extract_slot_product_names(slots)
        if product_names and (_PRONOUN_RE.search(cleaned) or len(cleaned) <= 10):
            for pname in product_names:
                if pname not in cleaned and pname not in additions:
                    additions.append(pname)

        # 2. Colloquial customer-service phrasing -> canonical policy terms
        for pattern, canonical in _COLLOQUIAL_EXPANSIONS:
            if pattern.search(cleaned):
                for term in canonical.split():
                    if term not in cleaned and term not in additions:
                        additions.append(term)

        if not additions:
            return cleaned, False

        rewritten = f"{cleaned} ({' '.join(additions)})"
        return rewritten, True


def _extract_slot_product_names(slots: dict[str, Any]) -> list[str]:
    names: list[str] = []
    active_orders = slots.get("active_orders")
    if isinstance(active_orders, dict):
        for info in active_orders.values():
            if isinstance(info, dict) and info.get("product_name"):
                pname = str(info["product_name"]).strip()
                if pname and pname not in names:
                    names.append(pname)
    if slots.get("product_name"):
        pname = str(slots["product_name"]).strip()
        if pname and pname not in names:
            names.append(pname)
    return names
