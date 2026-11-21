"""System prompt assembly injecting independent active slots and L3 long-term memories."""

from __future__ import annotations

from typing import Any

BASE_SYSTEM_PROMPT = """你是一家垂直电商平台的专业售后客服智能助手。

【核心工作准则】
1. 事实严谨：涉及退换货政策、运费险、质保、发货规则时，优先调用 `search_knowledge_base`；涉及订单与物流信息时，必须调用 `query_order_status` 或 `get_user_orders` 查证，严禁编造订单状态或政策条款。
2. 善用上下文槽位：下方【当前会话已知业务信息】中的订单号、手机号、商品名称等均已由系统确认，直接复用即可，严禁向用户重复索要已经提供过的信息。
3. 谨慎写操作：仅在确认用户意图和关键问题后调用 `create_support_ticket` 创建售后工单。
4. 及时转人工：如果用户明确要求转人工、问题超出系统权限或工具多次调用失败无法解决，请调用 `transfer_to_human` 并附带完整交接摘要。
5. 表达风格：简洁、礼貌、有同理心，优先给出明确结论与下一步操作指引。"""


def build_system_prompt(
    *,
    user_id: str,
    slots: dict[str, Any],
    memory_context: dict[str, Any],
) -> str:
    """Build the turn-specific System Prompt with pinned slots and recalled L3 memory."""
    sections: list[str] = [BASE_SYSTEM_PROMPT, f"\n【当前用户信息】\n- user_id: {user_id}"]

    if slots:
        slot_lines: list[str] = []
        active_orders = slots.get("active_orders")
        if isinstance(active_orders, dict) and active_orders:
            slot_lines.append(f"- 进行中的订单列表（共 {len(active_orders)} 笔）：")
            for oid, info in active_orders.items():
                if isinstance(info, dict):
                    parts = [f"订单号={oid}"]
                    if info.get("product_name"):
                        parts.append(f"商品={info['product_name']}")
                    if info.get("status"):
                        parts.append(f"状态={info['status']}")
                    if info.get("tracking_no"):
                        parts.append(f"运单号={info['tracking_no']}")
                    if info.get("ticket_id"):
                        parts.append(f"关联工单={info['ticket_id']}")
                    slot_lines.append("  * " + "，".join(parts))
                else:
                    slot_lines.append(f"  * 订单号={oid}")

        active_tickets = slots.get("active_tickets")
        if isinstance(active_tickets, dict) and active_tickets:
            slot_lines.append("- 已创建的工单列表：")
            for tid, tinfo in active_tickets.items():
                if isinstance(tinfo, dict):
                    oid_part = f"，关联订单={tinfo['order_id']}" if tinfo.get("order_id") else ""
                    summary_part = f"，摘要={tinfo['issue_summary']}" if tinfo.get("issue_summary") else ""
                    slot_lines.append(f"  * 工单号={tid}{oid_part}{summary_part}")
                else:
                    slot_lines.append(f"  * 工单号={tid}")

        skip_keys = {"active_orders", "active_tickets", "order_ids", "tracking_nos"}
        for k, v in slots.items():
            if k in skip_keys or v is None:
                continue
            if isinstance(v, (dict, list)):
                if v:
                    slot_lines.append(f"- {k}: {v}")
            elif str(v).strip():
                slot_lines.append(f"- {k}: {v}")

        if slot_lines:
            sections.append(
                "\n【当前会话已知业务信息（活跃槽位，跨轮次有效）】\n" + "\n".join(slot_lines)
            )

    user_facts = memory_context.get("user_facts") or []
    if user_facts:
        fact_lines = [f"- {f['key']}: {f['value']}" for f in user_facts]
        sections.append("\n【用户长期画像与偏好】\n" + "\n".join(fact_lines))

    episodic = memory_context.get("episodic_memories") or []
    if episodic:
        ep_lines = [
            f"- [历史会话 {e['session_id']}] {e['summary']}"
            for e in episodic
        ]
        sections.append("\n【相关历史会话记忆】\n" + "\n".join(ep_lines))

    return "\n".join(sections)
