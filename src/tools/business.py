"""Vertical e-commerce customer service business tools (query orders + idempotent ticket creation)."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


@dataclass(slots=True)
class OrderRecord:
    order_id: str
    user_id: str
    product_name: str
    amount_cny: float
    status: str
    carrier: str
    tracking_no: str
    shipped_at: str | None
    delivered_at: str | None
    return_eligible: bool
    notes: str = ""


@dataclass(slots=True)
class SupportTicket:
    ticket_id: str
    idempotency_key: str
    user_id: str
    order_id: str
    issue_summary: str
    priority: str
    status: str = "open"
    already_existed: bool = False


class BusinessDataAdapter(Protocol):
    async def get_order(self, order_id: str) -> OrderRecord | None: ...

    async def list_user_orders(self, user_id: str) -> list[OrderRecord]: ...


class TicketStore(Protocol):
    async def create_or_get(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        order_id: str,
        issue_summary: str,
        priority: str,
    ) -> SupportTicket: ...


class MockEcommerceAdapter:
    """Phase 1 mock adapter with realistic vertical e-commerce orders (swappable with real ERP/OMS API)."""

    def __init__(self, orders: dict[str, OrderRecord] | None = None) -> None:
        self._orders: dict[str, OrderRecord] = orders or {
            "20261119001": OrderRecord(
                order_id="20261119001",
                user_id="user_001",
                product_name="声学降噪耳机 Pro",
                amount_cny=699.0,
                status="delivered",
                carrier="顺丰速运",
                tracking_no="SF109823456789",
                shipped_at="2026-11-15 10:20:00",
                delivered_at="2026-11-17 15:40:00",
                return_eligible=True,
                notes="已赠送运费险，支持7天无理由退换及1年质保",
            ),
            "20261119002": OrderRecord(
                order_id="20261119002",
                user_id="user_001",
                product_name="机械键盘 87键 静音轴",
                amount_cny=429.0,
                status="in_transit",
                carrier="中通快递",
                tracking_no="ZTO778899001122",
                shipped_at="2026-11-18 18:05:00",
                delivered_at=None,
                return_eligible=True,
                notes="运输中，预计明天送达",
            ),
            "20261119003": OrderRecord(
                order_id="20261119003",
                user_id="user_002",
                product_name="定制刻字保温杯",
                amount_cny=159.0,
                status="delivered",
                carrier="京东物流",
                tracking_no="JD009988776655",
                shipped_at="2026-11-10 09:00:00",
                delivered_at="2026-11-12 11:30:00",
                return_eligible=False,
                notes="私人定制刻字商品，非质量问题不支持7天无理由退货",
            ),
        }

    async def get_order(self, order_id: str) -> OrderRecord | None:
        return self._orders.get(order_id.strip())

    async def list_user_orders(self, user_id: str) -> list[OrderRecord]:
        return [o for o in self._orders.values() if o.user_id == user_id.strip()]


class PostgresTicketStore:
    """Idempotent ticket persistence backed by PostgreSQL `support_tickets` table."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def create_or_get(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        order_id: str,
        issue_summary: str,
        priority: str,
    ) -> SupportTicket:
        new_ticket_id = f"TK-{uuid.uuid4().hex[:8].upper()}"
        insert_sql = """
            INSERT INTO support_tickets (
                ticket_id, idempotency_key, user_id, order_id, issue_summary, priority, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, 'open')
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING ticket_id, idempotency_key, user_id, order_id, issue_summary, priority, status
        """
        select_sql = """
            SELECT ticket_id, idempotency_key, user_id, order_id, issue_summary, priority, status
            FROM support_tickets
            WHERE idempotency_key = %s
        """
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                insert_sql,
                (new_ticket_id, idempotency_key, user_id, order_id, issue_summary, priority),
            )
            row = await cur.fetchone()
            if row is not None:
                return SupportTicket(
                    ticket_id=row["ticket_id"],
                    idempotency_key=row["idempotency_key"],
                    user_id=row["user_id"],
                    order_id=row["order_id"] or "",
                    issue_summary=row["issue_summary"],
                    priority=row["priority"],
                    status=row["status"],
                    already_existed=False,
                )
            await cur.execute(select_sql, (idempotency_key,))
            existing = await cur.fetchone()
            return SupportTicket(
                ticket_id=existing["ticket_id"],
                idempotency_key=existing["idempotency_key"],
                user_id=existing["user_id"],
                order_id=existing["order_id"] or "",
                issue_summary=existing["issue_summary"],
                priority=existing["priority"],
                status=existing["status"],
                already_existed=True,
            )


class InMemoryTicketStore:
    """In-memory idempotent ticket store for tests and local prototyping."""

    def __init__(self) -> None:
        self._by_idempotency_key: dict[str, SupportTicket] = {}

    async def create_or_get(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        order_id: str,
        issue_summary: str,
        priority: str,
    ) -> SupportTicket:
        existing = self._by_idempotency_key.get(idempotency_key)
        if existing is not None:
            return SupportTicket(
                ticket_id=existing.ticket_id,
                idempotency_key=existing.idempotency_key,
                user_id=existing.user_id,
                order_id=existing.order_id,
                issue_summary=existing.issue_summary,
                priority=existing.priority,
                status=existing.status,
                already_existed=True,
            )
        ticket = SupportTicket(
            ticket_id=f"TK-{uuid.uuid4().hex[:8].upper()}",
            idempotency_key=idempotency_key,
            user_id=user_id,
            order_id=order_id,
            issue_summary=issue_summary,
            priority=priority,
            status="open",
            already_existed=False,
        )
        self._by_idempotency_key[idempotency_key] = ticket
        return ticket


class QueryOrderStatusInput(BaseModel):
    order_id: str = Field(..., min_length=4, description="要查询的订单号，如 20261119001")


class GetUserOrdersInput(BaseModel):
    user_id: str = Field(..., min_length=1, description="用户 ID，用于查询该用户近期订单列表")


class CreateSupportTicketInput(BaseModel):
    user_id: str = Field(..., min_length=1, description="用户 ID")
    issue_summary: str = Field(..., min_length=4, description="问题摘要与售后诉求说明")
    order_id: str = Field(default="", description="关联订单号（如有）")
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal",
        description="工单优先级：low / normal / high / urgent",
    )
    idempotency_key: str = Field(
        default="",
        description="可选幂等键；留空时系统自动按用户+订单+摘要生成以防重复建单",
    )


def _derive_idempotency_key(user_id: str, order_id: str, issue_summary: str) -> str:
    raw = f"{user_id.strip()}|{order_id.strip()}|{issue_summary.strip()}"
    return "tk_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_business_tools(
    adapter: BusinessDataAdapter,
    ticket_store: TicketStore,
) -> list[StructuredTool]:
    """Create read/write vertical customer service business tools."""

    async def _query_order_status(order_id: str) -> str:
        order = await adapter.get_order(order_id)
        if order is None:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "not_found",
                    "message": f"未查询到订单号 {order_id}，请核对订单号是否正确。",
                },
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "order": asdict(order)}, ensure_ascii=False)

    async def _get_user_orders(user_id: str) -> str:
        orders = await adapter.list_user_orders(user_id)
        return json.dumps(
            {
                "ok": True,
                "count": len(orders),
                "orders": [asdict(o) for o in orders],
            },
            ensure_ascii=False,
        )

    async def _create_support_ticket(
        user_id: str,
        issue_summary: str,
        order_id: str = "",
        priority: Literal["low", "normal", "high", "urgent"] = "normal",
        idempotency_key: str = "",
    ) -> str:
        idem_key = idempotency_key.strip() or _derive_idempotency_key(
            user_id, order_id, issue_summary
        )
        ticket = await ticket_store.create_or_get(
            idempotency_key=idem_key,
            user_id=user_id,
            order_id=order_id,
            issue_summary=issue_summary,
            priority=priority,
        )
        return json.dumps(
            {
                "ok": True,
                "ticket": asdict(ticket),
                "message": (
                    f"已存在相同售后工单 {ticket.ticket_id}，无需重复创建。"
                    if ticket.already_existed
                    else f"售后工单 {ticket.ticket_id} 已创建成功，专员将尽快跟进。"
                ),
            },
            ensure_ascii=False,
        )

    return [
        StructuredTool.from_function(
            coroutine=_query_order_status,
            name="query_order_status",
            description="根据订单号查询订单详情、商品名称、物流状态、运单号及是否支持退换货。",
            args_schema=QueryOrderStatusInput,
        ),
        StructuredTool.from_function(
            coroutine=_get_user_orders,
            name="get_user_orders",
            description="根据用户 ID 查询该用户的历史订单列表（当用户忘记订单号时可调用）。",
            args_schema=GetUserOrdersInput,
        ),
        StructuredTool.from_function(
            coroutine=_create_support_ticket,
            name="create_support_ticket",
            description="为用户创建售后跟进工单（如退换货质检、破损补发、物流异常赔付）。内置幂等防重机制。",
            args_schema=CreateSupportTicketInput,
        ),
    ]
