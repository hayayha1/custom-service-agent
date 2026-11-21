"""Human escalation tool that cleanly flags the AgentState for handoff."""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class TransferToHumanInput(BaseModel):
    reason: str = Field(..., min_length=2, description="转人工原因，例如：用户强烈要求人工、超出自助售后权限、复杂投诉")
    context_summary: str = Field(..., min_length=4, description="给人工客服看的简洁交接摘要（含订单号、诉求与已排查结论）")


def build_escalation_tool() -> StructuredTool:
    """Create the `transfer_to_human` tool."""

    async def _transfer_to_human(reason: str, context_summary: str) -> str:
        return json.dumps(
            {
                "ok": True,
                "action": "transfer_to_human",
                "reason": reason,
                "context_summary": context_summary,
                "message": "已登记转人工请求，正在为您接入人工客服专员。",
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=_transfer_to_human,
        name="transfer_to_human",
        description="当用户明确要求人工客服、问题超出系统权限或情绪激动需要人工介入时调用，生成交接摘要并转接人工。",
        args_schema=TransferToHumanInput,
    )
