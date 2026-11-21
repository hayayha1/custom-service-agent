"""Helper to extract LLM token usage and record generation TPS on the active TraceContext."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.observability.tracer import get_current_trace


def extract_token_usage(message: Any) -> tuple[int, int, int]:
    """Extract `(prompt_tokens, completion_tokens, total_tokens)` from a LangChain AIMessage."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_meta = getattr(message, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        prompt_tokens = int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
        completion_tokens = int(
            usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0
        )
        total_tokens = int(
            usage_meta.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

    if total_tokens == 0:
        resp_meta = getattr(message, "response_metadata", None)
        if isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
            if isinstance(token_usage, dict):
                prompt_tokens = int(
                    token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
                )
                completion_tokens = int(
                    token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
                )
                total_tokens = int(
                    token_usage.get("total_tokens") or (prompt_tokens + completion_tokens)
                )

    # Fallback heuristic estimation for local/scripted fake LLMs in unit tests
    if total_tokens == 0 and isinstance(message, AIMessage):
        text = str(message.content or "")
        tc_count = len(message.tool_calls or [])
        if text or tc_count:
            completion_tokens = max(1, len(text) // 2 + tc_count * 18)
            prompt_tokens = max(10, completion_tokens * 2)
            total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def record_llm_call_on_current_trace(
    *,
    model: str,
    response: Any | None,
    latency_ms: float,
    error_message: str | None = None,
) -> None:
    ctx = get_current_trace()
    if ctx is None:
        return
    if response is not None:
        p_tok, c_tok, tot_tok = extract_token_usage(response)
        preview = str(getattr(response, "content", "") or "")
        if not preview and getattr(response, "tool_calls", None):
            preview = "tool_calls: " + ", ".join(
                c.get("name", "") for c in response.tool_calls
            )
    else:
        p_tok, c_tok, tot_tok = 0, 0, 0
        preview = ""

    ctx.record_llm_generation(
        model=model,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=tot_tok,
        latency_ms=latency_ms,
        output_preview=preview,
        error_message=error_message,
    )
