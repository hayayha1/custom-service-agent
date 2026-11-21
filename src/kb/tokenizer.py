"""Chinese & alphanumeric tokenizer powering PostgreSQL `tsvector` + GIN sparse full-text search."""

from __future__ import annotations

import re

import jieba

_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_#-]+")
_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "和",
        "是",
        "在",
        "我",
        "有",
        "就",
        "都",
        "而",
        "及",
        "与",
        "着",
        "或",
        "一个",
        "没有",
        "我们",
        "你们",
        "他们",
        "请问",
        "一下",
        "怎么",
        "什么",
        "呢",
        "吗",
        "吧",
        "啊",
    }
)


def tokenize_terms(text: str) -> list[str]:
    """Segment Chinese/English text into normalized search tokens using jieba search mode."""
    if not text or not text.strip():
        return []

    raw_tokens = jieba.cut_for_search(text.lower())
    terms: list[str] = []
    for tok in raw_tokens:
        cleaned = tok.strip()
        if not cleaned or cleaned in _STOPWORDS:
            continue
        if _TOKEN_RE.fullmatch(cleaned):
            terms.append(cleaned)
    return terms


def tokenize_for_fts(text: str) -> str:
    """Return space-delimited tokens suitable for `to_tsvector('simple', search_tokens)`."""
    return " ".join(tokenize_terms(text))


def build_postgres_tsquery(query: str, max_terms: int = 12) -> str:
    """Build an OR-joined Postgres tsquery string (e.g. `'耳机' | '质保' | '换新'`)."""
    terms: list[str] = []
    for tok in tokenize_terms(query):
        # Strip characters unsafe for tsquery literals
        safe = re.sub(r"[^0-9a-z\u4e00-\u9fff_-]", "", tok)
        if safe and safe not in terms:
            terms.append(safe)
        if len(terms) >= max_terms:
            break
    if not terms:
        return ""
    return " | ".join(terms)
