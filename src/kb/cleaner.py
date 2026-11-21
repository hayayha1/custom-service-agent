"""Data cleaning, noise removal, PII redaction, and SHA-256 content fingerprinting for KB documents."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_BR_RE = re.compile(r"<(?:br|p|/p|div|/div|li|/li|tr|/tr)\b[^>]*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PAGE_WATERMARK_RE = re.compile(
    r"^\s*(?:第\s*\d+\s*页(?:\s*/\s*共\s*\d+\s*页)?|page\s+\d+(?:\s+of\s+\d+)?|内部资料\s*请勿外传)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_DECORATIVE_LINE_RE = re.compile(r"^\s*[=*~_]{5,}\s*$", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ \t\u3000]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# PII patterns: Chinese mobile numbers, 18-digit ID cards, email addresses, 16-19 digit bank cards
_PHONE_PII_RE = re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)")
_ID_CARD_PII_RE = re.compile(
    r"(?<!\d)(\d{6})(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
_EMAIL_PII_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_BANK_CARD_PII_RE = re.compile(r"(?<!\d)(62\d{2})\d{8,11}(\d{4})(?!\d)")


@dataclass(slots=True)
class CleanedDocument:
    text: str
    content_hash: str
    pii_redacted_count: int = 0
    noise_removed_count: int = 0


def clean_document_text(raw_text: str, *, redact_pii: bool = True) -> CleanedDocument:
    """Clean raw KB text: normalize encoding/HTML, strip headers/watermarks, redact PII, and hash."""
    if not raw_text:
        empty_hash = hashlib.sha256(b"").hexdigest()
        return CleanedDocument(text="", content_hash=empty_hash)

    text = unicodedata.normalize("NFKC", raw_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    noise_count = 0

    # 1. Strip HTML script/style blocks and inline tags
    text, n_script = _SCRIPT_STYLE_RE.subn("", text)
    noise_count += n_script
    text = _HTML_BR_RE.sub("\n", text)
    text, n_tags = _HTML_TAG_RE.subn("", text)
    noise_count += n_tags
    text = html.unescape(text)

    # 2. Remove invisible control chars, page number footers, and decorative divider lines
    text, n_ctrl = _CONTROL_CHAR_RE.subn("", text)
    text, n_page = _PAGE_WATERMARK_RE.subn("", text)
    text, n_deco = _DECORATIVE_LINE_RE.subn("", text)
    noise_count += n_ctrl + n_page + n_deco

    # 3. Optional PII redaction
    pii_count = 0
    if redact_pii:
        text, n_phone = _PHONE_PII_RE.subn(r"\1****\2", text)
        text, n_id = _ID_CARD_PII_RE.subn(r"\1************", text)
        text, n_email = _EMAIL_PII_RE.subn(r"\1***@\2", text)
        text, n_bank = _BANK_CARD_PII_RE.subn(r"\1********\2", text)
        pii_count = n_phone + n_id + n_email + n_bank

    # 4. Normalize spaces per line and collapse >=3 consecutive newlines
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return CleanedDocument(
        text=text,
        content_hash=content_hash,
        pii_redacted_count=pii_count,
        noise_removed_count=noise_count,
    )
