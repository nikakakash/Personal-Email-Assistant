"""Turn one email into retrievable chunks.

1. `model_text` (the same normalizer the prompts use) picks plain text or sanitized HTML.
2. `clean_body` removes what would duplicate other messages in the index: quoted replies
   (`>` lines, "On … wrote:", Outlook headers) and signatures. Forwarded content is kept, since
   it is often the only copy of the forwarded information.
3. The body is packed into word windows (paragraphs kept whole when they fit), consecutive
   windows overlapping by `overlap_words`, and each chunk starts with a header line so it
   carries who/when/what even when retrieved alone.

Word counts stand in for tokens (English averages ~1.3 tokens per word): 260 words ≈ 350 tokens.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.mail.content import model_text

DEFAULT_MAX_WORDS = 260
DEFAULT_OVERLAP_WORDS = 40

# A reply marker: everything from here down is the quoted earlier conversation.
_REPLY_MARKERS = (
    # Gmail / Apple: "On <date> <name> wrote:", possibly wrapped onto a second line.
    re.compile(r"^On\b[^\n]{0,300}(?:\n[^\n]{0,300})?\bwrote:[ \t]*$", re.MULTILINE),
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}[ \t]*$", re.MULTILINE | re.IGNORECASE),
    # Outlook without a marker: a From: line followed by Sent:/Date:.
    re.compile(r"^From:[^\n]*\n(?:Sent|Date):", re.MULTILINE),
)
_SIGNATURE_MARKERS = (
    re.compile(r"^-- ?$", re.MULTILINE),  # RFC 3676 signature delimiter
    re.compile(r"^Sent from my [^\n]+$", re.MULTILINE),
)
_FORWARDED = re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}", re.MULTILINE | re.IGNORECASE)
_QUOTE_LINE = re.compile(r"^[ \t]*>")
_BLANK_RUNS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    content_hash: str


def _drop_quote_lines(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not _QUOTE_LINE.match(line))


def _tidy(text: str) -> str:
    lines = [line.strip() for line in text.split("\n")]
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def clean_body(text: str) -> str:
    """Remove quoted replies and signatures; never return empty if the input had content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    forwarded = _FORWARDED.search(text)
    limit = forwarded.start() if forwarded else len(text)

    cut = limit
    for marker in (*_REPLY_MARKERS, *_SIGNATURE_MARKERS):
        found = marker.search(text, 0, limit)
        if found:
            cut = min(cut, found.start())
    kept = _tidy(_drop_quote_lines(text[:cut] + text[limit:]))
    if kept:
        return kept
    # Nothing new outside the quote: keep the quoted text rather than index nothing.
    return _tidy(text)


def header_line(sender: str, to: Sequence[str], subject: str, received_at: datetime) -> str:
    return (
        f"From: {sender} | To: {', '.join(to)} | "
        f"Date: {received_at.date().isoformat()} | Subject: {subject}"
    )


def _windows(words: list[list[str]], max_words: int, overlap: int) -> list[list[str]]:
    """Pack paragraphs (lists of words) into windows of at most `max_words`."""
    out: list[list[str]] = []
    current: list[str] = []
    fresh = 0  # words in `current` not already emitted

    def emit() -> None:
        nonlocal current, fresh
        out.append(current)
        current = current[-overlap:] if overlap else []
        fresh = 0

    for para in words:
        if fresh and len(current) + len(para) > max_words:
            emit()
        while len(current) + len(para) > max_words:
            take = max_words - len(current)
            current = current + para[:take]
            para = para[take:]
            fresh += take
            emit()
        current = current + para
        fresh += len(para)
    if fresh:
        out.append(current)
    return out


def chunk_message(
    *,
    sender: str,
    to: Sequence[str],
    subject: str,
    received_at: datetime,
    body_text: str | None,
    body_html: str | None,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words")
    body = clean_body(model_text(body_text, body_html))
    paragraphs = [p.split() for p in body.split("\n\n") if p.strip()]
    if not paragraphs and not subject.strip():
        return []
    header = header_line(sender, to, subject, received_at)
    windows = _windows(paragraphs, max_words, overlap_words) or [[]]
    chunks = []
    for i, window in enumerate(windows):
        text = f"{header}\n{' '.join(window)}".rstrip()
        chunks.append(Chunk(i, text, hashlib.sha256(text.encode()).hexdigest()))
    return chunks
