"""Model-facing text extraction and HTML sanitising for email bodies."""

import html
import re

import nh3

MAX_BODY_CHARS = 20_000

_BLOCK_TAG = re.compile(
    r"</?(?:p|br|div|li|ul|ol|tr|td|th|table|h[1-6]|blockquote|pre|hr)\b[^>]*>",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


def model_text(body_text: str | None, body_html: str | None) -> str:
    """Return bounded text for the model: plain text if present, else derived from HTML."""
    if body_text and body_text.strip():
        text = body_text.strip()
    elif body_html:
        spaced = _BLOCK_TAG.sub(" ", body_html)
        # nh3 drops <script>/<style> content by default; tags=set() strips all markup.
        stripped = html.unescape(nh3.clean(spaced, tags=set()))
        text = _WHITESPACE.sub(" ", stripped).strip()
    else:
        text = ""
    if len(text) > MAX_BODY_CHARS:
        extra = len(text) - MAX_BODY_CHARS
        text = text[:MAX_BODY_CHARS] + f"\n[… truncated, {extra} more characters]"
    return text


def sanitize_html(html_body: str) -> str:
    """Sanitize HTML for display: no images, no style attributes, safe link rel.

    nh3 also removes scripts, ``on*`` event attributes and ``javascript:`` URLs by default.
    """
    tags = set(nh3.ALLOWED_TAGS) - {"img"}
    attributes = {
        tag: set(attrs) - {"style"} for tag, attrs in nh3.ALLOWED_ATTRIBUTES.items()
    }
    return nh3.clean(
        html_body,
        tags=tags,
        attributes=attributes,
        link_rel="noopener noreferrer nofollow",
    )
