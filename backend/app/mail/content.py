"""Model-facing text extraction and HTML sanitising for email bodies."""

import html
import re

import nh3

MAX_BODY_CHARS = 20_000

_BLOCK_TAGS = {
    "p",
    "br",
    "div",
    "li",
    "ul",
    "ol",
    "tr",
    "td",
    "th",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "hr",
    "section",
    "article",
    "header",
    "footer",
    "aside",
    "nav",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "caption",
    "main",
}
# After the first nh3 pass the only tags left are attribute-free block tags.
_BLOCK_TAG = re.compile(r"</?[a-z0-9]+\s*/?>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def model_text(body_text: str | None, body_html: str | None) -> str:
    """Return bounded text for the model: plain text if present, else derived from HTML."""
    if body_text and body_text.strip():
        text = body_text.strip()
    elif body_html:
        # nh3 parses the HTML properly (a ">" inside a quoted attribute is safe) and
        # drops <script>/<style> content; keep only bare block tags, make them spaces,
        # then strip whatever markup remains.
        blocks_only = nh3.clean(body_html, tags=_BLOCK_TAGS, attributes={})
        spaced = _BLOCK_TAG.sub(" ", blocks_only)
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
    attributes = {tag: set(attrs) - {"style"} for tag, attrs in nh3.ALLOWED_ATTRIBUTES.items()}
    return nh3.clean(
        html_body,
        tags=tags,
        attributes=attributes,
        link_rel="noopener noreferrer nofollow",
    )
