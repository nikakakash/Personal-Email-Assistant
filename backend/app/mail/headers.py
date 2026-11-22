"""Header safety helpers: injection guard and reply subjects."""


class HeaderInjectionError(ValueError):
    """Raised when a header value contains a line break or control character."""


# Every char str.splitlines() treats as a line break, plus NUL and other C0
# controls (tab is allowed) and DEL.
_FORBIDDEN = frozenset(
    {chr(c) for c in range(0x20) if c != 0x09} | {"\x7f", "\x85", "\u2028", "\u2029"}
)


def assert_safe_header(value: str) -> str:
    """Reject line breaks and control characters; return the value unchanged otherwise."""
    if any(ch in _FORBIDDEN for ch in value):
        raise HeaderInjectionError("header value contains a line break or control character")
    return value


def reply_subject(subject: str) -> str:
    """Prefix ``Re: `` unless the subject already starts with ``re:`` (case-insensitive)."""
    stripped = subject.strip()
    if stripped.lower().startswith("re:"):
        return stripped
    return f"Re: {stripped}"
