"""Header safety helpers: injection guard and reply subjects."""


class HeaderInjectionError(ValueError):
    """Raised when a header value contains CR or LF."""


def assert_safe_header(value: str) -> str:
    """Reject values containing line breaks; return the value unchanged otherwise."""
    if "\r" in value or "\n" in value:
        raise HeaderInjectionError("header value contains a line break")
    return value


def reply_subject(subject: str) -> str:
    """Prefix ``Re: `` unless the subject already starts with ``re:`` (case-insensitive)."""
    stripped = subject.strip()
    if stripped.lower().startswith("re:"):
        return stripped
    return f"Re: {stripped}"
