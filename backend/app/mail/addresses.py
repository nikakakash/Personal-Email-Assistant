"""Email address parsing and owner-identity matching (pure functions)."""

from collections.abc import Iterable
from email.utils import getaddresses, parseaddr


def normalize_address(raw: str) -> str:
    """Return the bare, lowercased ``local@domain`` address; ValueError if invalid."""
    addr = parseaddr(raw)[1].strip().lower()
    local, sep, domain = addr.partition("@")
    if not sep or not local or not domain or "@" in domain or any(c.isspace() for c in addr):
        raise ValueError(f"not a valid email address: {raw!r}")
    return addr


def parse_address_list(header: str | None) -> list[str]:
    """Parse an address-list header into normalized, de-duplicated, ordered addresses."""
    if not header:
        return []
    seen: dict[str, None] = {}
    for _name, raw in getaddresses([header]):
        try:
            seen.setdefault(normalize_address(raw), None)
        except ValueError:
            continue
    return list(seen)


def is_owner_address(addr: str, identities: Iterable[str]) -> bool:
    """True if ``addr`` exactly equals one of the owner's identities (never a substring)."""
    try:
        candidate = normalize_address(addr)
    except ValueError:
        return False
    owned: set[str] = set()
    for identity in identities:
        try:
            owned.add(normalize_address(identity))
        except ValueError:
            continue
    return candidate in owned
