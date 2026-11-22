"""Tests for address normalisation and owner matching."""

import pytest

from app.mail.addresses import is_owner_address, normalize_address, parse_address_list


def test_normalize_display_name() -> None:
    assert normalize_address("Ana <Ana@Example.COM>") == "ana@example.com"


@pytest.mark.parametrize("raw", ["", "nobody", "a@", "@b.com"])
def test_normalize_rejects_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_address(raw)


def test_parse_address_list_dedupes_preserving_order() -> None:
    header = "Bob <bob@x.com>, ana@x.com, BOB@x.com"
    assert parse_address_list(header) == ["bob@x.com", "ana@x.com"]


def test_parse_address_list_none() -> None:
    assert parse_address_list(None) == []


def test_is_owner_exact_not_substring() -> None:
    assert is_owner_address("me@x.com.evil.test", ["me@x.com"]) is False
    assert is_owner_address("Alias@X.com", ["me@x.com", "alias@x.com"]) is True
