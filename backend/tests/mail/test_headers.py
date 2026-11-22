"""Tests for header safety helpers."""

import pytest

from app.mail.headers import HeaderInjectionError, assert_safe_header, reply_subject


def test_header_injection_rejected() -> None:
    with pytest.raises(HeaderInjectionError):
        assert_safe_header("Hi\r\nBcc: x@y")


def test_safe_header_returned() -> None:
    assert assert_safe_header("Hello") == "Hello"


def test_reply_subject_idempotent() -> None:
    assert reply_subject("RE: Lunch") == "RE: Lunch"
    assert reply_subject("Lunch") == "Re: Lunch"


@pytest.mark.parametrize(
    "ch",
    ["\x00", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "\x07", "\x7f"],
)
def test_header_control_and_line_break_chars_rejected(ch: str) -> None:
    with pytest.raises(HeaderInjectionError):
        assert_safe_header(f"Hi{ch}Bcc: x@y")


def test_header_tab_allowed() -> None:
    assert assert_safe_header("a\tb") == "a\tb"
