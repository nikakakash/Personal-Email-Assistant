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
