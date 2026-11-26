"""Tests for canonical JSON and payload hashing."""

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from app.actions.hashing import canonical_json, payload_hash


def test_hash_stable_under_key_order() -> None:
    assert payload_hash({"a": 1, "b": 2}) == payload_hash({"b": 2, "a": 1})


def test_hash_changes_on_body_edit() -> None:
    assert payload_hash({"body": "Hi"}) != payload_hash({"body": "Hi!"})


def test_canonical_json_format() -> None:
    assert canonical_json({"b": "é", "a": [1, 2]}) == '{"a":[1,2],"b":"é"}'


def test_canonical_json_special_types() -> None:
    payload = {
        "t": datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        "d": date(2026, 1, 1),
        "n": Decimal("1.50"),
        "u": UUID(int=1),
    }
    assert canonical_json(payload) == (
        '{"d":"2026-01-01","n":"1.50","t":"2026-01-01T09:00:00+00:00",'
        '"u":"00000000-0000-0000-0000-000000000001"}'
    )


def test_canonical_json_rejects_unknown_type() -> None:
    with pytest.raises(TypeError):
        canonical_json({"x": object()})
