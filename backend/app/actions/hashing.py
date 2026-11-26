"""Canonical JSON serialisation and sha256 hashing of action payloads."""

import hashlib
import json
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


def _default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal | UUID):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic JSON: sorted keys, compact separators, non-ASCII kept."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default
    )


def payload_hash(payload: Mapping[str, Any]) -> str:
    """sha256 hex digest of the canonical JSON."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
