"""Opaque keyset cursors over `(created_at, id)`.

A cursor is URL-safe base64 of `["<created_at ISO>", "<uuid>"]`: the sort key of the last
row on the previous page. The next page is the rows strictly after it in the list's order,
so concurrent inserts never shift or duplicate rows the way OFFSET paging would.
"""

import base64
import binascii
import json
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import HTTPException, Query
from pydantic import BaseModel

DEFAULT_LIMIT = 50
MAX_LIMIT = 100

Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT)]


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None


def encode_cursor(created_at: datetime, row_id: uuid.UUID) -> str:
    raw = json.dumps([created_at.isoformat(), str(row_id)]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    """The cursor's sort key; 422 `invalid_cursor` if it is not one we issued."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created_at, row_id = json.loads(base64.urlsafe_b64decode(padded))
        parsed = datetime.fromisoformat(created_at)
        if parsed.tzinfo is None:
            raise ValueError("naive timestamp")
        return parsed, uuid.UUID(row_id)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="invalid_cursor") from exc
