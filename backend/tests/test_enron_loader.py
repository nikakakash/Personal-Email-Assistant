"""evals/retrieval/load_enron.py on a tiny synthetic maildir and tarball (no real corpus)."""

import io
import tarfile
from pathlib import Path

import pytest
from evals.retrieval.load_enron import iter_messages, load_enron, parse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import EmailMessage, EmailThread

MAIL = b"""Message-ID: <1@enron.example>
Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)
From: phillip.allen@enron.com
To: tim.belden@enron.com, "John" <john.arnold@enron.com>
Subject: Re: Gas forecast

Here is our forecast for the gas desk.
"""

DUPLICATE = MAIL  # the same message filed in another folder
NO_DATE = b"Message-ID: <2@enron.example>\nFrom: a@enron.com\nSubject: x\n\nbody\n"
SECOND = MAIL.replace(b"<1@", b"<3@").replace(b"Re: Gas forecast", b"FW: Gas forecast")


def _maildir(root: Path) -> Path:
    for rel, raw in {
        "allen-p/inbox/1.": MAIL,
        "allen-p/sent/1.": DUPLICATE,
        "allen-p/inbox/2.": NO_DATE,
        "allen-p/inbox/3.": SECOND,
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root


def test_parse_fields() -> None:
    m = parse(MAIL)
    assert m is not None
    assert m.sender == "phillip.allen@enron.com"
    assert m.to == ["tim.belden@enron.com", "john.arnold@enron.com"]
    assert m.subject == "Re: Gas forecast"
    assert "forecast for the gas desk" in m.body
    assert m.received_at.year == 2001 and m.received_at.tzinfo is not None
    assert parse(NO_DATE) is None


def test_maildir_dedupes_skips_and_threads(tmp_path: Path) -> None:
    messages = list(iter_messages(_maildir(tmp_path), limit=10))
    assert [m.message_id for m in messages] == ["<1@enron.example>", "<3@enron.example>"]
    assert messages[0].thread_key == messages[1].thread_key  # Re:/FW: of the same subject
    assert len(list(iter_messages(tmp_path, limit=1))) == 1


def test_missing_source_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_messages(tmp_path / "nope.tar.gz", limit=1))


def test_tarball_is_streamed(tmp_path: Path) -> None:
    archive = tmp_path / "enron.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, raw in [("maildir/a/1.", MAIL), ("maildir/a/2.", SECOND)]:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    assert len(list(iter_messages(archive, limit=10))) == 2
    truncated = tmp_path / "partial.tar.gz"
    truncated.write_bytes(archive.read_bytes()[:-40])  # like a ranged partial download
    assert len(list(iter_messages(truncated, limit=10))) <= 2  # stops cleanly, no exception


async def test_load_into_database(
    tmp_path: Path, sessionmaker: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    root = _maildir(tmp_path)
    assert await load_enron(sessionmaker, root, "eval", limit=10) == 2
    assert await load_enron(sessionmaker, root, "eval", limit=10) == 2  # idempotent insert
    assert await session.scalar(select(func.count()).select_from(EmailMessage)) == 2
    assert await session.scalar(select(func.count()).select_from(EmailThread)) == 1
