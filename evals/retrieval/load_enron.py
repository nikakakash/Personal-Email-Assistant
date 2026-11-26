"""Load a slice of the public Enron email corpus as realistic noise for the retrieval eval.

Source: the CMU release (https://www.cs.cmu.edu/~enron/), either extracted (`maildir/`) or the
`enron_mail_*.tar.gz` itself, which is streamed member by member so a 10k-message slice never
needs the whole archive on disk. Files are parsed as untrusted data with the standard library
email parser (no code from the archive is run). Messages are deduplicated by Message-ID (the
same mail appears in several folders), and messages without a sender or date are skipped.

The Enron mail goes into the same eval account as the synthetic corpus, so in `--scope mailbox`
every query competes against thousands of real, unrelated messages. Its 1999-2002 dates never
collide with the synthetic 2025-2026 ones, and its senders never match a query sender, so the
golden answers are unchanged.
"""

import hashlib
import re
import tarfile
import uuid
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

MAX_FILE_BYTES = 512 * 1024
_SUBJECT_PREFIX = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.IGNORECASE)


@dataclass(frozen=True)
class EnronMessage:
    message_id: str
    thread_key: str
    sender: str
    to: list[str]
    cc: list[str]
    subject: str
    body: str
    received_at: datetime


def _addresses(values: list[str]) -> list[str]:
    return [addr.lower() for _, addr in getaddresses(values) if "@" in addr][:50]


def _body(msg: EmailMessage) -> str:
    part = msg.get_body(preferencelist=("plain",))
    if part is None:
        return ""
    try:
        return str(part.get_content())
    except (LookupError, ValueError):
        payload = part.get_payload(decode=True)
        return payload.decode("latin-1", "replace") if isinstance(payload, bytes) else ""


def parse(raw: bytes) -> EnronMessage | None:
    if len(raw) > MAX_FILE_BYTES:
        return None
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    assert isinstance(msg, EmailMessage)
    try:
        senders = _addresses([str(msg.get("From", ""))])
        date = parsedate_to_datetime(str(msg.get("Date", "")))
    except (TypeError, ValueError, IndexError):
        return None
    if not senders or date is None:
        return None
    subject = " ".join(str(msg.get("Subject", "")).split())[:300]
    message_id = str(msg.get("Message-ID", "")).strip() or hashlib.sha1(raw).hexdigest()
    normalized = _SUBJECT_PREFIX.sub("", subject).lower() or message_id
    return EnronMessage(
        message_id=message_id,
        thread_key="enron-" + hashlib.sha1(normalized.encode()).hexdigest()[:16],
        sender=senders[0],
        to=_addresses([str(v) for v in msg.get_all("To", [])]),
        cc=_addresses([str(v) for v in msg.get_all("Cc", [])]),
        subject=subject,
        body=_body(msg),
        received_at=date if date.tzinfo else date.replace(tzinfo=UTC),
    )


def iter_messages(source: Path, limit: int) -> Iterator[EnronMessage]:
    """Unique, parseable messages from a maildir or a .tar.gz, at most `limit`."""
    if not source.exists():
        raise FileNotFoundError(f"Enron source not found: {source}")
    seen: set[str] = set()

    def accept(raw: bytes) -> EnronMessage | None:
        message = parse(raw)
        if message is None or message.message_id in seen:
            return None
        seen.add(message.message_id)
        return message

    count = 0
    if source.is_file():
        # Streaming, no random access. A partial download (e.g. `curl -r 0-40000000`) is fine:
        # reading stops cleanly where the truncated archive ends.
        try:
            with tarfile.open(source, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or member.size > MAX_FILE_BYTES:
                        continue
                    handle = archive.extractfile(member)  # read in memory, never written out
                    if handle is None:
                        continue
                    message = accept(handle.read())
                    if message is not None:
                        yield message
                        count += 1
                        if count >= limit:
                            return
        except (EOFError, tarfile.ReadError, zlib.error):
            return
        return
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        message = accept(path.read_bytes())
        if message is not None:
            yield message
            count += 1
            if count >= limit:
                return


async def load_enron(
    maker: async_sessionmaker[AsyncSession], source: Path, account_id: str, *, limit: int
) -> int:
    loaded = 0
    threads: dict[str, uuid.UUID] = {}
    batch: list[EnronMessage] = []

    async def flush() -> None:
        async with maker() as session, session.begin():
            for m in batch:
                if m.thread_key not in threads:
                    threads[m.thread_key] = await session.scalar(
                        text(
                            "insert into email_threads (account_id, provider_thread_id) "
                            "values (:a, :t) on conflict (account_id, provider_thread_id) "
                            "do update set account_id = excluded.account_id returning id"
                        ),
                        {"a": account_id, "t": m.thread_key},
                    )
                await session.execute(
                    text(
                        "insert into email_messages (thread_id, account_id, provider_message_id, "
                        "sender, to_addrs, cc_addrs, subject, body_text, received_at, "
                        "content_hash) values (:th, :a, :m, :s, :to, :cc, :subj, :body, :at, :h) "
                        "on conflict (account_id, provider_message_id) do nothing"
                    ),
                    {
                        "th": threads[m.thread_key],
                        "a": account_id,
                        "m": "enron-" + hashlib.sha1(m.message_id.encode()).hexdigest()[:20],
                        "s": m.sender,
                        "to": m.to,
                        "cc": m.cc,
                        "subj": m.subject,
                        "body": m.body.replace("\x00", ""),
                        "at": m.received_at,
                        "h": hashlib.sha1(m.body.encode("utf-8", "replace")).hexdigest(),
                    },
                )
        batch.clear()

    for message in iter_messages(source, limit):
        batch.append(message)
        loaded += 1
        if len(batch) >= 500:
            await flush()
    if batch:
        await flush()
    return loaded
