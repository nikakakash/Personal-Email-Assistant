"""Chunking: clean an email body (quoted replies, signatures), split it, hash the pieces."""

from datetime import UTC, datetime

from app.retrieval.chunking import (
    Chunk,
    chunk_message,
    clean_body,
    header_line,
)

WHEN = datetime(2026, 3, 4, 9, 30, tzinfo=UTC)


def _chunks(body: str, **kw: object) -> list[Chunk]:
    args: dict[str, object] = {
        "sender": "alex@example.com",
        "to": ["owner@example.com"],
        "subject": "Quote",
        "received_at": WHEN,
        "body_text": body,
        "body_html": None,
    }
    args.update(kw)
    return chunk_message(**args)  # type: ignore[arg-type]


# --- clean_body -----------------------------------------------------------------------------


def test_strips_gmail_quoted_reply() -> None:
    body = (
        "Yes, 4500 works for us.\n\n"
        "On Mon, Mar 2, 2026 at 10:00 AM Owner <owner@example.com> wrote:\n"
        "> Can you do 4500?\n> Thanks"
    )
    assert clean_body(body) == "Yes, 4500 works for us."


def test_strips_wrapped_on_wrote_line() -> None:
    body = (
        "Agreed.\n\nOn Mon, Mar 2, 2026 at 10:00 AM Owner Person\n<owner@example.com> wrote:\n> old"
    )
    assert clean_body(body) == "Agreed."


def test_strips_outlook_original_message() -> None:
    body = "See below.\n\n-----Original Message-----\nFrom: Bob\nSent: Monday\n\nold text"
    assert clean_body(body) == "See below."


def test_strips_outlook_header_block_without_marker() -> None:
    body = "Fine by me.\n\nFrom: Bob Smith <bob@x.com>\nSent: Monday, March 2, 2026\nTo: me\n\nold"
    assert clean_body(body) == "Fine by me."


def test_strips_quote_lines_interleaved() -> None:
    body = "> question one?\nAnswer one.\n> question two?\nAnswer two."
    assert clean_body(body) == "Answer one.\nAnswer two."


def test_strips_signature_delimiter_and_mobile_footer() -> None:
    assert clean_body("Call me tomorrow.\n-- \nAlex\nCEO, Acme") == "Call me tomorrow."
    assert clean_body("Ok!\n\nSent from my iPhone") == "Ok!"


def test_keeps_forwarded_content() -> None:
    body = (
        "FYI\n\n---------- Forwarded message ---------\n"
        "From: Bank <bank@example.com>\nDate: Mon, Mar 2, 2026\n\nYour invoice is due May 1."
    )
    assert "Your invoice is due May 1." in clean_body(body)


def test_does_not_strip_when_only_quote_marker_text_exists() -> None:
    # Nothing new before the marker: keep the text rather than return nothing.
    body = "On Mon, Mar 2, 2026 Owner wrote:\n> the agreed price is 4500"
    assert "4500" in clean_body(body)


def test_collapses_blank_lines_and_trims() -> None:
    assert clean_body("\n\n a \n\n\n\n b \n\n") == "a\n\nb"


# --- header + chunking -------------------------------------------------------------------------


def test_header_line_has_who_when_what() -> None:
    line = header_line("alex@example.com", ["a@x.com", "b@x.com"], "Quote", WHEN)
    assert (
        line == "From: alex@example.com | To: a@x.com, b@x.com | Date: 2026-03-04 | Subject: Quote"
    )


def test_short_message_is_one_chunk_with_header() -> None:
    chunks = _chunks("The price is 4500.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text.startswith("From: alex@example.com")
    assert chunks[0].text.endswith("The price is 4500.")


def test_long_message_splits_with_overlap_and_bounded_size() -> None:
    paragraphs = [" ".join(f"p{p}w{i}" for i in range(100)) for p in range(8)]
    chunks = _chunks("\n\n".join(paragraphs), max_words=250, overlap_words=40)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        body_words = c.text.split("\n", 1)[1].split()
        assert len(body_words) <= 250
    # Consecutive chunks share the overlap words.
    first = chunks[0].text.split()
    second = chunks[1].text.split("\n", 1)[1].split()
    assert first[-40:] == second[:40]
    # Every word of the body survives somewhere.
    every = {w for c in chunks for w in c.text.split()}
    assert {f"p{p}w{i}" for p in range(8) for i in range(100)} <= every


def test_single_huge_paragraph_is_split_by_words() -> None:
    chunks = _chunks(" ".join(f"w{i}" for i in range(1000)), max_words=250, overlap_words=40)
    assert len(chunks) >= 4
    assert all(len(c.text.split("\n", 1)[1].split()) <= 250 for c in chunks)


def test_html_only_message_is_used() -> None:
    chunks = _chunks("", body_html="<p>Meeting moved to <b>Friday</b></p>")
    assert "Meeting moved to Friday" in chunks[0].text


def test_empty_body_with_subject_still_indexes_header() -> None:
    chunks = _chunks("   ", subject="Contract signed")
    assert len(chunks) == 1 and "Contract signed" in chunks[0].text


def test_nothing_to_index() -> None:
    assert _chunks("", subject="") == []


def test_hash_is_stable_and_content_sensitive() -> None:
    a, b, c = _chunks("same"), _chunks("same"), _chunks("different")
    assert a[0].content_hash == b[0].content_hash != c[0].content_hash
    assert len(a[0].content_hash) == 64
