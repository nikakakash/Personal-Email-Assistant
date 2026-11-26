"""Tests for model-facing text and HTML sanitising."""

from app.mail.content import MAX_BODY_CHARS, model_text, sanitize_html


def test_plain_text_preferred() -> None:
    assert model_text("plain", "<p>html</p>") == "plain"


def test_html_only_body_derives_text() -> None:
    assert model_text(None, "<p>Hello <b>there</b></p>") == "Hello there"


def test_blocks_do_not_glue_words() -> None:
    assert model_text(None, "<p>One</p><p>Two</p>a<br>b") == "One Two a b"


def test_script_and_style_dropped() -> None:
    out = model_text(None, "<style>p{}</style><script>evil()</script><p>ok</p>")
    assert out == "ok"


def test_long_body_truncated_with_marker() -> None:
    out = model_text("x" * (MAX_BODY_CHARS + 500), None)
    assert len(out) <= MAX_BODY_CHARS + 60
    assert "truncated" in out
    assert "500 more characters" in out


def test_sanitize_strips_remote_images() -> None:
    out = sanitize_html("<img src='http://t.test/p.gif'><a href='x'>l</a>")
    assert "<img" not in out
    assert 'rel="noopener noreferrer nofollow"' in out


def test_sanitize_strips_script() -> None:
    out = sanitize_html("<p onclick='x()'>a</p><script>alert(1)</script>")
    assert "script" not in out
    assert "alert" not in out
    assert "onclick" not in out


def test_sanitize_strips_javascript_url_and_style() -> None:
    out = sanitize_html("<a href='javascript:x()' style='color:red'>l</a>")
    assert "javascript" not in out
    assert "style" not in out


def test_section_blocks_do_not_glue_words() -> None:
    assert model_text(None, "<section>A</section><section>B</section>") == "A B"


def test_gt_inside_attribute_does_not_leak() -> None:
    assert model_text(None, '<div title="a>b">Hi</div><p>there</p>') == "Hi there"
