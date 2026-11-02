import pytest

from app.rag import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=800, overlap=100) == ["hello world"]


def test_overlap_between_chunks():
    text = "abcdefghij" * 3  # 30 chars
    chunks = chunk_text(text, size=10, overlap=2)
    assert chunks[0] == text[0:10]
    assert chunks[1] == text[8:18]  # starts 2 chars before the end of the first
    assert "".join(c if i == 0 else c[2:] for i, c in enumerate(chunks)) == text


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=5, overlap=5)
