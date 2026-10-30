from src import config as C
from src.ingest import _window, chunk_markdown

DOC = """# Title line

> intro blockquote

## Trigger
Initiate when lactate is greater than 2 mmol/L.

## One-Hour Bundle
1. Measure lactate.
2. Obtain blood cultures.

## Empty Section

## Dosing
Metformin ER 500 mg daily.
"""


def test_sections_become_chunks_with_titles():
    chunks = chunk_markdown(DOC, "sepsis.md")
    sections = [c["section"] for c in chunks]
    assert sections == ["Title line", "Trigger", "One-Hour Bundle", "Dosing"]
    assert all(c["source"] == "sepsis.md" for c in chunks)
    assert "lactate" in chunks[1]["text"]


def test_empty_sections_are_dropped():
    chunks = chunk_markdown(DOC, "x.md")
    assert "Empty Section" not in {c["section"] for c in chunks}


def test_text_before_first_heading_is_intro():
    chunks = chunk_markdown("plain text, no headings at all", "x.md")
    assert chunks == [{"text": "plain text, no headings at all", "source": "x.md", "section": "intro"}]


def test_window_overlap():
    words = " ".join(f"w{i}" for i in range(10))
    pieces = _window(words, max_tokens=4, overlap=1)
    assert pieces == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9", "w9"]
    assert _window("   ", 4, 1) == []


def test_long_section_is_windowed(monkeypatch):
    monkeypatch.setattr(C, "CHUNK_MAX_TOKENS", 5)
    monkeypatch.setattr(C, "CHUNK_OVERLAP", 2)
    body = " ".join(f"tok{i}" for i in range(12))
    chunks = chunk_markdown(f"## S\n{body}\n", "x.md")
    assert len(chunks) == 4  # step = 3 -> starts at 0, 3, 6, 9
    assert all(c["section"] == "S" for c in chunks)
    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]  # overlap carried over
