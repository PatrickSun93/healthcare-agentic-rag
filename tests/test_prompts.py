import pytest

from src import prompts, rag
from src.prompts import PINNED


def test_every_pinned_prompt_loads_and_is_nonempty():
    for name in PINNED:
        p = prompts.get(name)
        assert p.text and p.name == name and p.version == PINNED[name]
        assert p.id == f"{name}@{PINNED[name]}"


def test_grounded_prompt_contains_the_exact_refusal_string():
    # rag.REFUSAL is matched verbatim by the self-RAG grader; the prompt must instruct that exact sentence.
    assert rag.REFUSAL in prompts.get("grounded_answer").text


def test_version_override_via_env(monkeypatch):
    monkeypatch.setenv("PROMPT_VERSIONS", "grounded_answer=v9, router=v1,junk,unknown=v3")
    v = prompts.versions()
    assert v["grounded_answer"] == "v9" and v["router"] == "v1"
    assert "unknown" not in v
    with pytest.raises(FileNotFoundError):
        prompts.get("grounded_answer")  # v9 doesn't exist (yet)


def test_unknown_prompt_name():
    with pytest.raises(KeyError):
        prompts.get("nope")


def test_available_lists_versions_on_disk():
    assert prompts.available("grounded_answer") == ["v1"]
