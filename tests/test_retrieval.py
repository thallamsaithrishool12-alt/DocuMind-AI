"""Chunking, stored metadata and the retrieval rules."""
import pytest

from backend import retriever
from backend.chunking import chunk_text, window_transcript
from tests.conftest import make_pdf


def test_chunks_respect_size_and_never_lose_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=40)
    assert len(chunks) > 5 and all(len(c) <= 360 for c in chunks)
    assert "Sentence number 0 is here." in chunks[0] and "Sentence number 199 is here." in chunks[-1]


def test_short_text_is_one_chunk_and_empty_text_is_none():
    assert chunk_text("Short.") == ["Short."]
    assert chunk_text("  \n ") == []


def test_chunks_prefer_sentence_boundaries():
    text = "Alpha beta gamma. " * 30
    for chunk in chunk_text(text, size=100, overlap=0):
        assert chunk.endswith(".")


def test_every_chunk_stores_required_metadata(manager):
    manager.add_file("doc.pdf", make_pdf("Page one about rivers.", "Page two about mountains."))
    data = manager.store.collection.get(include=["metadatas"])
    for meta in data["metadatas"]:
        assert set(meta) >= {"document_id", "filename", "source_type", "page_number",
                             "video_timestamp", "chunk_id"}
        assert meta["filename"] == "doc.pdf" and meta["source_type"] == "pdf"
        assert meta["video_timestamp"] == -1            # not a video
    assert {m["page_number"] for m in data["metadatas"]} == {1, 2}


def test_top_k_is_respected(manager):
    manager.add_file("a.txt", ("Solar panels convert sunlight. " * 60).encode())
    manager.add_file("b.txt", ("Solar farms need sunlight. " * 60).encode())
    found = retriever.retrieve(manager.store, "solar sunlight", manager.ids(), top_k=3)
    assert 1 <= len(found) <= 3
    assert [e.label for e in found] == [f"S{i}" for i in range(1, len(found) + 1)]
    assert [e.distance for e in found] == sorted(e.distance for e in found)


def test_unrelated_question_retrieves_nothing(manager):
    manager.add_file("a.txt", b"Solar panels convert sunlight into electricity.")
    assert retriever.retrieve(manager.store, "zzqx qwerty", manager.ids()) == []


def test_no_sources_or_blank_question(manager):
    assert retriever.retrieve(manager.store, "anything", []) == []
    manager.add_file("a.txt", b"Solar panels convert sunlight.")
    assert retriever.retrieve(manager.store, "   ", manager.ids()) == []


def test_distance_threshold_is_configurable(manager):
    manager.add_file("a.txt", b"Solar panels convert sunlight into electricity.")
    assert retriever.retrieve(manager.store, "solar electricity", manager.ids(), max_distance=0.0) == []
    assert retriever.retrieve(manager.store, "solar electricity", manager.ids(), max_distance=0.9)


def test_retrieval_only_looks_inside_the_given_sources(manager):
    manager.add_file("a.txt", b"Octopuses have three hearts.")
    manager.add_file("b.txt", b"Penguins live in the southern hemisphere.")
    only_b = [s.document_id for s in manager.sources if s.filename == "b.txt"]
    assert retriever.retrieve(manager.store, "octopus hearts", only_b) == []


def test_video_windows_keep_real_times():
    lines = [("hello there", 0.0, 4.0), ("this is RAG", 4.0, 9.0), ("more", 50.0, 55.0), ("end", 56.0, 60.0)]
    chunks = window_transcript(lines, window_seconds=45)
    # the 41-second silence between 9s and 50s starts a new chunk, so 50s is not hidden inside the first one
    assert [(c.start, c.end) for c in chunks] == [(0.0, 9.0), (50.0, 60.0)]


def test_a_timestamp_never_points_before_the_words_it_refers_to():
    lines = [("intro", 0.0, 30.0), ("RAG means retrieval augmented generation", 85.0, 100.0)]
    chunks = window_transcript(lines)
    rag = next(c for c in chunks if "RAG" in c.text)
    assert rag.start == 85.0


def test_windows_do_not_exceed_the_length_limit():
    lines = [(f"line {i}", i * 4.0, i * 4.0 + 4.0) for i in range(60)]
    for chunk in window_transcript(lines, window_seconds=30):
        assert chunk.end - chunk.start <= 30 + 4


@pytest.mark.parametrize("k", [2, 3, 5, 6, 8, 10])
def test_single_source_returns_exactly_top_k_passages(manager, k):
    """Regression: with one source, a fixed per-source cap used to make the slider look dead."""
    manager.add_file("lab.pdf", make_pdf(*[f"Quantum entanglement experiment number {n} results." for n in range(14)]))
    found = retriever.retrieve(manager.store, "quantum entanglement experiment results", manager.ids(), top_k=k)
    assert len(found) == k


@pytest.mark.parametrize("k", [2, 6, 10])
def test_several_sources_also_follow_top_k(manager, k):
    for name in ("a", "b", "c"):
        manager.add_file(f"{name}.pdf", make_pdf(*[f"Quantum entanglement experiment {name}{n} results." for n in range(6)]))
    found = retriever.retrieve(manager.store, "quantum entanglement experiment results", manager.ids(), top_k=k)
    assert len(found) == k
    assert len({e.filename for e in found}) == min(3, k)        # still spread over the sources


def test_default_top_k_comes_from_settings_not_a_literal(manager, monkeypatch):
    from config import settings

    manager.add_file("lab.pdf", make_pdf(*[f"Quantum entanglement experiment number {n} results." for n in range(14)]))
    monkeypatch.setattr(settings, "TOP_K", 4)
    assert len(retriever.retrieve(manager.store, "quantum entanglement experiment results", manager.ids())) == 4
