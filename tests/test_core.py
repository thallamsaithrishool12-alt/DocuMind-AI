import io
import sys
import types

import pytest
from reportlab.pdfgen import canvas

import rag_core


def blank_pdf():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, invariant=1)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_chunk_text_splits_with_overlap():
    chunks = rag_core.chunk_text("a" * 1200, size=500, overlap=50)
    assert [len(c) for c in chunks] == [500, 500, 300]


def test_chunk_text_ignores_empty_text():
    assert rag_core.chunk_text("   \n  ") == []


def test_read_pdf_returns_numbered_pages(sample_pdf):
    pages = rag_core.read_pdf(sample_pdf)
    assert [number for number, _ in pages] == [1, 2, 3, 4, 5]
    assert "KMeans" in pages[1][1]


def test_file_id_is_stable_and_unique(sample_pdf):
    assert rag_core.file_id(sample_pdf) == rag_core.file_id(sample_pdf)
    assert rag_core.file_id(sample_pdf) != rag_core.file_id(b"something else")


def test_build_index_stores_chunks_with_pages(sample_pdf, client, embedder):
    collection = rag_core.build_index(sample_pdf, client, embedder)
    data = collection.get(include=["metadatas"])
    assert collection.count() > 5
    assert {m["page"] for m in data["metadatas"]} == {1, 2, 3, 4, 5}


def test_build_index_does_not_duplicate(sample_pdf, client, embedder):
    first = rag_core.build_index(sample_pdf, client, embedder).count()
    second = rag_core.build_index(sample_pdf, client, embedder).count()
    assert first == second


def test_build_index_rejects_pdf_without_text(client, embedder):
    with pytest.raises(ValueError, match="No readable text"):
        rag_core.build_index(blank_pdf(), client, embedder)


def test_retrieve_finds_the_right_page(sample_pdf, client, embedder):
    collection = rag_core.build_index(sample_pdf, client, embedder)
    chunks, metas = rag_core.retrieve(
        collection, embedder, "KMeans clustering groups data points", max_distance=1.0
    )
    assert chunks
    assert 2 in {m["page"] for m in metas}


def test_retrieve_returns_nothing_for_unrelated_question(sample_pdf, client, embedder):
    collection = rag_core.build_index(sample_pdf, client, embedder)
    chunks, metas = rag_core.retrieve(collection, embedder, "zzqx")
    assert chunks == [] and metas == []


def test_find_topics_returns_named_groups(sample_pdf, client, embedder):
    collection = rag_core.build_index(sample_pdf, client, embedder)
    topics = rag_core.find_topics(collection)
    assert 1 <= len(topics) <= rag_core.NUM_TOPICS
    assert all(keywords and size > 0 for keywords, size in topics)
    sizes = [size for _, size in topics]
    assert sizes == sorted(sizes, reverse=True)
    assert sum(sizes) == collection.count()


def test_ask_llm_sends_context_and_rules(monkeypatch):
    sent = {}

    def fake_chat(model, messages):
        sent["model"] = model
        sent["messages"] = messages
        return {"message": {"content": "fake answer"}}

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=fake_chat))

    answer = rag_core.ask_llm("What is KMeans?", ["KMeans groups data.", "Second chunk."])

    assert answer == "fake answer"
    assert sent["model"] == rag_core.LLM
    assert "only using the provided PDF context" in sent["messages"][0]["content"]
    assert "KMeans groups data." in sent["messages"][1]["content"]
    assert "What is KMeans?" in sent["messages"][1]["content"]


def test_ask_llm_stream_yields_pieces_in_order(monkeypatch):
    sent = {}

    def fake_chat(model, messages, stream=False):
        sent["stream"] = stream
        return iter([
            {"message": {"content": "Hello "}},
            {"message": {"content": ""}},        # empty pieces are skipped
            {"message": {"content": "world"}},
        ])

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=fake_chat))

    pieces = list(rag_core.ask_llm_stream("Q?", ["context"]))

    assert pieces == ["Hello ", "world"]
    assert sent["stream"] is True


def test_retrieve_scored_returns_text_page_and_distance(sample_pdf, client, embedder):
    collection = rag_core.build_index(sample_pdf, client, embedder)
    exact_text = rag_core.chunk_text(rag_core.read_pdf(sample_pdf)[1][1])[0]

    hits = rag_core.retrieve_scored(collection, embedder, exact_text)

    assert 1 <= len(hits) <= rag_core.TOP_K
    assert {"text", "page", "distance"} <= set(hits[0])
    assert hits[0]["page"] == 2
    assert hits[0]["distance"] < 0.01                       # identical text
    assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)


def test_retrieve_scored_respects_top_k_and_threshold(sample_pdf, client, embedder):
    collection = rag_core.build_index(sample_pdf, client, embedder)
    exact_text = rag_core.chunk_text(rag_core.read_pdf(sample_pdf)[1][1])[0]

    assert len(rag_core.retrieve_scored(collection, embedder, exact_text, top_k=1)) == 1
    assert rag_core.retrieve_scored(collection, embedder, "zzqx", max_distance=0.5) == []


def test_match_strength_labels():
    assert rag_core.match_strength(0.20) == ("Strong", "strong")
    assert rag_core.match_strength(0.55) == ("Good", "good")
    assert rag_core.match_strength(0.69) == ("Weak", "weak")


def test_settings_come_from_day_13_evaluation():
    assert (rag_core.CHUNK_SIZE, rag_core.OVERLAP) == (800, 80)
    assert (rag_core.TOP_K, rag_core.MAX_DISTANCE) == (3, 0.70)
