"""Scenarios 5, 6 and 15: removing and replacing documents, plus index hygiene and session isolation."""
import pytest

from backend import retriever
from backend.source_manager import SourceManager
from backend.vector_store import VectorStore, cleanup_stale_sessions
from tests.conftest import make_pdf
from utils.error_handler import UploadLimitError, VectorStoreError

FACTS = {
    1: "The Zorblax reactor operates at 450 degrees.",
    2: "Axolotls regenerate lost limbs completely.",
    3: "The Quillon bridge spans 2 kilometres over the bay.",
    4: "Mount Fenwick is 3000 metres above sea level.",
    5: "Nebula cakes require exactly three eggs.",
}


@pytest.fixture
def five(manager):
    for n, fact in FACTS.items():
        manager.add_file(f"pdf{n}.pdf", make_pdf(fact))
    return manager


def test_remove_pdf_3_keeps_the_others(five):                           # scenario 5
    doc3 = next(s for s in five.sources if s.filename == "pdf3.pdf")
    five.remove(doc3.document_id)
    assert [s.filename for s in five.sources] == ["pdf1.pdf", "pdf2.pdf", "pdf4.pdf", "pdf5.pdf"]
    assert five.store.count(doc3.document_id) == 0
    for s in five.sources:
        assert five.store.count(s.document_id) == s.chunk_count          # untouched


def test_replacement_pdf_can_be_added_and_nothing_is_replaced(five):     # scenario 6
    doc3 = next(s for s in five.sources if s.filename == "pdf3.pdf")
    five.remove(doc3.document_id)
    five.add_file("replacement.pdf", make_pdf("The Harbour tunnel is 9 kilometres long."))
    names = [s.filename for s in five.sources]
    assert len(names) == 5 and "replacement.pdf" in names
    assert {"pdf1.pdf", "pdf2.pdf", "pdf4.pdf", "pdf5.pdf"} <= set(names)
    found = retriever.retrieve(five.store, "How long is the harbour tunnel?", five.ids())
    assert found[0].filename == "replacement.pdf"
    with pytest.raises(UploadLimitError):                                # still capped at five
        five.add_file("extra.pdf", make_pdf("Extra document."))


def test_a_removed_pdf_can_be_uploaded_again(five):
    doc = next(s for s in five.sources if s.filename == "pdf2.pdf")
    five.remove(doc.document_id)
    five.add_file("pdf2.pdf", make_pdf(FACTS[2]))
    assert five.count("pdf") == 5


def test_removed_content_is_never_retrieved_again(five):                 # scenario 15
    before = retriever.retrieve(five.store, "At what temperature does the Zorblax reactor operate?", five.ids())
    assert before and before[0].filename == "pdf1.pdf"
    five.remove(before[0].document_id)
    after = retriever.retrieve(five.store, "At what temperature does the Zorblax reactor operate?", five.ids())
    assert all(e.filename != "pdf1.pdf" for e in after)
    assert all("Zorblax" not in e.text for e in after)
    # even a search that names the removed id directly finds nothing
    assert retriever.retrieve(five.store, "Zorblax reactor", [before[0].document_id]) == []
    assert "Zorblax" not in " ".join(five.store.collection.get()["documents"])


def test_removing_an_unknown_id_is_harmless(manager):
    manager.add_file("a.txt", b"Some text about turtles.")
    manager.remove("doc_does_not_exist")
    assert manager.count("document") == 1


def test_remove_failure_is_reported_and_source_stays_listed(manager, monkeypatch):
    info = manager.add_file("a.txt", b"Some text about turtles.")
    monkeypatch.setattr(manager.store, "delete_source",
                        lambda _id: (_ for _ in ()).throw(VectorStoreError("could not remove")))
    with pytest.raises(VectorStoreError):
        manager.remove(info.document_id)
    assert manager.get(info.document_id) is not None


def test_reindexing_the_same_document_leaves_no_stale_chunks(manager):
    from backend.document_processor import process_file

    processed = process_file("a.pdf", make_pdf("Turtles live a long time. " * 5))
    first = manager.store.add_source(processed)
    second = manager.store.add_source(processed)
    assert first == second == manager.store.count(processed.document_id)


def test_sessions_cannot_see_each_others_documents(chroma_client, embedder):
    a = SourceManager(VectorStore(chroma_client, "session_a_test", embedder))
    b = SourceManager(VectorStore(chroma_client, "session_b_test", embedder))
    try:
        a.add_file("secret.txt", b"The launch code is swordfish.")
        b.add_file("public.txt", b"Bananas are rich in potassium.")
        assert retriever.retrieve(b.store, "launch code swordfish", b.ids()) == []
        assert retriever.retrieve(b.store, "launch code swordfish", a.ids()) == []   # wrong collection too
        assert retriever.retrieve(a.store, "launch code swordfish", a.ids())
        assert "swordfish" not in " ".join(b.store.collection.get()["documents"])
    finally:
        a.store.drop_session()
        b.store.drop_session()


def test_idle_sessions_are_cleaned_up(chroma_client, embedder):
    from backend.vector_store import SessionRegistry

    registry = SessionRegistry()
    store = VectorStore(chroma_client, "idle_session_test", embedder, registry)
    SourceManager(store).add_file("a.txt", b"temporary text about owls")
    registry._last_used["idle_session_test"] -= 10 * 3600
    assert cleanup_stale_sessions(chroma_client, registry, ttl_hours=6) == 1
    assert "dm_idle_session_test" not in [c.name for c in chroma_client.list_collections()]
