
"""UI tests: the real app.py runs headlessly (Streamlit AppTest) with a fake embedding model and a fake
Groq reply. File *uploading* through the browser widget is covered by the Playwright check described in the
README, because AppTest cannot drive file_uploader."""
import json
import sys
import types
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.conftest import FakeModel, make_pdf

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeSentenceTransformer(FakeModel):
    def __init__(self, name):
        pass


@pytest.fixture
def app(monkeypatch, fake_groq):
    st.cache_resource.clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    at = AppTest.from_file(APP, default_timeout=60)
    return at.run()


def texts(at):
    return [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning] \
        + [i.value for i in at.info]


def test_starts_clean_with_limit_notice_and_disabled_chat(app):
    assert not app.exception
    assert any("Maximum 5 PDFs can be uploaded." in t for t in texts(app))
    assert any("PDFs added: <b>0/5</b>" in t for t in texts(app))
    assert app.chat_input[0].disabled
    assert any("DocuMind AI" in t for t in texts(app))


def test_file_cards_show_with_a_remove_button_each(app):
    manager = app.session_state["manager"]
    ids = [manager.add_file(f"doc{n}.pdf", make_pdf(f"Topic{n} text here.")).document_id for n in range(3)]
    app.run()
    assert any("PDFs added: <b>3/5</b>" in t for t in texts(app))
    for n, doc_id in enumerate(ids):
        assert any(f"doc{n}.pdf" in t for t in texts(app))
        assert app.sidebar.button(key=f"remove_{doc_id}") is not None
    assert not app.chat_input[0].disabled


def test_clicking_x_removes_only_that_file(app):
    manager = app.session_state["manager"]
    ids = [manager.add_file(f"doc{n}.pdf", make_pdf(f"Topic{n} text here.")).document_id for n in range(3)]
    app.run()
    app.sidebar.button(key=f"remove_{ids[1]}").click().run()
    assert not app.exception
    remaining = [s.filename for s in app.session_state["manager"].sources]
    assert remaining == ["doc0.pdf", "doc2.pdf"]
    assert any("PDFs added: <b>2/5</b>" in t for t in texts(app))
    keys = [b.key for b in app.sidebar.button]
    assert f"remove_{ids[1]}" not in keys and f"remove_{ids[0]}" in keys


def test_question_shows_answer_numbered_citations_and_sources(app, fake_groq):
    manager = app.session_state["manager"]
    manager.add_file("zoo.pdf", make_pdf("Pandas eat bamboo every single day."))
    app.run()
    fake_groq.reply = json.dumps({"sufficient": True, "answer": "Pandas eat bamboo [S1].",
                                  "source_findings": [], "conflict": {"status": "none"}})
    app.chat_input[0].set_value("What do pandas eat?").run()
    assert not app.exception
    assistant = app.chat_message[1]
    assert any("Pandas eat bamboo [1]." in m.value for m in assistant.markdown)
    assert any(e.label.startswith("Sources (") for e in app.expander)
    assert any("zoo.pdf" in m.value for m in assistant.markdown)
    assert any(c.value == "page 1" for c in assistant.caption)


def test_conflict_banner_and_comparison_are_displayed(app, fake_groq):
    manager = app.session_state["manager"]
    manager.add_file("a.pdf", make_pdf("The Eiffel Tower is 300 metres tall."))
    manager.add_file("b.pdf", make_pdf("The Eiffel Tower is 450 metres tall."))
    app.run()
    from backend import retriever

    found = retriever.retrieve(manager.store, "How tall is the Eiffel Tower?", manager.ids())
    labels = {e.filename: e.label for e in found}
    fake_groq.reply = json.dumps({
        "sufficient": True, "answer": "The sources disagree.",
        "source_findings": [{"source": labels["a.pdf"], "finding": "300 m"},
                            {"source": labels["b.pdf"], "finding": "450 m"}],
        "conflict": {"status": "contradiction", "differences": ["300 m vs 450 m"], "conclusion": "Cannot tell."}})
    app.chat_input[0].set_value("How tall is the Eiffel Tower?").run()
    assert not app.exception
    assert any("Conflicting information detected in the uploaded documents." in w.value for w in app.warning)
    assert any(e.label == "Source comparison" for e in app.expander)


def test_unanswerable_question_shows_the_refusal(app, fake_groq):
    app.session_state["manager"].add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    app.run()
    app.sidebar.radio(key="answer_mode").set_value("Document only").run()
    app.chat_input[0].set_value("zzqx qwerty").run()
    assert any("could not find sufficient information" in i.value for i in app.info)
    assert fake_groq.calls == []


def test_llm_failure_shows_a_friendly_message_not_a_traceback(app, fake_groq, monkeypatch):
    from backend import llm_client
    from utils.error_handler import LLMError

    app.session_state["manager"].add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    app.run()
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: (_ for _ in ()).throw(LLMError("The AI service took too long.")))
    app.chat_input[0].set_value("solar sunlight panels").run()
    assert not app.exception
    assert any("took too long" in e.value for e in app.error)


def test_clear_conversation_keeps_the_documents(app, fake_groq):
    app.session_state["manager"].add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    app.run()
    app.chat_input[0].set_value("solar sunlight panels").run()
    assert len(app.chat_message) == 2
    clear = next(b for b in app.sidebar.button if b.label == "Clear conversation")
    clear.click().run()
    assert len(app.chat_message) == 0
    assert len(app.session_state["manager"].sources) == 1


def test_missing_api_key_banner(monkeypatch):
    st.cache_resource.clear()
    from config import settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(settings, "get_secret", lambda name, default=None: None)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("No Groq API key found" in w.value for w in at.warning)


def test_two_sessions_do_not_share_documents(monkeypatch, fake_groq):
    st.cache_resource.clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    one = AppTest.from_file(APP, default_timeout=60).run()
    two = AppTest.from_file(APP, default_timeout=60).run()
    one.session_state["manager"].add_file("secret.pdf", make_pdf("The launch code is swordfish."))
    two.run()
    assert one.session_state["session_id"] != two.session_state["session_id"]
    assert two.session_state["manager"].sources == []


def test_slider_defaults_to_6_range_2_to_10_and_shows_its_value(app):
    slider = app.sidebar.slider(key="top_k")
    assert (slider.value, slider.min, slider.max) == (6, 2, 10)
    assert any("up to **6** passages" in c.value for c in app.sidebar.caption)
    app.sidebar.slider(key="top_k").set_value(9).run()
    assert any("up to **9** passages" in c.value for c in app.sidebar.caption)


def test_slider_moves_freely_and_every_new_question_uses_the_new_value(app, fake_groq):
    app.session_state["manager"].add_file(
        "lab.pdf", make_pdf(*[f"Quantum entanglement experiment number {n} results." for n in range(14)]))
    app.run()
    for k in (3, 8, 2, 10, 6, 4):                   # up and down, every value in between is allowed
        app.sidebar.slider(key="top_k").set_value(k).run()
        assert app.sidebar.slider(key="top_k").value == k
        app.chat_input[0].set_value("quantum entanglement experiment results").run()
        assert fake_groq.calls[-1][-1]["content"].count("source: ") == k
        assert any(e.label == f"Sources ({k})" for e in app.expander)


def test_hybrid_is_the_default_mode_and_labels_general_answers_once(app, fake_groq):
    app.session_state["manager"].add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    app.run()
    assert app.sidebar.radio(key="answer_mode").value == "Hybrid"
    fake_groq.reply = "Paris is the capital of France."
    app.chat_input[0].set_value("What is the capital of France?").run()
    badges = [m.value for m in app.chat_message[1].markdown if "General knowledge" in m.value]
    assert len(badges) == 1                                         # shown exactly once
    assert any("Paris is the capital" in m.value for m in app.chat_message[1].markdown)
    assert not any("Source comparison is not available" in c.value for c in app.caption)


def test_general_label_is_not_repeated_after_earlier_general_answers(app, fake_groq):
    """Regression: the label used to be stored in the text, copied by the model, and shown twice."""
    app.session_state["manager"].add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    app.run()
    fake_groq.reply = "Paris is the capital of France."
    app.chat_input[0].set_value("capital of france").run()
    fake_groq.reply = "General knowledge (not found in your uploaded sources)\n\nHyderabad is the capital."
    app.chat_input[0].set_value("capital of telangana").run()
    last = app.chat_message[-1]
    assert sum("General knowledge" in m.value for m in last.markdown) == 1
    assert any(m.value.strip() == "Hyderabad is the capital." for m in last.markdown)
    history_sent = fake_groq.calls[-1]
    assert all("General knowledge" not in m["content"] for m in history_sent[1:-1])


def test_document_only_mode_refuses_instead_of_using_general_knowledge(app, fake_groq):
    app.session_state["manager"].add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    app.run()
    app.sidebar.radio(key="answer_mode").set_value("Document only").run()
    app.chat_input[0].set_value("What is the capital of France?").run()
    assert any("could not find sufficient information" in i.value for i in app.info)
    assert fake_groq.calls == []                                    # the model was never asked


def test_video_success_message_stays_until_the_next_action(app, monkeypatch):
    from backend import youtube_processor

    monkeypatch.setattr(youtube_processor, "fetch_transcript_lines",
                        lambda vid: [("RAG means retrieval augmented generation.", 85.0, 100.0)])
    monkeypatch.setattr(youtube_processor, "fetch_title", lambda vid: "RAG explained")
    app.text_input[0].set_value("https://youtu.be/dQw4w9WgXcQ")
    next(b for b in app.sidebar.button if b.label == "Submit").click().run()
    assert not app.exception
    assert any("Submitted!" in s.value and "RAG explained" in s.value for s in app.success)
    app.run()                                                              # any later rerun keeps it
    assert any("Submitted!" in s.value for s in app.success)
