"""UI tests: run app.py headlessly with Streamlit's AppTest."""
import sys
import types
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import rag_core
from tests.conftest import FakeEmbedder

APP = str(Path(__file__).resolve().parent.parent / "app.py")
FAKE_ANSWER_PIECES = ["FAKE ", "ANSWER"]


class FakeSentenceTransformer(FakeEmbedder):
    def __init__(self, name):
        pass


def sample_button(at):
    return next(b for b in at.sidebar.button if "sample" in b.label.lower())


@pytest.fixture
def fresh_app(monkeypatch, tmp_path):
    """App with the model and LLM faked, using a temporary database."""
    st.cache_resource.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(
        rag_core, "ask_llm_stream", lambda question, chunks: iter(FAKE_ANSWER_PIECES)
    )
    return AppTest.from_file(APP, default_timeout=60)


@pytest.fixture
def loaded_app(fresh_app):
    """App after clicking 'Try the sample PDF'."""
    at = fresh_app.run()
    return sample_button(at).click().run()


def page2_question():
    sample = (Path(__file__).resolve().parent.parent / "sample" / "document.pdf").read_bytes()
    return rag_core.chunk_text(rag_core.read_pdf(sample)[1][1])[0]


def test_starts_with_welcome_screen_and_no_chat(fresh_app):
    at = fresh_app.run()
    assert not at.exception
    assert len(at.chat_input) == 0
    assert any("Chat with your PDF" in m.value for m in at.markdown)
    assert any("Try the sample PDF" in b.label for b in at.sidebar.button)


def test_sample_button_loads_pdf_and_shows_stats(loaded_app):
    at = loaded_app
    assert not at.exception
    assert at.session_state["file_name"] == "document.pdf"
    assert len(at.chat_input) == 1
    labels = {m.label: m.value for m in at.sidebar.metric}
    assert set(labels) == {"Pages", "Chunks", "Topics"}
    assert labels["Pages"] == "5"
    assert int(labels["Chunks"]) >= 5
    assert any(b.key and b.key.startswith("topic_") for b in at.sidebar.button)


def test_question_streams_answer_with_sources_and_badge(loaded_app):
    at = loaded_app.chat_input[0].set_value(page2_question()).run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    texts = [m.value for m in at.chat_message[1].markdown]
    assert "FAKE ANSWER" in texts
    assert any("Strong match" in t for t in texts)
    assert any(e.label.startswith("Sources: page") for e in at.expander)


def test_unrelated_question_is_refused_without_calling_llm(loaded_app, monkeypatch):
    def must_not_be_called(question, chunks):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(rag_core, "ask_llm_stream", must_not_be_called)
    at = loaded_app.chat_input[0].set_value("zzqx").run()
    assert not at.exception
    assert "could not find relevant information" in at.chat_message[1].markdown[0].value
    assert not any(e.label.startswith("Sources:") for e in at.expander)


def test_clicking_a_topic_asks_about_it(loaded_app):
    at = loaded_app.sidebar.button(key="topic_0").click().run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert at.chat_message[0].markdown[0].value.startswith("Tell me about ")


def test_llm_error_shows_friendly_message(loaded_app, monkeypatch):
    def broken(question, chunks):
        raise ConnectionError("ollama is not running")

    monkeypatch.setattr(rag_core, "ask_llm_stream", broken)
    at = loaded_app.chat_input[0].set_value(page2_question()).run()
    assert not at.exception
    assert any("Ollama is running" in e.value or "Ollama" in e.value for e in at.error)


def test_clear_chat_keeps_pdf_but_removes_messages(loaded_app):
    at = loaded_app.chat_input[0].set_value(page2_question()).run()
    assert len(at.chat_message) == 2
    at = at.sidebar.button(key="clear").click().run()
    assert len(at.chat_message) == 0
    assert at.session_state["file_name"] == "document.pdf"


def test_remove_pdf_returns_to_welcome_screen(loaded_app):
    at = loaded_app.sidebar.button(key="remove").click().run()
    assert not at.exception
    assert at.session_state["collection"] is None
    assert len(at.chat_input) == 0
    assert any("Try the sample PDF" in b.label for b in at.sidebar.button)


def test_settings_sliders_change_strictness(loaded_app):
    at = loaded_app
    at.sidebar.slider(key="max_distance").set_value(0.40).run()
    at = at.chat_input[0].set_value(page2_question()).run()
    # An exact copy of the text has distance ~0, so it still matches at the strictest setting
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "FAKE ANSWER" in [m.value for m in at.chat_message[1].markdown]
