"""Shared test helpers.

Tests use a tiny fake embedding model (hashed bag-of-words) and a fake Groq client, so they run in
seconds with no model download, no API key and no network.
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import chromadb
import numpy as np
import pytest
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import llm_client  # noqa: E402
from backend.embeddings import Embedder  # noqa: E402
from backend.source_manager import SourceManager  # noqa: E402
from backend.vector_store import SessionRegistry, VectorStore  # noqa: E402

DIMS = 512
STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "what", "how", "does", "do",
        "it", "its", "for", "on", "this", "that", "with", "as", "by", "from", "at", "be"}


def fake_vector(text):
    vec = np.zeros(DIMS)
    for word in text.lower().split():
        word = word.strip(".,?!:;()\"'[]")
        if word in STOP:
            continue
        if len(word) > 3 and word.endswith("s"):        # crude stemming: "means" ~ "mean"
            word = word[:-1]
        if word:
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
    return vec


class FakeModel:
    """Same .encode() interface as SentenceTransformer."""

    def encode(self, texts):
        return np.array([fake_vector(t) for t in texts])


def make_pdf(*pages):
    """A real PDF with one text line per page (text can be long; it is wrapped)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, invariant=1)
    for text in pages:
        y = 800
        words, line = text.split(), ""
        for word in words:
            if len(line) + len(word) > 90:
                c.drawString(40, y, line)
                y -= 14
                line = ""
            line = (line + " " + word).strip()
        c.drawString(40, y, line)
        c.showPage()
    c.save()
    return buf.getvalue()


def blank_pdf():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, invariant=1)
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def chroma_client():
    return chromadb.EphemeralClient()


@pytest.fixture
def embedder():
    return Embedder(model=FakeModel())


@pytest.fixture
def store(chroma_client, embedder, request):
    session_id = hashlib.md5(request.node.nodeid.encode()).hexdigest()
    s = VectorStore(chroma_client, session_id, embedder, SessionRegistry())
    yield s
    s.drop_session()


@pytest.fixture
def manager(store):
    return SourceManager(store)


@pytest.fixture
def fake_groq(monkeypatch):
    """Replace the Groq chat call. Set fake_groq.reply to the text the 'model' returns."""
    class Fake:
        reply = json.dumps({"sufficient": True, "answer": "ok", "source_findings": [],
                            "conflict": {"status": "none"}})
        calls = []

    fake = Fake()
    fake.calls = []

    def chat(messages, json_mode=False, temperature=0.1):
        fake.calls.append(messages)
        return fake.reply

    monkeypatch.setattr(llm_client, "chat", chat)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return fake
