"""Shared test helpers.

The tests use a tiny fake embedder (hashed bag-of-words) instead of the real
sentence-transformers model, so they run in seconds without downloads or Ollama.
"""
import hashlib
from pathlib import Path

import chromadb
import numpy as np
import pytest

DIMS = 256
ROOT = Path(__file__).resolve().parent.parent


def fake_vector(text):
    vec = np.zeros(DIMS)
    for word in text.lower().split():
        word = word.strip(".,?!:;()\"'")
        if word:
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
    return vec


class FakeEmbedder:
    """Same interface as SentenceTransformer.encode()."""

    def encode(self, texts):
        if isinstance(texts, str):
            return fake_vector(texts)
        return np.array([fake_vector(t) for t in texts])


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


@pytest.fixture
def sample_pdf():
    return (ROOT / "sample" / "document.pdf").read_bytes()
