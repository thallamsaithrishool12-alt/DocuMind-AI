"""Embeddings with sentence-transformers. The model loads once and is reused."""
from typing import List

from config import settings
from utils.error_handler import EmbeddingError


class Embedder:
    """Wraps any object with .encode(texts). Pass `model=` to inject a fake one in tests."""

    def __init__(self, model=None, model_name=None):
        self._model = model
        self._model_name = model_name or settings.EMBED_MODEL

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception:
                raise EmbeddingError(
                    "The embedding model could not be loaded. Check your internet connection "
                    "(it is downloaded on first use) and try again."
                )
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = []
        try:
            for start in range(0, len(texts), settings.EMBED_BATCH_SIZE):
                batch = texts[start:start + settings.EMBED_BATCH_SIZE]
                encoded = model.encode(batch, normalize_embeddings=True) \
                    if _accepts_normalize(model) else model.encode(batch)
                vectors.extend(encoded.tolist())
        except EmbeddingError:
            raise
        except Exception:
            raise EmbeddingError("Creating embeddings failed. Please try again.")
        return vectors

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


def _accepts_normalize(model):
    import inspect

    try:
        return "normalize_embeddings" in inspect.signature(model.encode).parameters
    except (TypeError, ValueError):
        return False
