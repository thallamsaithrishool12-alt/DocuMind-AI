"""ChromaDB storage with one collection per session (session isolation).

Every chunk carries: document_id, filename, source_type, page_number,
video_timestamp (start second, -1 if not a video), end_timestamp, chunk_id, video_id.
Chroma metadata cannot hold None, so "not applicable" is stored as 0 / -1 / "".
"""
import threading
import time
from typing import Dict, List, Optional

from backend.models import Chunk, ProcessedSource
from config import settings
from utils.error_handler import VectorStoreError

_NOT_A_VIDEO = -1


def make_client():
    """In-memory Chroma: uploaded private documents are never written to disk."""
    import chromadb

    return chromadb.EphemeralClient()


class SessionRegistry:
    """Remembers when each session was last used, so idle sessions can be deleted."""

    def __init__(self):
        self._last_used: Dict[str, float] = {}
        self._lock = threading.Lock()

    def touch(self, session_id):
        with self._lock:
            self._last_used[session_id] = time.time()

    def forget(self, session_id):
        with self._lock:
            self._last_used.pop(session_id, None)

    def stale_sessions(self, ttl_hours):
        cutoff = time.time() - ttl_hours * 3600
        with self._lock:
            return [s for s, t in self._last_used.items() if t < cutoff]


def cleanup_stale_sessions(client, registry, ttl_hours=None):
    """Delete the collections of sessions idle for longer than the time-to-live."""
    removed = 0
    for session_id in registry.stale_sessions(ttl_hours or settings.SESSION_TTL_HOURS):
        try:
            client.delete_collection(f"dm_{session_id}")
        except Exception:
            pass
        registry.forget(session_id)
        removed += 1
    return removed


class VectorStore:
    def __init__(self, client, session_id, embedder, registry: Optional[SessionRegistry] = None):
        self.client = client
        self.session_id = session_id
        self.embedder = embedder
        self.registry = registry
        self.name = f"dm_{session_id}"
        self._collection = None
        self._touch()

    # ----- internals -----
    def _touch(self):
        if self.registry:
            self.registry.touch(self.session_id)

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.name, metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                raise VectorStoreError("The document database could not be opened. Please reload the page.")
        return self._collection

    # ----- writing -----
    def add_source(self, source: ProcessedSource) -> int:
        """Index one source. Existing chunks of the same document are replaced (no stale entries)."""
        self._touch()
        texts = [c.text for c in source.chunks]
        embeddings = self.embedder.embed(texts)          # may raise EmbeddingError before we touch the DB
        try:
            self.collection.delete(where={"document_id": source.document_id})
            self.collection.add(
                ids=[f"{source.document_id}:{i}" for i in range(len(texts))],
                documents=texts,
                embeddings=embeddings,
                metadatas=[self._metadata(source, c, i) for i, c in enumerate(source.chunks)],
            )
        except Exception:
            self._safe_delete(source.document_id)
            raise VectorStoreError("Saving the document to the database failed. Please try again.")
        return len(texts)

    @staticmethod
    def _metadata(source: ProcessedSource, chunk: Chunk, index: int):
        return {
            "document_id": source.document_id,
            "filename": source.filename,
            "source_type": source.source_type,
            "page_number": int(chunk.page) if chunk.page else 0,
            "video_timestamp": int(chunk.start) if chunk.start is not None else _NOT_A_VIDEO,
            "end_timestamp": int(chunk.end) if chunk.end is not None else _NOT_A_VIDEO,
            "chunk_id": f"{source.document_id}:{index}",
            "video_id": source.video_id or "",
        }

    def _safe_delete(self, document_id):
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass

    def delete_source(self, document_id: str) -> None:
        """Remove only this document's chunks, then verify they are gone."""
        self._touch()
        try:
            self.collection.delete(where={"document_id": document_id})
            left = self.collection.get(where={"document_id": document_id}, include=[])
        except Exception:
            raise VectorStoreError("The document could not be removed. Please try again.")
        if left["ids"]:
            raise VectorStoreError("The document could not be fully removed. Please try again.")

    def drop_session(self):
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self._collection = None
        if self.registry:
            self.registry.forget(self.session_id)

    # ----- reading -----
    def count(self, document_id: Optional[str] = None) -> int:
        try:
            if document_id is None:
                return self.collection.count()
            return len(self.collection.get(where={"document_id": document_id}, include=[])["ids"])
        except Exception:
            raise VectorStoreError("The document database could not be read.")

    def query_source(self, query_embedding, document_id: str, n_results: int):
        """Nearest chunks inside ONE document, as dicts with their metadata and distance."""
        self._touch()
        available = self.count(document_id)
        if available == 0:
            return []
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, available),
                where={"document_id": document_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            raise VectorStoreError("Searching the document database failed. Please try again.")
        return [
            {"text": text, "metadata": meta, "distance": float(dist)}
            for text, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
