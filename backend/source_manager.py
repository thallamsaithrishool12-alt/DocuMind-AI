"""The list of sources in one session: adding, removing, limits and duplicates.

Limits are enforced HERE (the backend), so the UI cannot bypass them.
"""
from typing import Dict, List

from backend import document_processor
from backend.models import ProcessedSource, SourceInfo
from backend.vector_store import VectorStore
from config import settings
from utils.error_handler import DuplicateSourceError, UploadLimitError
from utils.validators import extract_youtube_id


class SourceManager:
    def __init__(self, store: VectorStore):
        self.store = store
        self._sources: Dict[str, SourceInfo] = {}

    # ----- reading -----
    @property
    def sources(self) -> List[SourceInfo]:
        return list(self._sources.values())

    def ids(self) -> List[str]:
        return list(self._sources)

    def get(self, document_id):
        return self._sources.get(document_id)

    def count(self, source_type: str) -> int:
        return sum(1 for s in self._sources.values() if s.source_type == source_type)

    def video_count(self) -> int:
        return self.count("youtube") + self.count("audio")

    def pdf_slots_left(self) -> int:
        return settings.MAX_PDFS - self.count("pdf")

    # ----- limits -----
    def _check_limits(self, source_type: str):
        if source_type == "pdf" and self.count("pdf") >= settings.MAX_PDFS:
            raise UploadLimitError(settings.PDF_LIMIT_ERROR)
        if source_type == "document" and self.count("document") >= settings.MAX_OTHER_FILES:
            raise UploadLimitError(f"You can add at most {settings.MAX_OTHER_FILES} TXT/MD/DOCX/CSV files.")
        if source_type in ("youtube", "audio") and self.video_count() >= settings.MAX_VIDEOS:
            raise UploadLimitError(f"You can add at most {settings.MAX_VIDEOS} videos or audio files.")

    @staticmethod
    def _kind_to_type(kind: str) -> str:
        return {"pdf": "pdf", "document": "document", "audio": "audio"}[kind]

    # ----- adding -----
    def add_file(self, filename: str, data: bytes) -> SourceInfo:
        kind = document_processor.detect_kind(filename)          # unsupported -> clear error
        self._check_limits(self._kind_to_type(kind))              # limit checked before any heavy work
        return self._index(document_processor.process_file(filename, data))

    def add_youtube(self, url: str) -> SourceInfo:
        self._check_limits("youtube")
        video_id = extract_youtube_id(url)                        # invalid link -> clear error
        if f"yt_{video_id}" in self._sources:
            raise DuplicateSourceError("This video has already been added.")
        return self._index(document_processor.process_youtube(url))

    def _index(self, processed: ProcessedSource) -> SourceInfo:
        if processed.document_id in self._sources:
            raise DuplicateSourceError(f"'{processed.filename}' has already been added.")
        self._check_limits(processed.source_type)
        chunk_count = self.store.add_source(processed)           # only this source is indexed
        info = SourceInfo(
            document_id=processed.document_id,
            filename=processed.filename,
            source_type=processed.source_type,
            chunk_count=chunk_count,
            size_bytes=processed.size_bytes,
            page_count=processed.page_count,
            video_id=processed.video_id,
            duration=processed.duration,
        )
        self._sources[info.document_id] = info
        return info

    # ----- removing -----
    def remove(self, document_id: str) -> None:
        """Delete only this source's chunks. Other sources are untouched."""
        if document_id not in self._sources:
            return
        self.store.delete_source(document_id)                    # raises if it could not be removed
        del self._sources[document_id]

    def clear(self):
        for document_id in list(self._sources):
            self.remove(document_id)
