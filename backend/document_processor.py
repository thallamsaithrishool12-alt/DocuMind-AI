"""Turns an uploaded file into a ProcessedSource, whatever its format."""
from backend import file_processor, pdf_processor, youtube_processor
from backend.chunking import segments_to_chunks
from backend.models import ProcessedSource
from config import settings
from utils.error_handler import EmptyDocumentError
from utils.helpers import clean_text, content_id
from utils.validators import classify_file, validate_size


def detect_kind(filename: str) -> str:
    """'pdf', 'document' or 'audio'. Raises UnsupportedFileError for anything else."""
    return classify_file(filename)


def process_file(filename: str, data: bytes) -> ProcessedSource:
    kind = classify_file(filename)
    validate_size(filename, len(data), kind)

    if kind == "audio":
        return youtube_processor.process_audio(filename, data)

    page_count = 0
    if kind == "pdf":
        segments, page_count = pdf_processor.extract_pdf(data, filename)
    else:
        segments = file_processor.extract_file(data, filename)

    total = sum(len(s.text) for s in segments)
    if total > settings.MAX_CHARS_PER_SOURCE:
        raise EmptyDocumentError(
            f"'{filename}' is too long to index here (limit {settings.MAX_CHARS_PER_SOURCE:,} characters)."
        )
    chunks = segments_to_chunks(segments)
    if not chunks:
        raise EmptyDocumentError(f"'{filename}' contains no readable text.")
    return ProcessedSource(
        document_id=content_id("doc", data),
        filename=filename,
        source_type="pdf" if kind == "pdf" else "document",
        chunks=chunks,
        size_bytes=len(data),
        page_count=page_count,
    )


def process_youtube(url: str) -> ProcessedSource:
    return youtube_processor.process_youtube(url)
