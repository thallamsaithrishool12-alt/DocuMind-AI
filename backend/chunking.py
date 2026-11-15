"""Splitting text into chunks, and grouping transcript lines into time windows."""
import re
from typing import List

from backend.models import Chunk, Segment
from config import settings

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text, size=None, overlap=None):
    """Split text into chunks of about `size` characters.

    Paragraph and sentence boundaries are preferred so a chunk reads naturally;
    a small overlap keeps context that sits across a boundary.
    """
    size = size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            pieces.append(paragraph)
        else:
            for sentence in _SENTENCE_END.split(paragraph):
                # a single very long sentence is cut hard
                while len(sentence) > size:
                    pieces.append(sentence[:size])
                    sentence = sentence[size - overlap:]
                if sentence.strip():
                    pieces.append(sentence.strip())

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = (tail + " " + piece).strip() if tail else piece
        else:
            current = (current + "\n" + piece).strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


def segments_to_chunks(segments: List[Segment]) -> List[Chunk]:
    """Chunk every page/section separately so each chunk keeps its page number."""
    chunks = []
    for segment in segments:
        for piece in chunk_text(segment.text):
            chunks.append(Chunk(text=piece, page=segment.page, start=segment.start, end=segment.end))
    return chunks


def window_transcript(lines, window_seconds=None, max_chars=None, max_gap=None) -> List[Chunk]:
    """Group transcript lines [(text, start, end)] into chunks of at most about N seconds.

    Every chunk keeps the real start and end time of the lines it contains. A new chunk also starts after a
    pause (gap) in the transcript, so a timestamp never points at silence before the words it refers to.
    """
    window_seconds = window_seconds or settings.VIDEO_WINDOW_SECONDS
    max_chars = max_chars or settings.VIDEO_MAX_CHARS
    max_gap = settings.VIDEO_MAX_GAP_SECONDS if max_gap is None else max_gap
    chunks, texts = [], []
    start = end = None

    def flush():
        nonlocal texts, start, end
        if texts:
            chunks.append(Chunk(text=" ".join(texts), start=start, end=end))
        texts, start, end = [], None, None

    for text, line_start, line_end in lines:
        text = " ".join((text or "").split())
        if not text:
            continue
        if texts:
            too_long = (line_end - start) > window_seconds
            too_big = sum(len(t) for t in texts) + len(text) > max_chars
            after_pause = (line_start - end) > max_gap
            if too_long or too_big or after_pause:
                flush()
        if start is None:
            start = line_start
        texts.append(text)
        end = line_end
    flush()
    return chunks
