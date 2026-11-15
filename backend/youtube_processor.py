"""YouTube transcripts (with timestamps) and audio/video file transcription."""
from typing import List, Tuple

import requests

from backend import llm_client
from backend.chunking import window_transcript
from backend.models import ProcessedSource
from utils.error_handler import EmptyDocumentError, LLMError, TranscriptUnavailableError
from utils.helpers import content_id
from utils.validators import extract_youtube_id

PREFERRED_LANGUAGES = ("en", "en-US", "en-GB", "en-IN")

Lines = List[Tuple[str, float, float]]


def _lines_from_snippets(snippets) -> Lines:
    lines = []
    for item in snippets:
        if isinstance(item, dict):                       # older library versions
            text, start, duration = item.get("text", ""), item.get("start", 0), item.get("duration", 0)
        else:                                            # youtube-transcript-api >= 1.0
            text, start, duration = item.text, item.start, item.duration
        lines.append((text, float(start), float(start) + float(duration)))
    return lines


def fetch_transcript_lines(video_id: str) -> Lines:
    """Download the video's captions. Raises TranscriptUnavailableError with the reason."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api import _errors as yt_errors
    except ImportError:
        raise TranscriptUnavailableError("The youtube-transcript-api package is not installed.")

    try:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=list(PREFERRED_LANGUAGES))
        except yt_errors.NoTranscriptFound:
            transcript_list = api.list(video_id)
            first = next(iter(transcript_list), None)
            if first is None:
                raise
            fetched = first.fetch()
        return _lines_from_snippets(fetched)
    except TranscriptUnavailableError:
        raise
    except Exception as error:  # the library has many specific errors; map them to clear messages
        name = type(error).__name__
        if name in ("TranscriptsDisabled", "NoTranscriptFound"):
            reason = "This video has no captions, so its content cannot be searched."
        elif name in ("IpBlocked", "RequestBlocked"):
            reason = ("YouTube blocked the request (this is common from cloud servers). "
                      "Try again later, or upload the video's audio file instead.")
        elif name in ("VideoUnavailable", "VideoUnplayable", "InvalidVideoId"):
            reason = "This video is unavailable, private, or the link is wrong."
        elif name == "AgeRestricted":
            reason = "This video is age-restricted, so its captions cannot be fetched."
        elif isinstance(error, requests.exceptions.RequestException):
            reason = "Could not reach YouTube. Check your internet connection."
        else:
            reason = "The transcript could not be retrieved."
        raise TranscriptUnavailableError(
            f"{reason} No video content was processed. "
            "You can upload the audio/video file instead if you have it."
        )


def fetch_title(video_id: str) -> str:
    """Best-effort video title (falls back to the id)."""
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        if response.ok:
            return (response.json().get("title") or "").strip() or f"YouTube video {video_id}"
    except Exception:
        pass
    return f"YouTube video {video_id}"


def process_youtube(url: str) -> ProcessedSource:
    video_id = extract_youtube_id(url)
    lines = fetch_transcript_lines(video_id)
    chunks = window_transcript(lines)
    if not chunks:
        raise EmptyDocumentError("The video's transcript is empty.")
    return ProcessedSource(
        document_id=f"yt_{video_id}",
        filename=fetch_title(video_id),
        source_type="youtube",
        chunks=chunks,
        video_id=video_id,
        duration=max(end for _, _, end in lines),
    )


def process_audio(filename: str, data: bytes) -> ProcessedSource:
    """Transcribe an uploaded audio/video file with Groq speech-to-text."""
    try:
        lines = llm_client.transcribe(filename, data)
    except LLMError:
        raise
    chunks = window_transcript(lines)
    if not chunks:
        raise EmptyDocumentError(f"No speech was detected in '{filename}'.")
    return ProcessedSource(
        document_id=content_id("aud", data),
        filename=filename,
        source_type="audio",
        chunks=chunks,
        size_bytes=len(data),
        duration=max(end for _, _, end in lines),
    )
