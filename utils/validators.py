"""Input checks. They run in the backend, so the UI cannot bypass them."""
import os
import re
from urllib.parse import parse_qs, urlparse

from config import settings
from utils.error_handler import (
    FileTooLargeError,
    InvalidYouTubeURLError,
    UnsupportedFileError,
)

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


def file_extension(filename):
    return os.path.splitext(filename or "")[1].lower()


def classify_file(filename):
    """Return 'pdf', 'document' or 'audio'; raise for anything else."""
    ext = file_extension(filename)
    if ext == ".pdf":
        return "pdf"
    if ext in settings.DOCUMENT_EXTENSIONS:
        return "document"
    if ext in settings.AUDIO_EXTENSIONS:
        return "audio"
    allowed = ", ".join(sorted(settings.DOCUMENT_EXTENSIONS))
    raise UnsupportedFileError(
        f"'{filename}' is not supported. Supported document formats: {allowed.upper().replace('.', '')}."
    )


def validate_size(filename, num_bytes, kind):
    limit_mb = settings.MAX_AUDIO_MB if kind == "audio" else settings.MAX_FILE_MB
    if num_bytes > limit_mb * 1024 * 1024:
        raise FileTooLargeError(f"'{filename}' is larger than {limit_mb} MB.")
    if num_bytes == 0:
        raise FileTooLargeError(f"'{filename}' is empty.")


def extract_youtube_id(url):
    """Return the 11-character video id, or raise InvalidYouTubeURLError."""
    url = (url or "").strip()
    if not url:
        raise InvalidYouTubeURLError("Please paste a YouTube link.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    video_id = None
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host in _YT_HOSTS:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live", "v"}:
                video_id = parts[1]
    if not video_id or not _VIDEO_ID.match(video_id):
        raise InvalidYouTubeURLError(
            "That does not look like a valid YouTube video link. "
            "Example: https://www.youtube.com/watch?v=XXXXXXXXXXX"
        )
    return video_id


def youtube_link(video_id, seconds=0):
    """Link that opens the video at the given second."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"
