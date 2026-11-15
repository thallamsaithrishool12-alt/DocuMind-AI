import hashlib
import re
import uuid


def new_session_id():
    return uuid.uuid4().hex


def content_id(prefix, data):
    """Stable id from the content: the same file always gets the same id."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return f"{prefix}_{hashlib.sha256(data).hexdigest()[:16]}"


def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def human_size(num_bytes):
    for unit in ("B", "KB", "MB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def clean_text(text):
    """Remove null bytes and collapse runs of blank lines and spaces."""
    text = (text or "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text, limit):
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
