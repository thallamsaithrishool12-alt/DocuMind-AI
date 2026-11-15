"""All tunable settings in one place. No secrets are stored here."""
import os

APP_NAME = "DocuMind AI"

# ---------- Models ----------
EMBED_MODEL = "all-MiniLM-L6-v2"
# answers; change to "llama-3.1-8b-instant" for speed
GROQ_MODEL = "openai/gpt-oss-120b"
# speech-to-text for audio/video files
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
LLM_TIMEOUT_SECONDS = 60

# ---------- Upload limits (enforced in the backend, not only in the UI) ----------
MAX_PDFS = 5
MAX_OTHER_FILES = 10          # TXT / MD / DOCX / CSV
MAX_VIDEOS = 3                # YouTube videos and transcribed audio/video files
MAX_FILE_MB = 15
MAX_AUDIO_MB = 24             # Groq speech-to-text accepts about 25 MB per file
MAX_CHARS_PER_SOURCE = 600_000

DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4",
                    ".webm", ".mpeg", ".mpga", ".ogg", ".flac"}

# ---------- Chunking ----------
CHUNK_SIZE = 800              # characters per text chunk
CHUNK_OVERLAP = 100
# transcript is grouped into windows of at most about this length
VIDEO_WINDOW_SECONDS = 30
VIDEO_MAX_GAP_SECONDS = 8     # a pause longer than this starts a new window
VIDEO_MAX_CHARS = 1200

# ---------- Retrieval ----------
# DEFAULT number of passages; the sidebar dropdown (1-10) overrides it per question
TOP_K = 6
TOP_K_MIN, TOP_K_MAX = 1, 10
MAX_DISTANCE = 0.75           # cosine distance; passages farther than this are ignored
RELATIVE_MARGIN = 0.40        # ignore passages much worse than the overall best match
EMBED_BATCH_SIZE = 32

# ---------- Answering ----------
HISTORY_TURNS = 4             # previous question/answer pairs given to the LLM
MAX_CONTEXT_CHARS = 9000      # evidence text sent to the LLM
MAX_ANSWER_TOKENS = 900

# ---------- Sessions ----------
SESSION_TTL_HOURS = 6         # idle sessions' vector collections are deleted after this

NOT_FOUND_MESSAGE = (
    "I could not find sufficient information in your uploaded sources to answer this question."
)
CONFLICT_MESSAGE = "Conflicting information detected in the uploaded documents."
PDF_LIMIT_INFO = f"Maximum {MAX_PDFS} PDFs can be uploaded."
PDF_LIMIT_ERROR = f"Maximum upload limit reached. You can upload only {MAX_PDFS} PDF files."


def get_secret(name, default=None):
    """Read a secret from the environment or Streamlit secrets. Never hardcode keys."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default
