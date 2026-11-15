"""Groq API calls (chat and speech-to-text). The key comes from secrets/environment only."""
import requests

from config import settings
from utils.error_handler import LLMError, MissingAPIKeyError

_RETRY_STATUS = {429, 500, 502, 503, 504}


def api_key():
    return settings.get_secret("GROQ_API_KEY")


def has_api_key():
    return bool(api_key())


def _headers():
    key = api_key()
    if not key:
        raise MissingAPIKeyError(
            "No Groq API key is configured. Add GROQ_API_KEY to Streamlit secrets or your environment."
        )
    return {"Authorization": f"Bearer {key}"}


def _explain(response):
    code = response.status_code
    if code in (401, 403):
        return "The Groq API key was rejected. Check that GROQ_API_KEY is correct."
    if code == 429:
        return "The Groq free-tier rate limit was reached. Wait a minute and try again."
    if code == 413:
        return "The request was too large for the AI service."
    return f"The AI service returned an error (HTTP {code}). Please try again."


def _post(url, retries=1, **kwargs):
    last = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=_headers(), timeout=settings.LLM_TIMEOUT_SECONDS, **kwargs)
        except requests.exceptions.Timeout:
            last = LLMError("The AI service took too long to respond. Please try again.")
            continue
        except requests.exceptions.RequestException:
            last = LLMError("Could not reach the AI service. Check your internet connection.")
            continue
        if response.status_code in _RETRY_STATUS and attempt < retries:
            last = LLMError(_explain(response))
            continue
        if response.status_code >= 400:
            raise LLMError(_explain(response))
        return response
    raise last


def chat(messages, json_mode=False, temperature=0.1):
    """Send chat messages to Groq and return the reply text."""
    payload = {
        "model": settings.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": settings.MAX_ANSWER_TOKENS,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = _post(settings.GROQ_CHAT_URL, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError("The AI service returned an unexpected response. Please try again.")


def transcribe(filename, data):
    """Speech-to-text with timestamps. Returns [(text, start, end)]."""
    response = _post(
        settings.GROQ_STT_URL,
        files={"file": (filename, data)},
        data={"model": settings.GROQ_STT_MODEL, "response_format": "verbose_json"},
    )
    try:
        segments = response.json().get("segments") or []
        return [(s["text"], float(s["start"]), float(s["end"])) for s in segments if s.get("text")]
    except (ValueError, KeyError, TypeError):
        raise LLMError("The speech-to-text service returned an unexpected response.")
