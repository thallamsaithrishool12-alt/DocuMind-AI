"""Builds the prompt from retrieved evidence, calls Groq, validates the result."""
import re
from typing import Dict, List, Optional

from backend import conflict_detector, llm_client, retriever
from backend.models import Answer, ConflictReport, Evidence
from config import settings
from utils.helpers import truncate


def _history_messages(history: Optional[List[Dict]]):
    messages = []
    for turn in (history or [])[-settings.HISTORY_TURNS * 2:]:
        role = turn.get("role")
        content = truncate(strip_general_label(str(turn.get("content", ""))), 600)
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


def build_messages(question: str, evidence: List[Evidence], history=None):
    evidence_block = conflict_detector.format_evidence(evidence, settings.MAX_CONTEXT_CHARS)
    user = (
        f"Evidence passages:\n\n{evidence_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the evidence above and reply with the JSON object described in the rules."
    )
    return (
        [{"role": "system", "content": conflict_detector.SYSTEM_PROMPT}]
        + _history_messages(history)
        + [{"role": "user", "content": user}]
    )


def not_found() -> Answer:
    return Answer(text=settings.NOT_FOUND_MESSAGE, sufficient=False, mode="not_found")


_LABEL = re.compile(r"^\s*(?:[*_#>\-\s]*general knowledge[*_\s]*(?:\(\s*not found in your uploaded sources\s*\))?[*_\s:.\-]*)+", re.I)

GENERAL_PROMPT = """You are DocuMind AI, a helpful assistant. The user's uploaded sources do not contain the answer, so answer from your general knowledge.
- Be accurate and concise (usually under 150 words) unless the user asks for detail. Use short paragraphs or a short list; avoid big tables.
- Interpret obvious spelling mistakes from context (for example 'lamgchain' means 'LangChain'). If the intended meaning is unclear, ask one short clarifying question.
- Do not write a label such as "General knowledge" yourself (the app adds it). Never invent document citations and never say the answer came from the uploaded files."""


def strip_general_label(text: str) -> str:
    """Remove a 'General knowledge (not found...)' label the model may have copied from earlier turns."""
    return _LABEL.sub("", text or "").strip()


def _general_knowledge_answer(question: str, history=None) -> Answer:
    """Answer from the model's own knowledge. The app shows the label; the stored text has none."""
    messages = (
        [{"role": "system", "content": GENERAL_PROMPT}]
        + _history_messages(history)
        + [{"role": "user", "content": question}]
    )
    text = strip_general_label(llm_client.chat(messages, json_mode=False, temperature=0.2) or "")
    if not text:
        return not_found()
    return Answer(text=text, sufficient=True, evidence=[], structured=False, mode="general")


def _fallback(question, history, allow_general):
    """Hybrid mode: labelled general knowledge. Document-only mode: the fixed 'not found' message."""
    return _general_knowledge_answer(question, history) if allow_general else not_found()


def generate_answer(question: str, evidence: List[Evidence], history=None, allow_general: bool = True) -> Answer:
    """Prefer cited source answers. With nothing usable in the sources: hybrid mode answers from general
    knowledge (clearly labelled); document-only mode says it could not find the answer."""
    if not evidence:
        return _fallback(question, history, allow_general)

    raw = llm_client.chat(build_messages(question, evidence, history), json_mode=True)
    parsed = conflict_detector.parse_analysis(raw, evidence)

    if parsed is None:
        # The model did not return valid JSON. Keep its text, but make no comparison claims.
        text = (raw or "").strip()
        if not text:
            return _fallback(question, history, allow_general)
        return Answer(text=text, sufficient=True, evidence=evidence, structured=False)

    if not parsed["sufficient"] or not parsed["answer"]:
        return _fallback(question, history, allow_general)

    return Answer(
        text=parsed["answer"],
        sufficient=True,
        findings=parsed["findings"],
        conflict=parsed["conflict"],
        evidence=evidence,
    )


def answer_question(manager, question: str, history=None, top_k: Optional[int] = None,
                    allow_general: bool = True) -> Answer:
    """Search ALL sources of the session (retrieving `top_k` passages), then answer from what was found."""
    evidence = retriever.retrieve(manager.store, question, manager.ids(), top_k=top_k)
    return generate_answer(question, evidence, history, allow_general)
