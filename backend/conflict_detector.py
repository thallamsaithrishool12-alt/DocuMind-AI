"""Prompt rules and validation for multi-source comparison and conflict detection.

The LLM does the reading, but this module never trusts it blindly:
  * findings may only refer to labels (S1, S2, ...) that were really retrieved,
  * filenames, pages and timestamps are always taken from the stored metadata, never from the LLM,
  * a "conflict" is only accepted when at least two different sources have findings.
"""
import json
import re
from typing import Dict, List

from backend.models import ConflictClaim, ConflictReport, Evidence, SourceFinding

VALID_STATUSES = {"none", "contradiction", "different_perspectives"}

SCHEMA_INSTRUCTIONS = """Reply with ONE JSON object and nothing else, in exactly this shape:
{
  "sufficient": true or false,
  "answer": "the combined explanation, using [S1], [S2] labels after the claims they support",
  "source_findings": [{"source": "S1", "finding": "what this passage says about the question"}],
  "conflict": {
    "status": "none" | "contradiction" | "different_perspectives",
    "explanation": "",
    "differences": ["short description of each difference"],
    "claims": [{"source": "S1", "quote": "the conflicting statement copied EXACTLY from the evidence"}],
    "conclusion": "what can and cannot be established from this evidence"
  }
}"""

SYSTEM_PROMPT = f"""You are DocuMind AI. You answer questions ONLY from the numbered evidence passages the user message provides (S1, S2, ...). They come from the user's uploaded documents and videos.

Rules:
1. Use only the evidence. Never use outside knowledge, and never present your own knowledge as coming from a source.
2. If the evidence does not contain enough information to answer, set "sufficient" to false.
3. Cite with the labels exactly as given, like [S1]. Do not invent labels, page numbers, filenames or timestamps.
4. If several sources address the question, explain what EACH says in "source_findings" (one entry per source that is relevant), then combine them in "answer".
5. Conflict check. Compare the findings of different sources on the same factual question:
   - "contradiction": the sources make claims that cannot both be true (different numbers, dates, yes versus no, incompatible definitions of the same fact).
   - "different_perspectives": the sources describe different aspects, scopes or opinions that do not exclude each other. This is NOT a conflict.
   - "none": the sources agree or only one source is relevant.
   A single document can contradict ITSELF: two statements in the same passage or page that cannot both be true are a "contradiction" too.
   For a contradiction, put each conflicting statement in "claims", copied word for word from the evidence (never paraphrased), with its source label. Then list the differences and write a conclusion that says what can and cannot be established. Do NOT pick a winner unless the evidence itself gives a reason (for example a newer date), and then say what that reason is.
6. Keep answers clear and concise. Plain text inside the JSON strings.

{SCHEMA_INSTRUCTIONS}"""


def format_evidence(evidence: List[Evidence], max_chars: int) -> str:
    """Numbered evidence block. Location text comes from metadata only."""
    from utils.helpers import format_timestamp

    blocks, used = [], 0
    for item in evidence:
        if item.page:
            where = f"page {item.page}"
        elif item.start is not None:
            where = f"time {format_timestamp(item.start)}"
        else:
            where = "document"
        block = f"[{item.label}] source: {item.filename} ({where})\n{item.text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _extract_json(raw: str):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return json.loads(raw)
    except ValueError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                return None
    return None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’`]", "", text or "")).strip().lower().rstrip(".,;: ")


def _verify_claims(raw_claims, by_label) -> List[ConflictClaim]:
    """Keep only quotes that really appear in the evidence. Invented quotes are dropped."""
    claims, seen = [], set()
    for entry in raw_claims or []:
        if not isinstance(entry, dict):
            continue
        quote = entry.get("quote")
        label = _as_text(entry.get("source")).strip("[] ").upper()
        if not isinstance(quote, str) or len(_normalise(quote)) < 6:
            continue
        needle = _normalise(quote)
        owner = by_label.get(label)
        if owner is None or needle not in _normalise(owner.text):
            owner = next((e for e in by_label.values() if needle in _normalise(e.text)), None)
            if owner is None:
                continue                                   # not in any retrieved passage: ignored
        if (owner.label, needle) not in seen:
            seen.add((owner.label, needle))
            claims.append(ConflictClaim(label=owner.label, quote=quote.strip(), evidence=owner))
    return claims


def _as_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_analysis(raw: str, evidence: List[Evidence]):
    """Validate the LLM's JSON. Returns (data dict) or None when it is unusable."""
    data = _extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("answer", ""), str):
        return None

    by_label: Dict[str, Evidence] = {e.label: e for e in evidence}

    findings, seen = [], set()
    for entry in data.get("source_findings") or []:
        if not isinstance(entry, dict):
            continue
        label = _as_text(entry.get("source")).strip("[] ").upper()
        text = _as_text(entry.get("finding"))
        if label in by_label and text and label not in seen:     # unknown labels are dropped
            seen.add(label)
            findings.append(SourceFinding(label=label, finding=text, evidence=by_label[label]))

    raw_conflict = data.get("conflict") if isinstance(data.get("conflict"), dict) else {}
    status = _as_text(raw_conflict.get("status")).lower()
    if status not in VALID_STATUSES:
        status = "none"
    claims = _verify_claims(raw_conflict.get("claims"), by_label)
    distinct_sources = {f.evidence.document_id for f in findings}
    # A conflict needs real support: findings from two different sources, OR two different statements
    # that were quoted word for word from the evidence (this also catches a document contradicting itself).
    if status != "none" and len(distinct_sources) < 2 and len(claims) < 2:
        status = "none"
    differences = [d.strip() for d in (raw_conflict.get("differences") or []) if isinstance(d, str) and d.strip()]
    conflict = ConflictReport(
        status=status,
        explanation=_as_text(raw_conflict.get("explanation")),
        differences=differences,
        conclusion=_as_text(raw_conflict.get("conclusion")),
        claims=claims if status != "none" else [],
    )
    return {
        "sufficient": bool(data.get("sufficient", True)),
        "answer": _as_text(data.get("answer")),
        "findings": findings,
        "conflict": conflict,
    }
