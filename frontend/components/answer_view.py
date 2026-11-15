"""Draws an assistant answer: text, conflict banner, source comparison, citations, timestamps."""
import html
import re

import streamlit as st

from backend.models import Answer, ConflictClaim, Evidence
from config import settings
from utils.helpers import format_timestamp, truncate
from utils.validators import youtube_link

_LABEL_GROUP = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def number_citations(text):
    """[S1] -> [1], [S1, S2] -> [1, 2], so the text matches the numbered source list."""
    return _LABEL_GROUP.sub(lambda m: "[" + re.sub(r"S(\d+)", r"\1", m.group(1)) + "]", text)


def location_text(item: Evidence) -> str:
    if item.page:
        return f"page {item.page}"
    if item.start is not None:
        end = f"–{format_timestamp(item.end)}" if item.end is not None and item.end > item.start else ""
        return f"{format_timestamp(item.start)}{end}"
    return "whole document"


def location_markdown(item: Evidence) -> str:
    """Page text, or a clickable timestamp that opens the YouTube video at that moment."""
    text = location_text(item)
    if item.video_id and item.start is not None:
        return f"[▶ {text}]({youtube_link(item.video_id, item.start)})"
    return text


def _finding_columns(answer: Answer):
    findings = answer.findings
    columns = st.columns(min(len(findings), 3))
    for index, finding in enumerate(findings):
        with columns[index % len(columns)]:
            item = finding.evidence
            st.markdown(f"**{html.escape(item.filename)}**  \n{location_markdown(item)}")
            st.write(number_citations(finding.finding))


def render_answer(answer: Answer, key: str):
    conflict = answer.conflict

    if answer.mode == "not_found" or not answer.sufficient:
        st.info(answer.text)
        _feedback(key)
        return

    if answer.mode == "general":
        st.markdown('<span class="dm-badge">General knowledge · not found in your uploaded sources</span>',
                    unsafe_allow_html=True)
        st.markdown(answer.text)
        _feedback(key)
        return

    if conflict.status == "contradiction":
        st.warning(settings.CONFLICT_MESSAGE, icon="⚠️")

    st.markdown(number_citations(answer.text))

    multi_source = len({f.evidence.document_id for f in answer.findings}) >= 2
    if multi_source or conflict.status != "none":
        with st.expander("Source comparison", expanded=conflict.status == "contradiction"):
            if multi_source:
                _finding_columns(answer)
            _claims(conflict.claims)
            if conflict.status == "contradiction":
                st.markdown("**Differences**")
                for line in conflict.differences or [conflict.explanation]:
                    if line:
                        st.markdown(f"- {number_citations(line)}")
                if conflict.conclusion:
                    st.markdown(
                        f"**What can and cannot be established:** {number_citations(conflict.conclusion)}"
                    )
            elif conflict.status == "different_perspectives":
                st.info("The sources give different perspectives. This is not a contradiction.")
                for text in (conflict.explanation, conflict.conclusion):
                    if text:
                        st.markdown(number_citations(text))

    if not answer.structured and answer.mode == "document":
        st.caption("Source comparison is not available for this answer.")

    if answer.evidence:
        with st.expander(f"Sources ({len(answer.evidence)})"):
            for number, item in enumerate(answer.evidence, start=1):
                st.markdown(
                    f'<div class="dm-src-title">[{number}] {html.escape(item.filename)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(location_markdown(item))
                st.markdown(
                    f'<div class="dm-snippet">{html.escape(truncate(item.text, 400))}</div>',
                    unsafe_allow_html=True,
                )

    _feedback(key)


def _claims(claims):
    """The conflicting statements, quoted from the sources (each quote was verified against the evidence)."""
    if not claims:
        return
    st.markdown("**Conflicting statements**")
    for claim in claims:
        item = claim.evidence
        st.markdown(f"- “{claim.quote}” — **{html.escape(item.filename)}**, {location_markdown(item)}")


def _feedback(key):
    """Star rating, keyed per answer so it never jumps to another response after a rerun."""
    rating = st.feedback("stars", key=f"rating_{key}")
    if rating is not None:
        st.session_state.setdefault("answer_ratings", {})[key] = rating + 1
        st.caption(f"Thanks for your feedback: {rating + 1}/5")
