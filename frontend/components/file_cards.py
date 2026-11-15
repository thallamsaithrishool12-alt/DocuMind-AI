"""Sidebar file cards: one card per source, each with its own X (remove) button."""
import html

import streamlit as st

from utils.helpers import format_timestamp, human_size

ICONS = {"pdf": "📕", "document": "📄", "youtube": "▶️", "audio": "🎧"}
LABELS = {"pdf": "PDF", "document": "Document", "youtube": "YouTube", "audio": "Audio"}


def _meta(info):
    parts = [LABELS.get(info.source_type, "File")]
    if info.source_type == "pdf" and info.page_count:
        parts.append(f"{info.page_count} page{'s' if info.page_count != 1 else ''}")
    if info.source_type in ("youtube", "audio") and info.duration:
        parts.append(format_timestamp(info.duration))
    if info.size_bytes:
        parts.append(human_size(info.size_bytes))
    parts.append(f"{info.chunk_count} chunk{'s' if info.chunk_count != 1 else ''}")
    return " · ".join(parts)


def render_file_card(info, on_remove):
    """Draw one card. `on_remove(document_id)` runs when the X is clicked."""
    card, button = st.columns([0.84, 0.16], vertical_alignment="center")
    with card:
        st.markdown(
            f'<div class="dm-card"><span class="ico">{ICONS.get(info.source_type, "📄")}</span>'
            f'<div class="txt"><div class="name" title="{html.escape(info.filename)}">'
            f"{html.escape(info.filename)}</div>"
            f'<div class="meta">{html.escape(_meta(info))}</div></div></div>',
            unsafe_allow_html=True,
        )
    with button:
        st.button(
            ":material/close:",
            key=f"remove_{info.document_id}",
            help=f"Remove {info.filename}",
            on_click=on_remove,
            args=(info.document_id,),
        )
