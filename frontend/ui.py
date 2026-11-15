"""Page layout: sidebar (uploads, file cards, video link) and the chat area."""
import logging
import uuid
from pathlib import Path

import streamlit as st

from backend import answer_generator, llm_client
from config import settings
from frontend.components.answer_view import render_answer
from frontend.components.file_cards import render_file_card
from utils.error_handler import DocuMindError, friendly_message

log = logging.getLogger("documind")
CSS_PATH = Path(__file__).parent / "styles" / "main.css"

EXAMPLES = ["What is this document about?",
            "Summarise the main points", "Compare what my sources say"]


def inject_css():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def notify(kind, text):
    """Queue a message that is shown after the next rerun."""
    st.session_state.notices.append((kind, text))


def show_notices():
    notices, st.session_state.notices = st.session_state.notices, []
    for kind, text in notices:
        if kind == "success":
            # confirmations are brief; errors stay visible
            st.toast(text, icon="✅")
        else:
            {"error": st.error, "warning": st.warning}.get(kind, st.info)(text)


# ---------- actions ----------

def _remove_source(document_id):
    manager = st.session_state.manager
    info = manager.get(document_id)
    try:
        manager.remove(document_id)
        st.session_state.video_status = None
        notify("success", f"Removed {info.filename if info else 'document'}.")
    except DocuMindError as error:
        notify("error", friendly_message(error))
    except Exception:
        log.exception("remove failed")
        notify("error", friendly_message(None))


def _add_files(files, audio=False):
    """Index uploaded files one by one. One failing file never blocks the others."""
    manager = st.session_state.manager
    for uploaded in files:
        try:
            with st.spinner(f"Reading {uploaded.name}…"):
                info = manager.add_file(uploaded.name, uploaded.getvalue())
            notify(
                "success", f"Added {info.filename} ({info.chunk_count} chunk{'s' if info.chunk_count != 1 else ''}).")
        except DocuMindError as error:
            notify("error", friendly_message(error))
        except Exception:
            log.exception("upload failed")
            notify("error", f"'{uploaded.name}' could not be processed.")
    # empties the uploader; our own list is the source of truth
    st.session_state.uploader_key += 1
    st.rerun()


def _add_video(url):
    """Result is kept in session state, so the green/red box stays visible until the next video action."""
    manager = st.session_state.manager
    try:
        with st.spinner("Fetching the transcript…"):
            info = manager.add_youtube(url)
        st.session_state.video_status = (
            "success", f"Submitted! \"{info.filename}\" was added to Sources ({info.chunk_count} sections)."
        )
    except DocuMindError as error:
        st.session_state.video_status = ("error", friendly_message(error))
    except Exception:
        log.exception("youtube failed")
        st.session_state.video_status = (
            "error", "The video could not be processed.")
    st.rerun()


def _clear_chat():
    """Clear only the conversation and its feedback; keep uploaded sources indexed."""
    st.session_state.messages = []
    st.session_state.answer_ratings = {}


# ---------- sidebar ----------

def render_sidebar():
    manager = st.session_state.manager
    with st.sidebar:
        st.markdown(
            '<div class="dm-brand"><div class="dm-logo">🧠</div><div class="dm-title">DocuMind AI</div></div>',
            unsafe_allow_html=True,
        )
        show_notices()

        pdfs = manager.count("pdf")
        st.markdown(
            f'<div class="dm-info"><b>{settings.PDF_LIMIT_INFO}</b><br>'
            f"PDFs added: <b>{pdfs}/{settings.MAX_PDFS}</b></div>",
            unsafe_allow_html=True,
        )

        files = st.file_uploader(
            "Add files",
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help="PDF, TXT, MD, DOCX or CSV. You can add more files at any time.",
            label_visibility="collapsed",
        )
        st.caption("Supported: PDF, TXT, MD, DOCX, CSV")
        if files:
            _add_files(files)

        with st.form("video_form", clear_on_submit=True, border=False):
            box, submit = st.columns([0.66, 0.34], vertical_alignment="bottom")
            url = box.text_input(
                "YouTube link", placeholder="Paste a YouTube link", label_visibility="collapsed")
            submitted = submit.form_submit_button(
                "Submit", use_container_width=True)
        if submitted:
            _add_video(url)
        status = st.session_state.get("video_status")
        if status:
            {"success": st.success, "error": st.error}[status[0]](status[1])

        with st.expander("Transcribe an audio/video file"):
            if not llm_client.has_api_key():
                st.caption("Needs a Groq API key.")
            else:
                st.caption("For videos without captions. Max "
                           f"{settings.MAX_AUDIO_MB} MB.")
                audio = st.file_uploader(
                    "Audio or video file",
                    type=sorted(e.lstrip(".")
                                for e in settings.AUDIO_EXTENSIONS),
                    accept_multiple_files=True,
                    key=f"audio_{st.session_state.uploader_key}",
                    label_visibility="collapsed",
                )
                if audio:
                    _add_files(audio, audio=True)

        st.markdown("##### Sources")
        if not manager.sources:
            st.caption("Nothing added yet.")
        for info in manager.sources:
            render_file_card(info, _remove_source)

        st.divider()
        st.radio(
            "Answer mode",
            ["Hybrid", "Document only"],
            key="answer_mode",
            help="Hybrid: answer from your sources and cite them; if they do not contain the answer, give a "
                 "clearly labelled general-knowledge answer. Document only: answer strictly from your sources.",
        )
        top_k = st.selectbox(
            "Passages per answer",
            # 1 ... 10
            options=list(range(settings.TOP_K_MIN, settings.TOP_K_MAX + 1)),
            index=settings.TOP_K - settings.TOP_K_MIN,                              # default 6
            key="top_k",
            help="Pick how many passages are retrieved across all sources for each question. "
                 "Higher = more complete but slower.",
        )
        st.caption(f"Each question retrieves up to **{top_k}** passages.")
        # not disabled when empty: the sidebar is drawn before the chat, so its state would lag one run behind
        st.button("Clear conversation", on_click=_clear_chat,
                  use_container_width=True)


# ---------- chat ----------

def _empty_state():
    chips = "".join(f'<span class="dm-chip">{c}</span>' for c in EXAMPLES)
    st.markdown(
        '<div class="dm-empty"><h3>Ask questions across your documents and videos</h3>'
        "Add PDFs, documents or a YouTube link in the sidebar, then ask anything.<br><br>"
        f"{chips}</div>",
        unsafe_allow_html=True,
    )


def _history_for_llm():
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


def render_chat():
    manager = st.session_state.manager
    st.markdown(
        '<div class="dm-brand"><div class="dm-title">DocuMind AI</div></div>'
        '<p class="dm-tagline">Multi-source document and video intelligence, with citations.</p>',
        unsafe_allow_html=True,
    )

    if not llm_client.has_api_key():
        st.warning("No Groq API key found. Documents can be indexed, but answers need GROQ_API_KEY "
                   "(see the README).")

    if not st.session_state.messages and not manager.sources:
        _empty_state()

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("answer") is not None:
                render_answer(message["answer"], key=message.get(
                    "answer_id", f"a{index}"))
            elif message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])

    question = st.chat_input(
        "Ask about your sources…" if manager.sources else "Add a source first…",
        disabled=not manager.sources,
    )
    if not question:
        return

    history = _history_for_llm()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your sources…"):
                answer = answer_generator.answer_question(
                    manager, question, history,
                    # the dropdown's current value
                    top_k=st.session_state.get("top_k", settings.TOP_K),
                    allow_general=st.session_state.get(
                        "answer_mode", "Hybrid") == "Hybrid",
                )
            answer_id = f"answer_{uuid.uuid4().hex}"
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text,
                    "answer": answer, "answer_id": answer_id}
            )
            render_answer(answer, key=answer_id)
        except DocuMindError as error:
            text = friendly_message(error)
            st.error(text)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "error": True})
        except Exception:
            log.exception("answering failed")
            text = friendly_message(None)
            st.error(text)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "error": True})
