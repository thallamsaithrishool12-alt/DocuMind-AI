"""DocuMind AI: entry point. Run with:  streamlit run app.py"""
import streamlit as st

from backend.embeddings import Embedder
from backend.source_manager import SourceManager
from backend.vector_store import (
    SessionRegistry,
    VectorStore,
    cleanup_stale_sessions,
    make_client,
)
from config import settings
from frontend import ui
from utils.helpers import new_session_id

st.set_page_config(page_title=settings.APP_NAME, page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedder():
    """One embedding model for all sessions (loaded on first use)."""
    return Embedder()


@st.cache_resource(show_spinner=False)
def get_vector_backend():
    """One in-memory Chroma client and session registry for the whole server."""
    return make_client(), SessionRegistry()


def init_session():
    state = st.session_state
    if "manager" in state:
        return
    client, registry = get_vector_backend()
    cleanup_stale_sessions(client, registry)                     # drop idle sessions' data
    state.session_id = new_session_id()                          # isolates this visitor's documents
    store = VectorStore(client, state.session_id, get_embedder(), registry)
    state.manager = SourceManager(store)
    state.messages = []
    state.notices = []
    state.uploader_key = 0
    state.video_status = None


init_session()
ui.inject_css()
ui.render_sidebar()
ui.render_chat()
