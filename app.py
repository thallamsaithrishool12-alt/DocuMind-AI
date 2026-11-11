from pathlib import Path

import streamlit as st

import rag_core


# ---------- Page setup ----------

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

AVATARS = {"user": "👨‍💻", "assistant": "🤖"}
SAMPLE_PDF = Path(__file__).resolve().parent / "sample" / "document.pdf"
NOT_FOUND = (
    "I could not find relevant information "
    "in the uploaded PDF to answer your question."
)


# ---------- Styling and animations ----------

CSS = """
@media (prefers-reduced-motion: no-preference) {
  @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes floaty {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
  }
  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
  }
  .hero { animation: gradientShift 12s ease infinite; }
  .feature-card { animation: fadeUp 0.6s ease both; }
  .feature-card:nth-child(2) { animation-delay: 0.12s; }
  .feature-card:nth-child(3) { animation-delay: 0.24s; }
  .feature-icon { animation: floaty 3s ease-in-out infinite; }
  [data-testid="stChatMessage"] { animation: fadeUp 0.35s ease both; }
  .typing span { animation: bounce 1.2s infinite ease-in-out; }
  .typing span:nth-child(2) { animation-delay: 0.15s; }
  .typing span:nth-child(3) { animation-delay: 0.3s; }
  .robot-loader { animation: floaty 1.4s ease-in-out infinite; }
  .stButton > button { transition: transform 0.15s ease, border-color 0.15s ease; }
  .stButton > button:hover { transform: translateY(-1px); }
  section[data-testid="stSidebar"] .stButton > button:hover { transform: translateX(3px); }
}

.hero {
  background: linear-gradient(120deg, #5b3df5, #8b5cf6, #06b6d4, #5b3df5);
  background-size: 300% 300%;
  border-radius: 16px;
  padding: 26px 28px;
  color: #fff;
  margin-bottom: 18px;
}
.hero-title { font-size: 30px; font-weight: 700; line-height: 1.2; }
.hero-sub { opacity: 0.92; margin-top: 6px; }

.features { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 14px; }
.feature-card {
  flex: 1 1 160px;
  border: 1px solid rgba(127, 127, 127, 0.25);
  background: rgba(127, 127, 127, 0.08);
  border-radius: 14px;
  padding: 16px;
}
.feature-icon { font-size: 26px; display: inline-block; }
.feature-title { font-weight: 600; margin-top: 6px; }
.feature-text { opacity: 0.75; font-size: 14px; }

.generating { display: flex; align-items: center; gap: 12px; padding: 4px 0; }
.robot-loader { font-size: 30px; display: inline-block; }
.generating strong { font-size: 15px; }
.typing { margin-top: 8px; }
.typing span {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 5px;
  border-radius: 50%;
  background: #8b7bff;
}

.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  margin: 6px 0 2px;
}
.badge.strong { background: rgba(34, 197, 94, 0.18); color: #22c55e; }
.badge.good { background: rgba(234, 179, 8, 0.18); color: #eab308; }
.badge.weak { background: rgba(249, 115, 22, 0.18); color: #f97316; }

section[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  border-radius: 999px;
  text-align: left;
}
"""

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

TYPING_HTML = (
    '<div class="typing"><span></span><span></span><span></span></div>'
)

HERO_HTML = (
    '<div class="hero">'
    '<div class="hero-title">📄 Chat with your PDF</div>'
    '<div class="hero-sub">{subtitle}</div>'
    "</div>"
)

FEATURES_HTML = (
    '<div class="features">'
    '<div class="feature-card"><div class="feature-icon">📤</div>'
    '<div class="feature-title">1. Upload</div>'
    '<div class="feature-text">Add a PDF in the sidebar, or try the sample.</div></div>'
    '<div class="feature-card"><div class="feature-icon">💬</div>'
    '<div class="feature-title">2. Ask</div>'
    '<div class="feature-text">Ask anything. Answers use only your document.</div></div>'
    '<div class="feature-card"><div class="feature-icon">🔎</div>'
    '<div class="feature-title">3. Verify</div>'
    '<div class="feature-text">Every answer shows the pages it came from.</div></div>'
    "</div>"
)


# ---------- Load models and database ----------

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(rag_core.EMBED_MODEL)


@st.cache_resource
def get_client():
    import chromadb
    return chromadb.PersistentClient(path="chroma_db")


# ---------- Helpers ----------

def show_sources(sources):
    """Show which pages and text the answer was based on."""
    pages = sorted({s["page"] for s in sources})
    label = "Sources: page " if len(pages) == 1 else "Sources: pages "

    with st.expander(label + ", ".join(str(p) for p in pages)):
        for source in sources:
            similarity = 1 - source["distance"]
            st.markdown(
                f"**Page {source['page']}** · similarity {similarity:.2f}")
            st.write(source["text"])
            st.divider()


def show_extras(sources):
    """Match-strength badge and sources under an answer."""
    if not sources:
        return

    label, css_class = rag_core.match_strength(
        min(s["distance"] for s in sources))
    st.markdown(
        f'<span class="badge {css_class}" '
        'title="How close the best matching text is to your question. '
        'It does not guarantee the answer is correct.">'
        f"● {label} match</span>",
        unsafe_allow_html=True,
    )
    show_sources(sources)


def chat_as_text():
    lines = [f"Chat about: {st.session_state.file_name}", ""]
    for message in st.session_state.messages:
        who = "You" if message["role"] == "user" else "Assistant"
        lines.append(f"{who}: {message['content']}")
        if message.get("sources"):
            pages = sorted({s["page"] for s in message["sources"]})
            lines.append("  (sources: pages " + ", ".join(str(p)
                         for p in pages) + ")")
        lines.append("")
    return "\n".join(lines)


def reset_pdf():
    st.session_state.pending_question = None
    st.session_state.generating = False
    st.session_state.collection = None
    st.session_state.file_id = None
    st.session_state.file_name = None
    st.session_state.topics = []
    st.session_state.messages = []
    st.session_state.pages = 0
    st.session_state.chunks = 0
    # A new key gives an empty uploader, so the removed file is not loaded again
    st.session_state.uploader_key += 1


def load_pdf(data, name):
    """Index a PDF and store it in the session. Returns True on success."""
    with st.spinner("Reading your PDF..."):
        try:
            collection = rag_core.build_index(
                data, get_client(), load_embedder())
        except ValueError as e:
            st.sidebar.error(str(e))
            return False
        except Exception as e:
            st.sidebar.error(f"PDF processing failed: {e}")
            return False

        st.session_state.collection = collection
        st.session_state.file_id = rag_core.file_id(data)
        st.session_state.file_name = name
        st.session_state.topics = rag_core.find_topics(collection)
        st.session_state.pages = len(rag_core.read_pdf(data))
        st.session_state.chunks = collection.count()
        st.session_state.messages = []
    return True


# ---------- Session State ----------

defaults = {
    "messages": [],
    "collection": None,
    "file_id": None,
    "file_name": None,
    "topics": [],
    "pages": 0,
    "chunks": 0,
    "uploader_key": 0,
    "pending_question": None,
    "generating": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- Sidebar ----------

with st.sidebar:

    st.markdown("### 📄 PDF Chat")

    uploaded = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded:
        pdf_data = uploaded.getvalue()
        if rag_core.file_id(pdf_data) != st.session_state.file_id:
            load_pdf(pdf_data, uploaded.name)

    if st.session_state.collection is None and SAMPLE_PDF.exists():
        if st.button("✨ Try the sample PDF", key="sample"):
            load_pdf(SAMPLE_PDF.read_bytes(), "document.pdf")

    if st.session_state.collection is not None:

        st.success(f"Loaded: {st.session_state.file_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", st.session_state.pages)
        col2.metric("Chunks", st.session_state.chunks)
        col3.metric("Topics", len(st.session_state.topics))

        if st.session_state.topics:
            st.markdown("**Main topics** (click to ask)")
            for i, (keywords, size) in enumerate(st.session_state.topics):
                if st.button(
                    f"{keywords} ({size})",
                    key=f"topic_{i}",
                    disabled=st.session_state.generating,
                ):
                    topic_question = f"Tell me about {keywords}"
                    st.session_state.pending_question = topic_question
                    st.session_state.generating = True
                    st.session_state.messages.append(
                        {"role": "user", "content": topic_question}
                    )

        with st.expander("⚙️ Settings"):
            st.slider(
                "Chunks to read (TOP_K)", 1, 6, rag_core.TOP_K, key="top_k",
                help="How many pieces of the PDF the AI reads for each answer.",
            )
            st.slider(
                "Match strictness (MAX_DISTANCE)", 0.40, 0.90, rag_core.MAX_DISTANCE,
                step=0.05, key="max_distance",
                help="Lower = stricter. Questions whose best match is farther "
                     "than this are answered with 'not found'.",
            )

        if st.button("🧹 Clear chat", key="clear"):
            st.session_state.messages = []

        if st.button("🗑️ Remove PDF / Upload another", key="remove"):
            reset_pdf()
            st.rerun()

        st.download_button(
            "⬇️ Download chat",
            data=chat_as_text(),
            file_name="chat.txt",
            mime="text/plain",
            key="download",
        )

    with st.expander("How it works"):
        st.markdown(
            "1. The PDF is split into chunks\n"
            "2. Each chunk becomes numbers (an embedding)\n"
            "3. Your question is matched to the closest chunks\n"
            "4. If nothing is close enough, you get 'not found'\n"
            "5. Otherwise a cloud LLM answers from those chunks only"
        )


# ---------- Header ----------

if st.session_state.collection is None:
    subtitle = "Upload a document and ask questions. I'll answer using your PDF."
else:
    subtitle = f"Chatting with <b>{st.session_state.file_name}</b>"

st.markdown(HERO_HTML.format(subtitle=subtitle), unsafe_allow_html=True)


# ---------- Empty state ----------

if st.session_state.collection is None:
    st.markdown(FEATURES_HTML, unsafe_allow_html=True)
    st.caption(
        "Powered by Groq cloud AI. Answers are generated using retrieved PDF context.")


# ---------- Chat ----------

if st.session_state.collection is not None:

    # Display the conversation so far.
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            st.markdown(message["content"])
            show_extras(message.get("sources"))

    # Disable the composer while a question is being processed.
    typed = st.chat_input(
        "AI is generating an answer..." if st.session_state.generating
        else "Ask anything about your PDF...",
        disabled=st.session_state.generating,
    )

    # A submitted question is stored first, then the app reruns. On the next
    # run the input is disabled while the pending question is processed.
    if typed and not st.session_state.generating:
        st.session_state.pending_question = typed
        st.session_state.generating = True
        st.session_state.messages.append({"role": "user", "content": typed})
        st.rerun()

    # Topic suggestions also arrive here, but do not need another question
    # submission from the user.
    question = st.session_state.pending_question
    if question:
        st.session_state.pending_question = None

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            placeholder = st.empty()
            placeholder.markdown(
                '<div class="generating">'
                '<span class="robot-loader">🤖</span>'
                '<div><strong>Generating answer...</strong>'
                '<div class="typing"><span></span><span></span><span></span></div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            sources = []
            answer = ""

            try:
                hits = rag_core.retrieve_scored(
                    st.session_state.collection,
                    load_embedder(),
                    question,
                    top_k=st.session_state.get("top_k", rag_core.TOP_K),
                    max_distance=st.session_state.get(
                        "max_distance", rag_core.MAX_DISTANCE
                    ),
                )

                if not hits:
                    answer = NOT_FOUND
                    placeholder.markdown(answer)
                else:
                    for piece in rag_core.ask_llm_stream(
                        question, [h["text"] for h in hits]
                    ):
                        answer += piece
                        placeholder.markdown(answer + "▌")

                    placeholder.markdown(answer)
                    sources = hits
                    show_extras(sources)

            except Exception as e:
                answer = (
                    "Something went wrong while generating the answer. "
                    "Please check your Groq API key, model access, and internet connection."
                )
                placeholder.markdown(answer)
                st.caption(str(e))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.session_state.generating = False

        # Rerun once more to re-enable the chat input after the answer is saved.
        st.rerun()

