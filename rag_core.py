"""Core RAG logic: PDF reading, chunking, vector index, retrieval, LLM answer, topics.

This file has no Streamlit code, so it can be tested and reused by evaluation/.
"""
import hashlib
import io

import numpy as np
from pypdf import PdfReader
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Configuration ----------

LLM = "llama3.2:1b"
EMBED_MODEL = "all-MiniLM-L6-v2"
# Values chosen from the Day 13 evaluation (see README): 800 / 80 / 3 / 0.70
CHUNK_SIZE = 800
OVERLAP = 80
TOP_K = 3
MAX_DISTANCE = 0.70   # cosine distance; best match worse than this = "not in the PDF"
NUM_TOPICS = 4

SYSTEM_PROMPT = (
    "You are a helpful PDF assistant.\n\n"
    "Follow these rules:\n"
    "1. Answer only using the provided PDF context.\n"
    "2. Never use outside knowledge.\n"
    "3. Never invent facts or information.\n"
    "4. If the answer is not available in the context, "
    "say: I could not find this information in the PDF.\n"
    "5. Give clear and simple answers.\n"
    "6. Answer according to the user's question."
)


# ---------- PDF processing ----------

def file_id(data):
    """Same file -> same id, so a PDF is only indexed once."""
    return hashlib.md5(data).hexdigest()[:12]


def read_pdf(data):
    """Return [(page_number, text)] with page numbers starting at 1."""
    reader = PdfReader(io.BytesIO(data))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


# ---------- Create the PDF index ----------

def build_index(data, client, embedder):
    """Chunk the PDF, embed the chunks and store them in Chroma (once per PDF)."""
    collection = client.get_or_create_collection(
        name=f"pdf_{file_id(data)}",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        return collection

    chunks = []
    metadatas = []
    for page_num, page_text in read_pdf(data):
        for piece in chunk_text(page_text):
            chunks.append(piece)
            metadatas.append({"page": page_num})

    if not chunks:
        raise ValueError("No readable text found. This may be a scanned PDF.")

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embedder.encode(chunks).tolist(),
        metadatas=metadatas,
    )
    return collection


# ---------- Retrieve relevant information ----------

def retrieve_scored(collection, embedder, question, top_k=None, max_distance=None):
    """Return the closest chunks as [{"text", "page", "distance"}], best first.

    Returns an empty list when even the best match is too far away,
    which means the answer is probably not in the PDF.
    """
    top_k = TOP_K if top_k is None else top_k
    max_distance = MAX_DISTANCE if max_distance is None else max_distance

    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[embedder.encode(question).tolist()],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    hits = [
        {"text": doc, "page": meta["page"], "distance": float(dist)}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]

    if not hits or min(h["distance"] for h in hits) > max_distance:
        return []
    return hits


def retrieve(collection, embedder, question, top_k=None, max_distance=None):
    """Same as retrieve_scored but returns (chunks, metadatas). Used by evaluation/."""
    hits = retrieve_scored(collection, embedder, question, top_k, max_distance)
    return [h["text"] for h in hits], [{"page": h["page"]} for h in hits]


def match_strength(distance):
    """Turn a distance into a label for the interface: (text, css_class).

    This says how close the best matching text is to the question.
    It does not guarantee the answer is correct.
    """
    if distance <= 0.45:
        return "Strong", "strong"
    if distance <= 0.60:
        return "Good", "good"
    return "Weak", "weak"


# ---------- Generate the answer ----------

def _build_messages(question, chunks):
    context = "\n\n---\n\n".join(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PDF Context:\n{context}\n\nQuestion:\n{question}"},
    ]


def ask_llm(question, chunks):
    import ollama

    response = ollama.chat(model=LLM, messages=_build_messages(question, chunks))
    return response["message"]["content"]


def ask_llm_stream(question, chunks):
    """Same as ask_llm, but yields the answer piece by piece as the model writes it."""
    import ollama

    stream = ollama.chat(model=LLM, messages=_build_messages(question, chunks), stream=True)
    for part in stream:
        piece = part["message"]["content"]
        if piece:
            yield piece


# ---------- ML: main topics of the PDF ----------

def find_topics(collection, k=NUM_TOPICS):
    """KMeans groups chunks by meaning; TF-IDF names each group with keywords.

    Returns [(keywords, chunk_count)], biggest topic first.
    """
    data = collection.get(include=["documents", "embeddings"])
    texts = data["documents"]
    if len(texts) == 0:
        return []

    embeddings = np.array(data["embeddings"])
    k = min(k, len(texts))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)

    tfidf = TfidfVectorizer(stop_words="english")
    try:
        matrix = tfidf.fit_transform(texts)
    except ValueError:  # the text has only stop words
        return []
    terms = np.array(tfidf.get_feature_names_out())

    topics = []
    for cluster in range(k):
        idx = np.where(labels == cluster)[0]
        if len(idx) == 0:
            continue
        scores = np.asarray(matrix[idx].mean(axis=0)).ravel()
        words = terms[scores.argsort()[::-1][:4]]
        topics.append((", ".join(words), int(len(idx))))

    return sorted(topics, key=lambda t: t[1], reverse=True)
