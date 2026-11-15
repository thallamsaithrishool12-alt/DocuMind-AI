# 🧠 DocuMind AI – Multi-Source Document & Video Intelligence

Upload up to 5 PDFs plus TXT / MD / DOCX / CSV files, paste YouTube links, and ask questions across **all** of them.
Answers cite the file and page (or a clickable video timestamp), compare what different sources say, and
flag real contradictions instead of silently picking one.

## Features

- **Multi-PDF chat**: 1 to 5 PDFs (never forced to add all five). Each file has a card with an ✕ button.
  Removing one keeps the rest; adding one never replaces the others. The limit is enforced in the **backend**
  as well as shown in the UI: *"Maximum upload limit reached. You can upload only 5 PDF files."*
- **Other formats**: TXT, MD, DOCX (text + tables), CSV (rows keep their column names). Unsupported or broken
  files produce a clear message, never a traceback.
- **Cross-document retrieval**: every source is searched on its own and the best passage from each is merged
  first, so one big document cannot crowd the others out.
- **Comparison and conflict detection**: per-source findings, a combined answer, and a
  *"Conflicting information detected in the uploaded documents."* banner with the differences and what can and
  cannot be concluded. Different perspectives are reported as such, not as contradictions.
- **Honest citations**: filename, page and timestamp always come from stored metadata, never from the LLM.
  If the sources do not contain the answer you get *"I could not find sufficient information in your uploaded
  sources to answer this question."* and the LLM is not even called.
- **YouTube**: validates the link, fetches captions with timestamps, indexes ~30-second windows, and shows
  clickable timestamps (e.g. `▶ 01:25`) that open the video at that moment.
- **Audio/video files without captions**: upload the file and it is transcribed with Groq speech-to-text.
- **Per-session privacy**: each visitor gets their own in-memory vector collection. Nothing is written to disk
  and idle sessions are deleted.

## How it works

```
PDF/TXT/MD/DOCX/CSV ─► text + page ─┐
YouTube captions / audio ─► text + time ─┤► chunks ─► embeddings ─► ChromaDB (one collection per session)
                                    │
Question ─► embed ─► search EACH source ─► merge (best of every source first) ─► evidence S1..Sn
                                    │      nothing close enough ─► "could not find sufficient information"
                                    ▼
          Groq LLM returns JSON: answer + per-source findings + conflict check
                                    ▼
   app validates it (unknown labels dropped, one source cannot "conflict") ─► answer, comparison, citations
```

| Part | Tool |
|---|---|
| UI | Streamlit |
| LLM and speech-to-text | Groq API (`llama-3.3-70b-versatile`, `whisper-large-v3-turbo`) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (in-memory) |
| PDF / DOCX | pypdf / python-docx |
| Captions | youtube-transcript-api |

## Project structure

```
DocuMind-AI/
├── app.py                      # entry point
├── config/settings.py          # limits, models, retrieval settings, messages
├── frontend/                   # ui.py, components/ (file cards, answer view), styles/main.css
├── backend/
│   ├── document_processor.py   # routes a file to the right extractor
│   ├── pdf_processor.py  file_processor.py  youtube_processor.py
│   ├── chunking.py  embeddings.py  vector_store.py  retriever.py
│   ├── answer_generator.py  conflict_detector.py  llm_client.py
│   └── source_manager.py       # per-session source list, limits, duplicates, removal
├── utils/                      # validators, helpers, error_handler
├── tests/                      # pytest suite + e2e/ (browser check)
├── .streamlit/config.toml      # theme
└── requirements.txt  .env.example  .gitignore
```

## Run locally

Needs Python 3.10+ and a free Groq API key (https://console.groq.com → API Keys).

```bash
python -m venv venv
venv\Scripts\activate              # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# provide the key (pick one)
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then edit it
#   or:  set GROQ_API_KEY=your-key        (Mac/Linux: export GROQ_API_KEY=your-key)

streamlit run app.py
```

The first upload is slow because the embedding model (about 90 MB) is downloaded once.
Never commit `.streamlit/secrets.toml` or `.env`; both are in `.gitignore`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (see below).
2. Go to https://share.streamlit.io → **Create app** → choose the repo, branch `main`, file `app.py`.
3. **Advanced settings → Secrets**, paste: `GROQ_API_KEY = "your-key"`
4. Deploy. Ollama is not used, so nothing local is needed.

Streamlit's free tier has about 1 GB RAM; the embedding model fits but long videos or very large PDFs may be slow.
If the build reports an old SQLite version for Chroma, add `pysqlite3-binary` to `requirements.txt` and put
`__import__("pysqlite3"); import sys; sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")` at the very top of `app.py`.

## Upload to GitHub

```bash
git init
git add .
git commit -m "DocuMind AI"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

To update an existing repo, clone it, copy these files over the old ones, then `git add .`, `git commit`, `git push`.
Check `git status` first: no `secrets.toml`, `.env` or uploaded documents should be listed.

## Answer modes and settings

- **Hybrid (default):** answers from your sources and cites them. If they do not contain the answer, you get a
  short answer from the model's general knowledge with a clear *"General knowledge · not found in your uploaded
  sources"* label (shown once, never with fake citations).
- **Document only:** strict RAG. If the sources do not support an answer you get: *"I could not find sufficient
  information in your uploaded sources to answer this question."* and the model is not asked at all.
- **Passages per answer (2 to 10, default 6):** a slider in the sidebar. The chosen number is passed to retrieval
  (`retrieve(..., top_k=...)`) for every new question; `TOP_K = 6` in `config/settings.py` is only the starting value.
  You get up to that many passages. Fewer are used when fewer are relevant enough (`MAX_DISTANCE`).
- **Contradictions inside one document** are detected too (for example a page that says two different capitals).
  A conflict is only reported when the conflicting statements were quoted word for word from the retrieved text.

## Using it

1. Add files in the sidebar (several at once is fine) and/or paste a YouTube link.
2. Ask a question. Open **Source comparison** and **Sources** under an answer to see exactly where it came from.
3. Click ✕ on a card to remove that source. Its chunks are deleted from the index and cannot appear in later answers.

Settings you may want to change are in `config/settings.py` (`MAX_PDFS`, `TOP_K`, `MAX_DISTANCE`, `GROQ_MODEL`, chunk sizes).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

123 tests cover uploads (1, 3, 5 and a refused sixth PDF), removal and replacement, multi-document retrieval,
conflict handling, citations, TXT/MD/DOCX/CSV, YouTube links, timestamps, error cases, session isolation and the
Streamlit UI. They use a fake embedding model and a fake Groq reply, so they run in seconds with no key or internet.

Browser check with real file uploads (optional; needs `playwright install chromium`):

```bash
streamlit run tests/e2e/demo_app.py --server.port 8766 --server.headless true
python tests/e2e/browser_flow.py http://localhost:8766
```

## Limitations (please read)

- **Not tested against the real services.** The automated tests mock Groq, YouTube and the embedding model.
  Conflict detection quality in particular depends on the real LLM; test it with your own documents.
- **YouTube from cloud servers**: YouTube often blocks caption requests from hosting providers. If that happens
  the app says so and indexes nothing; upload the audio/video file instead.
- **Transcript-only video search.** It searches what is *said*. It does not see slides or on-screen content
  (visual understanding is not implemented). Timestamps jump to the start of the matching ~30-second section.
- **Scanned PDFs (images)** are rejected: there is no OCR.
- Audio/video transcription is limited by Groq's file-size cap (about 25 MB) and its free-tier rate limits.
- Text sent to answer a question (the question and the matching passages) goes to Groq.
- Follow-up questions use the chat history for the answer, but retrieval searches with the current question only.

## Troubleshooting

| Message | Fix |
|---|---|
| "No Groq API key found" | Add `GROQ_API_KEY` to secrets or the environment |
| "rate limit was reached" | Wait a minute (Groq free tier) |
| "No readable text found" | The PDF is a scan; use a text PDF |
| "YouTube blocked the request" | Try later, or upload the audio file |
