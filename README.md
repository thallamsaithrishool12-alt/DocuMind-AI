# 📄 Chat with your PDF

A RAG (Retrieval-Augmented Generation) app. Upload a PDF, see its main topics, and ask questions in a
chat. Answers come only from your document, with the source pages shown, and the app says so when the
answer isn't in the PDF. Everything runs **locally and free**: no API keys, and your documents never leave your computer.

## Features

- **Animated dark UI**: gradient hero, fade-in messages, typing dots, answers streaming word by word (animations switch off for users who prefer reduced motion)
- **Match-strength badge** (Strong / Good / Weak) on every answer, plus a Settings panel to tune Top-K and strictness
- **Quick actions**: try-the-sample button, clickable topics, clear chat, remove PDF, download the chat
- **Chat interface** with history (Streamlit chat UI)
- **Answers grounded in your PDF**, with source pages and the exact text used
- **"Not in the PDF" detection**: if no chunk is close enough to the question, the LLM is never called, so it can't make things up
- **Main topics** of the document found with machine learning (KMeans + TF-IDF)
- **Indexes each PDF once**: re-uploading the same file is instant
- **Accuracy check**: a script that measures retrieval accuracy on a set of test questions
- **Automated tests and CI** (GitHub Actions)

## How it works

```
PDF ──► split into chunks ──► embeddings ──► Chroma vector DB
                                                   │
Question ──► embedding ──► find closest chunks ────┘
                                   │
                    too far? ──► "not found in the PDF"
                                   │
                          closest chunks + question
                                   ▼
                       LLM (Ollama, llama3.2:1b)
                                   ▼
                        answer + source pages
```

| Concept | Tool used |
|---|---|
| LLM (generative AI) | Ollama with `llama3.2:1b` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database | Chroma |
| RAG | `rag_core.py` (retrieve, then generate) |
| Machine learning | scikit-learn: KMeans clusters chunks, TF-IDF names each topic |
| Web app | Streamlit |

## Project structure

```
pdf-chat-rag/
├── app.py                  # Streamlit chat website (animated UI)
├── .streamlit/config.toml  # dark theme
├── rag_core.py             # RAG logic: PDF, chunks, index, retrieval, LLM, topics
├── requirements.txt        # packages to run the app
├── requirements-dev.txt    # extra packages for the tests
├── sample/document.pdf     # sample PDF to try the app
├── evaluation/
│   ├── questions.json      # 19 test questions with the expected page
│   ├── holdout.json        # 7 unseen questions to check the tuning isn't overfitted
│   └── evaluate.py         # measures retrieval accuracy
├── tests/                  # automated tests
└── .github/workflows/ci.yml
```

## Run it

You need Python 3.10+ and [Ollama](https://ollama.com).

```bash
# 1. Get the LLM (one time, about 1.3 GB)
ollama pull llama3.2:1b

# 2. Install packages (use a virtual environment)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt

# 3. Start the app (Ollama must be running)
streamlit run app.py
```

Open the link shown in the terminal (usually http://localhost:8501) and upload `sample/document.pdf`.
The first upload is slow because the embedding model (about 90 MB) is downloaded.

Try asking: *What is KMeans?* · *Why do LLMs hallucinate?* · *What are the two phases of RAG?*
Then try one that is not in the PDF: *Who won the 2011 cricket World Cup?*

## Measure accuracy

```bash
python evaluation/evaluate.py
```

It checks, for each question in `evaluation/questions.json`, whether the right page is among the
retrieved chunks (hit rate), and whether questions that are not in the PDF are correctly refused.
This tests the retrieval step, which is where most wrong answers come from. It does not need Ollama.

Run it after changing `CHUNK_SIZE`, `OVERLAP`, `TOP_K`, `MAX_DISTANCE` or the embedding model in
`rag_core.py`, to see whether accuracy went up or down.

**My results** (measured on my machine with `sample/document.pdf`):

| Setting | Questions | Hit rate | Correctly refused |
|---|---|---|---|
| chunk 500, top_k 3, max_distance 0.65 (baseline) | first test set | 8/11 | 3/3 |
| chunk 800, overlap 80, top_k 3, max_distance 0.70 (tuned) | 19 | 14/14 | 5/5 |
| same settings on unseen hold-out (`holdout.json`) | 7 | 3/4 | 2/3 |
| combined | 26 | 17/18 | 7/8 |

Overall 24 of 26 questions handled correctly (92%). The tuned row is optimistic because the settings were
chosen on those same questions, so the hold-out row is the fairer number. This is a small test on one
document, not a benchmark against other systems.

Hold-out check: `python evaluation/evaluate.py sample/document.pdf evaluation/holdout.json`

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use a small fake embedder, so they run in seconds without downloading models or running Ollama.

## Settings (top of `rag_core.py`)

| Setting | Default | Meaning |
|---|---|---|
| `LLM` | `llama3.2:1b` | Ollama model. Use `llama3.2` for better answers if your computer can handle it |
| `CHUNK_SIZE` / `OVERLAP` | 800 / 80 | Size of each text chunk and how much neighbours share |
| `TOP_K` | 3 | Chunks given to the LLM |
| `MAX_DISTANCE` | 0.70 | Best match must be closer than this, otherwise "not in the PDF" |
| `NUM_TOPICS` | 4 | Number of topic clusters |

## Troubleshooting

- **"Something went wrong... is Ollama running?"**: start Ollama and run `ollama pull llama3.2:1b`.
- **"No readable text found"**: the PDF is a scan (images). Use a PDF with selectable text.
- **Slow answers**: normal for a local model on a laptop. Close other programs.
- **A `torch.classes` warning in the terminal**: harmless.
- **Changed chunk settings but results look the same**: delete the `chroma_db` folder so the PDF is re-indexed.

## Limitations and ideas

- Short definition questions can be wrongly refused, and questions on related topics (e.g. Kubernetes) can slip past the distance check; the system prompt is the second defense.
- The small 1B model can still misread tricky questions. Retrieval is the bigger factor for accuracy.
- Only text PDFs are supported (no OCR for scanned pages).
- Ideas: hybrid keyword + vector search, re-ranking, sentence-aware chunking, multiple PDFs, OCR.
