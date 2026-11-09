"""Measure how accurate the retrieval step is, using evaluation/questions.json.

Run from the project folder:
    python evaluation/evaluate.py
    python evaluation/evaluate.py sample/document.pdf evaluation/questions.json

For each question with a known page it checks whether that page is among the
retrieved chunks (hit rate). For questions that are NOT in the PDF
("page": null) it checks that the app correctly finds nothing relevant.

It tests retrieval only, so it does not need Ollama.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_core  # noqa: E402


def evaluate(questions, collection, embedder, top_k=rag_core.TOP_K,
             max_distance=rag_core.MAX_DISTANCE):
    rows = []
    hits = refused = 0
    answerable = unanswerable = 0

    for item in questions:
        chunks, metas = rag_core.retrieve(
            collection, embedder, item["question"], top_k, max_distance
        )
        pages = sorted({m["page"] for m in metas})

        if item["page"] is None:
            unanswerable += 1
            ok = len(chunks) == 0
            refused += ok
        else:
            answerable += 1
            ok = item["page"] in pages
            hits += ok

        rows.append({"question": item["question"], "expected": item["page"],
                     "retrieved": pages, "ok": ok})

    return {
        "rows": rows,
        "answerable": answerable,
        "hits": hits,
        "unanswerable": unanswerable,
        "refused": refused,
    }


def main():
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "sample/document.pdf")
    questions_path = Path(sys.argv[2] if len(sys.argv) > 2 else "evaluation/questions.json")

    import chromadb
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(rag_core.EMBED_MODEL)
    collection = rag_core.build_index(pdf_path.read_bytes(), chromadb.EphemeralClient(), embedder)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    result = evaluate(questions, collection, embedder)

    print(f"\nSettings: chunk={rag_core.CHUNK_SIZE}, overlap={rag_core.OVERLAP}, "
          f"top_k={rag_core.TOP_K}, max_distance={rag_core.MAX_DISTANCE}\n")
    for row in result["rows"]:
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"[{mark}] {row['question']}  (expected page {row['expected']}, got {row['retrieved']})")

    print(f"\nRetrieval hit rate : {result['hits']}/{result['answerable']}")
    print(f"Correctly refused  : {result['refused']}/{result['unanswerable']} "
          f"(questions that are not in the PDF)")


if __name__ == "__main__":
    main()
