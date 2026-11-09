import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

import evaluate  # noqa: E402

import rag_core  # noqa: E402


def test_questions_file_is_valid():
    path = Path(__file__).resolve().parent.parent / "evaluation" / "questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    assert len(questions) >= 10
    assert all("question" in q and "page" in q for q in questions)
    assert any(q["page"] is None for q in questions)
    assert all(q["page"] is None or 1 <= q["page"] <= 5 for q in questions)


def test_evaluate_counts_hits_and_refusals(sample_pdf, client, embedder):
    collection = rag_core.build_index(sample_pdf, client, embedder)
    page2_text = rag_core.chunk_text(rag_core.read_pdf(sample_pdf)[1][1])[0]
    questions = [
        {"question": page2_text, "page": 2},    # close match: should be a hit
        {"question": "zzqx", "page": None},     # not in the PDF: should be refused
        {"question": "zzqx", "page": 3},        # nothing retrieved: should be a miss
    ]
    result = evaluate.evaluate(questions, collection, embedder, max_distance=0.65)

    assert result["answerable"] == 2
    assert result["unanswerable"] == 1
    assert result["hits"] == 1
    assert result["refused"] == 1
    assert [row["ok"] for row in result["rows"]] == [True, True, False]
