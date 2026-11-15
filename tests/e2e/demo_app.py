"""Runs the REAL app.py with a fake embedding model, a fake Groq reply and a fake YouTube transcript, so the
browser flow can be tested without models, an API key or internet.

    streamlit run tests/e2e/demo_app.py --server.port 8766
"""
import json
import os
import re
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GROQ_API_KEY"] = "demo-key"

from backend import llm_client, youtube_processor  # noqa: E402
from tests.conftest import FakeModel  # noqa: E402


class _ST(FakeModel):
    def __init__(self, name):
        pass


sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=_ST)

VIDEO_LINES = [
    ("Welcome to this talk about machine learning.", 0.0, 30.0),
    ("RAG means retrieval augmented generation.", 85.0, 110.0),
    ("It lets a model look up documents before answering.", 110.0, 140.0),
    ("Thanks for watching.", 300.0, 320.0),
]
youtube_processor.fetch_transcript_lines = lambda vid: VIDEO_LINES
youtube_processor.fetch_title = lambda vid: "RAG explained (demo video)"


def fake_chat(messages, json_mode=False, temperature=0.1):
    """Builds a plausible structured answer from the evidence it was given."""
    prompt = messages[-1]["content"]
    if messages[0]["content"].startswith("You are DocuMind AI, a helpful assistant."):
        return "General knowledge (not found in your uploaded sources)\n\nThis is a general-knowledge demo answer."
    question = re.search(r"Question: (.*)", prompt)
    if question and "capital of" in question.group(1).lower():
        # pretend the model noticed the document does not mention this place
        place = question.group(1).strip().rstrip("?").split()[-1].lower()
        if place not in prompt.split("Question:")[0].lower():
            return json.dumps({"sufficient": False, "answer": "", "source_findings": [], "conflict": {"status": "none"}})
    if "Amaravati" in prompt and "Hyderabad" in prompt:
        label = re.search(r"\[(S\d+)\] source", prompt).group(1)
        return json.dumps({"sufficient": True, "answer": f"The document names two capitals [{label}].",
            "source_findings": [{"source": label, "finding": "Names two different capitals."}],
            "conflict": {"status": "contradiction", "explanation": "Two capitals are stated.",
                "differences": ["Amaravati versus Hyderabad"],
                "claims": [{"source": label, "quote": "The capital of Andhra Pradesh is Amaravati"},
                           {"source": label, "quote": "The capital of Andhra Pradesh is Hyderabad"}],
                "conclusion": "The document contradicts itself and does not say which is correct."}})
    blocks = re.findall(r"\[(S\d+)\] source: (.+?) \((.+?)\)\n(.*?)(?=\n\n\[S|\n\nQuestion:)", prompt, re.S)
    findings, seen = [], set()
    for label, name, where, text in blocks:
        if name not in seen:
            seen.add(name)
            findings.append({"source": label, "finding": text.strip().split(".")[0] + "."})
    answer = " ".join(f"{f['finding']} [{f['source']}]" for f in findings)
    numbers = {re.findall(r"\d+", f["finding"])[0] for f in findings if re.findall(r"\d+", f["finding"])}
    conflict = {"status": "none"}
    if len(findings) >= 2 and len(numbers) >= 2:
        conflict = {"status": "contradiction", "explanation": "The sources give different figures.",
                    "differences": [f"{f['source']}: {f['finding']}" for f in findings],
                    "conclusion": "Both cannot be right. The documents do not say which is correct."}
    return json.dumps({"sufficient": True, "answer": answer, "source_findings": findings, "conflict": conflict})


llm_client.chat = fake_chat
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
