"""Scenarios 7-10: questions across several PDFs, conflicts, and citations."""
import json

import pytest

from backend import answer_generator, conflict_detector, llm_client, retriever
from config import settings
from tests.conftest import make_pdf


def reply(**kwargs):
    base = {"sufficient": True, "answer": "", "source_findings": [], "conflict": {"status": "none"}}
    base.update(kwargs)
    return json.dumps(base)


@pytest.fixture
def two_ai_pdfs(manager):
    manager.add_file("industry.pdf", make_pdf(
        "Artificial Intelligence (AI) is used in industries to automate processes."))
    manager.add_file("robots.pdf", make_pdf(
        "Artificial Intelligence (AI) provides intelligence to robots."))
    return manager


def test_question_uses_information_from_both_pdfs(two_ai_pdfs):         # scenario 7
    found = retriever.retrieve(two_ai_pdfs.store, "What is Artificial Intelligence?", two_ai_pdfs.ids())
    assert {e.filename for e in found} == {"industry.pdf", "robots.pdf"}


def test_prompt_contains_evidence_from_both_pdfs(two_ai_pdfs, fake_groq):
    fake_groq.reply = reply(answer="AI automates industry [S1] and powers robots [S2].")
    answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?")
    prompt = fake_groq.calls[0][-1]["content"]
    assert "industry.pdf" in prompt and "robots.pdf" in prompt
    assert "automate processes" in prompt and "intelligence to robots" in prompt


def test_comparison_shows_each_pdf_and_combined_answer(two_ai_pdfs, fake_groq):
    found = retriever.retrieve(two_ai_pdfs.store, "What is Artificial Intelligence?", two_ai_pdfs.ids())
    labels = {e.filename: e.label for e in found}
    fake_groq.reply = reply(
        answer=f"AI automates processes [{labels['industry.pdf']}] and gives robots intelligence "
               f"[{labels['robots.pdf']}].",
        source_findings=[
            {"source": labels["industry.pdf"], "finding": "Automates industrial processes."},
            {"source": labels["robots.pdf"], "finding": "Provides intelligence to robots."},
        ],
    )
    answer = answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?")
    assert answer.sufficient and answer.conflict.status == "none"
    assert {f.evidence.filename for f in answer.findings} == {"industry.pdf", "robots.pdf"}
    assert "automates" in answer.text.lower() and "robots" in answer.text.lower()


def test_five_pdfs_are_all_searched(manager):                           # scenario 8
    topics = ["solar", "wind", "hydro", "geothermal", "tidal"]
    for t in topics:
        manager.add_file(f"{t}.pdf", make_pdf(f"Renewable energy from {t} power is clean and renewable."))
    found = retriever.retrieve(manager.store, "renewable energy clean power", manager.ids())
    assert {e.filename for e in found} == {f"{t}.pdf" for t in topics}


def test_answer_in_the_oldest_pdf_is_found_not_only_the_newest(manager):
    manager.add_file("first.pdf", make_pdf("The Zorblax reactor operates at 450 degrees."))
    for n in range(4):
        manager.add_file(f"other{n}.pdf", make_pdf(f"Gardening tips number {n}: water the tomatoes daily."))
    found = retriever.retrieve(manager.store, "At what temperature does the Zorblax reactor operate?",
                               manager.ids())
    assert found and found[0].filename == "first.pdf"


def test_one_document_with_many_matches_does_not_crowd_out_another(manager):
    manager.add_file("big.txt", b"\n\n".join(
        f"Climate report {i}: global temperature rise measured in year {i}.".encode() for i in range(30)))
    manager.add_file("small.txt", b"Climate policy: temperature targets agreed in Paris.")
    found = retriever.retrieve(manager.store, "climate temperature", manager.ids(), top_k=4)
    assert {"big.txt", "small.txt"} <= {e.filename for e in found}


def test_contradiction_is_detected_and_reported(manager, fake_groq):    # scenario 9
    manager.add_file("report_a.pdf", make_pdf("The Eiffel Tower is 300 metres tall."))
    manager.add_file("report_b.pdf", make_pdf("The Eiffel Tower is 450 metres tall."))
    found = retriever.retrieve(manager.store, "How tall is the Eiffel Tower?", manager.ids())
    labels = {e.filename: e.label for e in found}
    fake_groq.reply = reply(
        answer="The sources disagree on the height.",
        source_findings=[{"source": labels["report_a.pdf"], "finding": "300 metres."},
                         {"source": labels["report_b.pdf"], "finding": "450 metres."}],
        conflict={"status": "contradiction", "explanation": "Different heights.",
                  "differences": ["300 m versus 450 m"],
                  "conclusion": "The documents cannot both be right; no source says which is correct."},
    )
    answer = answer_generator.answer_question(manager, "How tall is the Eiffel Tower?")
    assert answer.conflict.status == "contradiction"
    assert answer.conflict.differences == ["300 m versus 450 m"]
    assert "cannot both be right" in answer.conflict.conclusion
    assert {f.evidence.filename for f in answer.findings} == {"report_a.pdf", "report_b.pdf"}
    assert settings.CONFLICT_MESSAGE == "Conflicting information detected in the uploaded documents."


def test_conflict_rules_forbid_picking_a_winner_and_separate_perspectives():
    rules = conflict_detector.SYSTEM_PROMPT
    assert "different_perspectives" in rules and "NOT a conflict" in rules
    assert "Do NOT pick a winner" in rules
    assert "Never use outside knowledge" in rules or "Use only the evidence" in rules


def test_conflict_claim_from_a_single_source_is_ignored(manager, fake_groq):
    manager.add_file("only.pdf", make_pdf("The Eiffel Tower is 300 metres tall."))
    found = retriever.retrieve(manager.store, "How tall is the Eiffel Tower?", manager.ids())
    fake_groq.reply = reply(
        answer="300 m [S1].",
        source_findings=[{"source": found[0].label, "finding": "300 metres."}],
        conflict={"status": "contradiction", "differences": ["x"], "conclusion": "y"},
    )
    answer = answer_generator.answer_question(manager, "How tall is the Eiffel Tower?")
    assert answer.conflict.status == "none"


def test_different_perspectives_are_not_called_conflicts(two_ai_pdfs, fake_groq):
    found = retriever.retrieve(two_ai_pdfs.store, "What is Artificial Intelligence?", two_ai_pdfs.ids())
    fake_groq.reply = reply(
        answer="Both describe AI from different angles.",
        source_findings=[{"source": e.label, "finding": "angle"} for e in found],
        conflict={"status": "different_perspectives", "explanation": "Industry vs robotics.",
                  "conclusion": "Complementary."},
    )
    answer = answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?")
    assert answer.conflict.status == "different_perspectives"


def test_citations_use_real_filename_and_page_number(manager, fake_groq):  # scenario 10
    manager.add_file("guide.pdf", make_pdf("Introduction to gardens.",
                                           "Axolotls regenerate lost limbs and organs completely.",
                                           "Closing remarks."))
    found = retriever.retrieve(manager.store, "Which animal regenerates limbs?", manager.ids())
    assert found[0].filename == "guide.pdf" and found[0].page == 2
    # the 'model' invents a page and a file; the app must ignore both
    fake_groq.reply = reply(
        answer="Axolotls [S1].",
        source_findings=[{"source": "S1", "finding": "Axolotls regenerate. (page 99 of fake.pdf)"},
                         {"source": "S7", "finding": "invented source"}],
    )
    answer = answer_generator.answer_question(manager, "Which animal regenerates limbs?")
    assert [(f.evidence.filename, f.evidence.page) for f in answer.findings] == [("guide.pdf", 2)]


def test_no_evidence_uses_general_knowledge_fallback(manager, fake_groq):
    manager.add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    fake_groq.reply = "LangChain is a framework for building applications powered by language models."
    answer = answer_generator.answer_question(manager, "what is lamgchain?")
    assert answer.sufficient is True and answer.mode == "general"
    assert "General knowledge" not in answer.text            # the label is drawn by the UI, not stored in the text
    assert "LangChain" in answer.text
    assert answer.evidence == []


def test_model_saying_insufficient_uses_general_knowledge_fallback(two_ai_pdfs, fake_groq, monkeypatch):
    replies = iter([reply(sufficient=False, answer="made-up text"), "Artificial intelligence is machines doing smart tasks."])
    monkeypatch.setattr(llm_client, "chat",
                        lambda messages, json_mode=False, temperature=0.1: (fake_groq.calls.append(messages) or next(replies)))
    answer = answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?")
    assert len(fake_groq.calls) == 2                       # first the document attempt, then the labelled fallback
    assert answer.mode == "general" and "smart tasks" in answer.text and answer.evidence == []
    assert "made-up text" not in answer.text


def test_document_only_mode_never_uses_general_knowledge(two_ai_pdfs, fake_groq):
    fake_groq.reply = reply(sufficient=False, answer="made-up text")
    answer = answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?", allow_general=False)
    assert answer.mode == "not_found" and answer.text == settings.NOT_FOUND_MESSAGE
    assert len(fake_groq.calls) == 1                       # no second, general-knowledge call


def test_document_only_mode_with_no_evidence_does_not_call_the_model(manager, fake_groq):
    manager.add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    answer = answer_generator.answer_question(manager, "zzqx qwerty", allow_general=False)
    assert answer.text == settings.NOT_FOUND_MESSAGE and fake_groq.calls == []


@pytest.mark.parametrize("echoed", [
    "General knowledge (not found in your uploaded sources)\n\nParis.",
    "**General knowledge** (not found in your uploaded sources)\n\nGeneral knowledge (not found in your uploaded sources)\nParis.",
    "Paris.",
])
def test_a_copied_general_knowledge_label_is_stripped(manager, fake_groq, echoed):
    manager.add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    fake_groq.reply = echoed
    assert answer_generator.answer_question(manager, "capital of france").text.endswith("Paris.")
    assert "General knowledge" not in answer_generator.answer_question(manager, "capital of france").text


def test_old_labelled_messages_are_cleaned_before_being_sent_as_history(manager, fake_groq):
    manager.add_file("a.pdf", make_pdf("Solar panels convert sunlight."))
    history = [{"role": "user", "content": "capital of japan"},
               {"role": "assistant", "content": "**General knowledge** (not found in your uploaded sources)\n\nTokyo."}]
    answer_generator.answer_question(manager, "capital of france", history)
    assert fake_groq.calls[-1][2]["content"] == "Tokyo."


def test_a_document_that_contradicts_itself_is_flagged(manager, fake_groq):
    """The Andhra Pradesh case: two incompatible statements on the SAME page of ONE document."""
    manager.add_file("ML.pdf", make_pdf(
        "The capital of Andhra Pradesh is Amaravati. The capital of Andhra Pradesh is Hyderabad."))
    fake_groq.reply = reply(
        answer="The document gives two different capitals [S1].",
        source_findings=[{"source": "S1", "finding": "Gives two capitals."}],
        conflict={"status": "contradiction", "explanation": "Two different capitals are stated.",
                  "differences": ["Amaravati versus Hyderabad"],
                  "claims": [{"source": "S1", "quote": "The capital of Andhra Pradesh is Amaravati"},
                             {"source": "S1", "quote": "the capital of andhra pradesh is HYDERABAD."}],
                  "conclusion": "The document contradicts itself; it does not say which is correct."})
    answer = answer_generator.answer_question(manager, "What is the capital of Andhra Pradesh?")
    assert answer.conflict.status == "contradiction"
    assert [c.evidence.filename for c in answer.conflict.claims] == ["ML.pdf", "ML.pdf"]
    assert answer.conflict.claims[0].evidence.page == 1


def test_invented_quotes_cannot_create_a_conflict(manager, fake_groq):
    manager.add_file("ML.pdf", make_pdf("The capital of Andhra Pradesh is Amaravati."))
    fake_groq.reply = reply(
        answer="Conflicting [S1].",
        source_findings=[{"source": "S1", "finding": "x"}],
        conflict={"status": "contradiction", "differences": ["a vs b"], "conclusion": "c",
                  "claims": [{"source": "S1", "quote": "The capital of Andhra Pradesh is Amaravati"},
                             {"source": "S1", "quote": "The capital of Andhra Pradesh is Vizag"}]})   # 2nd quote is made up
    assert answer_generator.answer_question(manager, "capital of andhra pradesh").conflict.status == "none"


def test_a_single_statement_is_never_a_conflict(manager, fake_groq):
    manager.add_file("ML.pdf", make_pdf("The capital of Andhra Pradesh is Amaravati."))
    fake_groq.reply = reply(
        answer="Amaravati [S1].", source_findings=[{"source": "S1", "finding": "Amaravati"}],
        conflict={"status": "contradiction", "differences": ["x"], "conclusion": "y",
                  "claims": [{"source": "S1", "quote": "The capital of Andhra Pradesh is Amaravati"},
                             {"source": "S1", "quote": "The capital of Andhra Pradesh is Amaravati."}]})
    assert answer_generator.answer_question(manager, "capital of andhra pradesh").conflict.status == "none"


def test_non_json_reply_is_kept_without_comparison_claims(two_ai_pdfs, fake_groq):
    fake_groq.reply = "Plain text answer [S1]."
    answer = answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?")
    assert answer.structured is False and answer.text.startswith("Plain text")
    assert answer.conflict.status == "none" and answer.findings == []


def test_json_wrapped_in_code_fences_is_parsed(two_ai_pdfs, fake_groq):
    fake_groq.reply = "```json\n" + reply(answer="Fenced [S1].") + "\n```"
    assert answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?").text == "Fenced [S1]."


def test_history_is_sent_to_the_model(two_ai_pdfs, fake_groq):
    history = [{"role": "user", "content": "Tell me about robots"},
               {"role": "assistant", "content": "Robots use AI."}]
    answer_generator.answer_question(two_ai_pdfs, "What is Artificial Intelligence?", history)
    roles = [m["role"] for m in fake_groq.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]


def test_context_is_limited(manager, fake_groq, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 500)
    manager.add_file("big.txt", ("Alpha particle physics experiment results. " * 400).encode())
    answer_generator.answer_question(manager, "alpha particle physics", top_k=8)
    assert len(fake_groq.calls[0][-1]["content"]) < 2000


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(settings, "get_secret", lambda name, default=None: None)
    with pytest.raises(llm_client.MissingAPIKeyError, match="GROQ_API_KEY"):
        llm_client.chat([{"role": "user", "content": "hi"}])
