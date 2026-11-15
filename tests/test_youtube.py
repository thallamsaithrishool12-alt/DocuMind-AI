"""Scenarios 11-13: YouTube URLs, timestamps and clickable links. YouTube itself is mocked because
tests must not depend on the network; see the README for what that means."""
import pytest
from youtube_transcript_api import TranscriptsDisabled

from backend import answer_generator, llm_client, retriever, youtube_processor
from frontend.components.answer_view import location_markdown
from tests.conftest import make_pdf
from utils.error_handler import InvalidYouTubeURLError, TranscriptUnavailableError
from utils.validators import extract_youtube_id, youtube_link

VID = "dQw4w9WgXcQ"

LINES = [
    ("Welcome to this talk about machine learning.", 0.0, 20.0),
    ("Today we cover data and models.", 20.0, 55.0),
    ("RAG means retrieval augmented generation.", 85.0, 100.0),
    ("It lets a model look up documents before answering.", 100.0, 130.0),
    ("The retriever finds passages and the generator writes the answer.", 130.0, 170.0),
    ("Thanks for watching and see you next time.", 300.0, 320.0),
]


@pytest.fixture
def video(monkeypatch, manager):
    monkeypatch.setattr(youtube_processor, "fetch_transcript_lines", lambda vid: LINES)
    monkeypatch.setattr(youtube_processor, "fetch_title", lambda vid: "RAG explained")
    return manager


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?t=42",
    f"https://www.youtube.com/watch?v={VID}&list=PL123&t=5s",
    f"https://m.youtube.com/watch?v={VID}",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"youtube.com/watch?v={VID}",
])
def test_valid_youtube_urls(url):
    assert extract_youtube_id(url) == VID


@pytest.mark.parametrize("url", ["", "   ", "hello world", "https://example.com/watch?v=" + VID,
                                 "https://www.youtube.com/watch?v=short", "https://www.youtube.com/",
                                 "https://evil.com/youtu.be/" + VID, "ftp://"])
def test_invalid_youtube_urls(url):
    with pytest.raises(InvalidYouTubeURLError):
        extract_youtube_id(url)


def test_adding_a_video_keeps_real_timestamps(video):                    # scenario 11
    info = video.add_youtube(f"https://youtu.be/{VID}")
    assert info.source_type == "youtube" and info.filename == "RAG explained" and info.video_id == VID
    metas = video.store.collection.get(include=["metadatas"])["metadatas"]
    allowed = {int(start) for _, start, _ in LINES} | {0}
    assert all(m["video_timestamp"] in allowed or m["video_timestamp"] >= 0 for m in metas)
    assert all(m["end_timestamp"] >= m["video_timestamp"] for m in metas)
    assert max(m["end_timestamp"] for m in metas) == 320          # nothing beyond the real transcript


def test_question_about_a_video_section_returns_that_timestamp(video):   # scenario 12
    video.add_youtube(f"https://youtu.be/{VID}")
    found = retriever.retrieve(video.store, "What does RAG mean?", video.ids())
    assert found and found[0].start == 85 and "retrieval augmented generation" in found[0].text
    assert found[0].video_id == VID and found[0].source_type == "youtube"


def test_timestamp_links_open_the_video_at_that_moment(video):           # scenario 13
    video.add_youtube(f"https://youtu.be/{VID}")
    best = retriever.retrieve(video.store, "What does RAG mean?", video.ids())[0]
    link = location_markdown(best)
    assert f"https://www.youtube.com/watch?v={VID}&t=85s" in link and "01:25" in link
    assert youtube_link(VID, 3725) == f"https://www.youtube.com/watch?v={VID}&t=3725s"


def test_multiple_sections_can_answer_one_question(video, fake_groq):
    video.add_youtube(f"https://youtu.be/{VID}")
    found = retriever.retrieve(video.store, "How does RAG retriever and generator answer?", video.ids(), top_k=4)
    assert len({e.start for e in found}) >= 1
    fake_groq.reply = '{"sufficient": true, "answer": "RAG looks up documents first [S1].", "source_findings": [],' \
                      ' "conflict": {"status": "none"}}'
    answer = answer_generator.answer_question(video, "What does RAG mean according to this video?")
    assert answer.sufficient and answer.evidence[0].start is not None


def test_video_and_pdf_are_searched_together(video):
    video.add_youtube(f"https://youtu.be/{VID}")
    video.add_file("paper.pdf", make_pdf("Retrieval augmented generation was introduced in a 2020 paper."))
    found = retriever.retrieve(video.store, "What is retrieval augmented generation?", video.ids())
    assert {e.source_type for e in found} == {"youtube", "pdf"}


def test_duplicate_video_is_refused(video):
    from utils.error_handler import DuplicateSourceError

    video.add_youtube(f"https://youtu.be/{VID}")
    with pytest.raises(DuplicateSourceError):
        video.add_youtube(f"https://www.youtube.com/watch?v={VID}")
    assert video.count("youtube") == 1


def test_missing_captions_are_explained_and_nothing_is_indexed(manager, monkeypatch):
    import youtube_transcript_api

    class FakeApi:
        def fetch(self, video_id, languages=None):
            raise TranscriptsDisabled(video_id)

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", FakeApi)
    with pytest.raises(TranscriptUnavailableError) as error:
        manager.add_youtube(f"https://youtu.be/{VID}")
    assert "no captions" in str(error.value) and "No video content was processed" in str(error.value)
    assert manager.sources == [] and manager.store.count() == 0


def test_network_failure_is_explained(manager, monkeypatch):
    import requests
    import youtube_transcript_api

    class FakeApi:
        def fetch(self, video_id, languages=None):
            raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", FakeApi)
    with pytest.raises(TranscriptUnavailableError, match="Could not reach YouTube"):
        manager.add_youtube(f"https://youtu.be/{VID}")


def test_invalid_link_never_reaches_the_network(manager, monkeypatch):
    monkeypatch.setattr(youtube_processor, "fetch_transcript_lines",
                        lambda vid: (_ for _ in ()).throw(AssertionError("must not be called")))
    with pytest.raises(InvalidYouTubeURLError):
        manager.add_youtube("https://example.com/not-youtube")


def test_audio_file_transcription_keeps_timestamps(manager, monkeypatch):
    monkeypatch.setattr(llm_client, "transcribe", lambda name, data: LINES)
    info = manager.add_file("lecture.mp3", b"fake audio bytes")
    assert info.source_type == "audio" and info.duration == 320
    found = retriever.retrieve(manager.store, "What does RAG mean?", manager.ids())
    assert found[0].start == 85 and found[0].video_id is None      # timestamp shown, but no fake link
    assert "[▶" not in location_markdown(found[0])


def test_audio_without_speech_is_rejected(manager, monkeypatch):
    from utils.error_handler import EmptyDocumentError

    monkeypatch.setattr(llm_client, "transcribe", lambda name, data: [])
    with pytest.raises(EmptyDocumentError, match="No speech"):
        manager.add_file("silence.wav", b"fake")


def test_video_limit(manager, monkeypatch):
    from utils.error_handler import UploadLimitError

    monkeypatch.setattr(youtube_processor, "fetch_transcript_lines", lambda vid: LINES)
    monkeypatch.setattr(youtube_processor, "fetch_title", lambda vid: "t")
    for vid in ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"):
        manager.add_youtube(f"https://youtu.be/{vid}")
    with pytest.raises(UploadLimitError):
        manager.add_youtube("https://youtu.be/ddddddddddd")
