"""Scenarios 1-4: uploading 1, 3, 5 PDFs and the sixth being refused; plus bad-file handling."""
import pytest

from config import settings
from tests.conftest import blank_pdf, make_pdf
from utils.error_handler import (
    CorruptedFileError,
    DuplicateSourceError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileError,
    UploadLimitError,
)


def pdf(n):
    return make_pdf(f"Document number {n} talks about topic{n}. " * 3)


def test_upload_one_pdf(manager):                                       # scenario 1
    info = manager.add_file("a.pdf", make_pdf("First page about kittens.", "Second page about puppies."))
    assert info.source_type == "pdf" and info.page_count == 2 and info.chunk_count >= 2
    assert [s.filename for s in manager.sources] == ["a.pdf"]
    assert manager.store.count() == info.chunk_count


def test_upload_three_pdfs(manager):                                    # scenario 2
    for n in range(3):
        manager.add_file(f"d{n}.pdf", pdf(n))
    assert [s.filename for s in manager.sources] == ["d0.pdf", "d1.pdf", "d2.pdf"]
    assert manager.count("pdf") == 3 and manager.pdf_slots_left() == 2


def test_upload_five_pdfs_is_allowed(manager):                          # scenario 3
    for n in range(5):
        manager.add_file(f"d{n}.pdf", pdf(n))
    assert manager.count("pdf") == 5 and manager.pdf_slots_left() == 0


def test_sixth_pdf_is_rejected_by_the_backend(manager):                 # scenario 4
    for n in range(5):
        manager.add_file(f"d{n}.pdf", pdf(n))
    chunks_before = manager.store.count()
    with pytest.raises(UploadLimitError) as error:
        manager.add_file("sixth.pdf", pdf(6))
    assert str(error.value) == "Maximum upload limit reached. You can upload only 5 PDF files."
    assert manager.count("pdf") == 5
    assert manager.store.count() == chunks_before          # the rejected file was never indexed


def test_limit_info_text():
    assert settings.PDF_LIMIT_INFO == "Maximum 5 PDFs can be uploaded."


def test_users_may_upload_fewer_than_five(manager):
    manager.add_file("only.pdf", pdf(1))
    assert manager.count("pdf") == 1          # nothing forces 5


def test_invalid_pdf_bytes(manager):
    with pytest.raises(CorruptedFileError, match="not a valid PDF"):
        manager.add_file("broken.pdf", b"this is not a pdf at all")
    assert manager.sources == []


def test_truncated_pdf(manager):
    with pytest.raises(CorruptedFileError):
        manager.add_file("cut.pdf", pdf(1)[:120])


def test_pdf_without_text_is_reported_as_empty(manager):
    with pytest.raises(EmptyDocumentError, match="scanned"):
        manager.add_file("blank.pdf", blank_pdf())


def test_unsupported_format_has_clear_message(manager):
    with pytest.raises(UnsupportedFileError, match="not supported"):
        manager.add_file("sheet.xlsx", b"PK...")
    with pytest.raises(UnsupportedFileError):
        manager.add_file("noextension", b"data")


def test_oversized_and_empty_files(manager):
    with pytest.raises(FileTooLargeError, match="larger than"):
        manager.add_file("big.pdf", b"0" * (settings.MAX_FILE_MB * 1024 * 1024 + 1))
    with pytest.raises(FileTooLargeError, match="empty"):
        manager.add_file("zero.txt", b"")


def test_duplicate_pdf_is_refused_and_not_double_indexed(manager):
    data = pdf(1)
    manager.add_file("a.pdf", data)
    chunks = manager.store.count()
    with pytest.raises(DuplicateSourceError, match="already"):
        manager.add_file("copy-of-a.pdf", data)
    assert manager.store.count() == chunks and manager.count("pdf") == 1


def test_failed_upload_leaves_existing_sources_untouched(manager):
    manager.add_file("good.pdf", pdf(1))
    with pytest.raises(CorruptedFileError):
        manager.add_file("bad.pdf", b"nope")
    assert [s.filename for s in manager.sources] == ["good.pdf"]
