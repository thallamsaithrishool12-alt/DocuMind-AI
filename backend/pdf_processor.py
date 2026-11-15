"""PDF text extraction with page numbers."""
import io
from typing import List

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from backend.models import Segment
from utils.error_handler import CorruptedFileError, EmptyDocumentError
from utils.helpers import clean_text


def extract_pdf(data: bytes, filename: str):
    """Return (segments, page_count): one Segment per page with text (pages start at 1)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise CorruptedFileError(f"'{filename}' is password protected and cannot be read.")
            except CorruptedFileError:
                raise
            except Exception:
                raise CorruptedFileError(f"'{filename}' is password protected and cannot be read.")
        segments = []
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                segments.append(Segment(text=text, page=number))
        page_count = len(reader.pages)
    except CorruptedFileError:
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, OSError, RecursionError):
        raise CorruptedFileError(f"'{filename}' is not a valid PDF or is corrupted.")
    except Exception:
        raise CorruptedFileError(f"'{filename}' could not be read as a PDF.")

    if not segments:
        raise EmptyDocumentError(
            f"No readable text found in '{filename}'. It may be a scanned PDF (images only); "
            "scanned PDFs need OCR, which this app does not do."
        )
    return segments, page_count
