"""Text extraction for TXT, MD, DOCX and CSV files."""
import csv
import io
from typing import List

from backend.models import Segment
from utils.error_handler import CorruptedFileError, EmptyDocumentError
from utils.helpers import clean_text
from utils.validators import file_extension

CSV_ROWS_PER_SEGMENT = 8


def _decode(data: bytes, filename: str) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise CorruptedFileError(f"'{filename}' could not be decoded as text.")


def _extract_docx(data: bytes, filename: str) -> str:
    try:
        import docx

        document = docx.Document(io.BytesIO(data))
    except Exception:
        raise CorruptedFileError(f"'{filename}' is not a valid DOCX file or is corrupted.")
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _extract_csv(data: bytes, filename: str) -> List[Segment]:
    text = _decode(data, filename)
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        raise CorruptedFileError(f"'{filename}' could not be parsed as CSV.")
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise EmptyDocumentError(f"'{filename}' has no data rows.")
    header, body = [h.strip() for h in rows[0]], rows[1:]
    segments = []
    for start in range(0, len(body), CSV_ROWS_PER_SEGMENT):
        lines = []
        for offset, row in enumerate(body[start:start + CSV_ROWS_PER_SEGMENT]):
            fields = [f"{h or f'column {i + 1}'}: {v.strip()}" for i, (h, v) in enumerate(zip(header, row))]
            lines.append(f"Row {start + offset + 1}: " + "; ".join(fields))
        segments.append(Segment(text="\n".join(lines)))
    return segments


def extract_file(data: bytes, filename: str) -> List[Segment]:
    """Return text segments for a non-PDF document. Raises a clear error on failure."""
    ext = file_extension(filename)
    if ext == ".csv":
        return _extract_csv(data, filename)
    if ext == ".docx":
        text = _extract_docx(data, filename)
    elif ext in (".txt", ".md"):
        text = _decode(data, filename)
    else:
        raise CorruptedFileError(f"No extractor for '{filename}'.")
    text = clean_text(text)
    if not text:
        raise EmptyDocumentError(f"'{filename}' contains no readable text.")
    return [Segment(text=text)]
