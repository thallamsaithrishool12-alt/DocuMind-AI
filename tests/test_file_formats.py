"""Scenario 14 (Markdown and TXT) plus DOCX and CSV."""
import io

import docx
import pytest

from backend import retriever
from utils.error_handler import CorruptedFileError, EmptyDocumentError


def make_docx(paragraphs, table=None):
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                t.cell(i, j).text = cell
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_txt_upload_is_searchable(manager):
    manager.add_file("notes.txt", b"Photosynthesis converts sunlight into chemical energy in plants.")
    found = retriever.retrieve(manager.store, "How does photosynthesis work?", manager.ids())
    assert found and found[0].filename == "notes.txt" and found[0].source_type == "document"


def test_markdown_upload_is_searchable(manager):
    md = b"# Volcanoes\n\nA volcano erupts when magma reaches the surface.\n\n## Types\n\n- shield\n- cone"
    info = manager.add_file("geo.md", md)
    found = retriever.retrieve(manager.store, "When does a volcano erupt?", manager.ids())
    assert info.source_type == "document" and found[0].filename == "geo.md"
    assert "magma" in found[0].text


def test_docx_text_and_tables(manager):
    data = make_docx(["The reactor uses thorium fuel."], table=[["Part", "Material"], ["Core", "Graphite"]])
    manager.add_file("plant.docx", data)
    found = retriever.retrieve(manager.store, "Which material is the core?", manager.ids())
    assert found and "Graphite" in found[0].text


def test_csv_rows_keep_their_headers(manager):
    csv_data = b"city,population\nParis,2100000\nLyon,520000\n"
    manager.add_file("cities.csv", csv_data)
    found = retriever.retrieve(manager.store, "population of Lyon", manager.ids())
    assert found and "city: Lyon; population: 520000" in found[0].text


def test_text_with_non_utf8_bytes(manager):
    manager.add_file("latin.txt", "Café société résumé".encode("cp1252"))
    assert manager.count("document") == 1


def test_empty_text_file_is_rejected(manager):
    with pytest.raises(EmptyDocumentError):
        manager.add_file("blank.txt", b"   \n  \n")


def test_corrupted_docx_is_rejected(manager):
    with pytest.raises(CorruptedFileError, match="DOCX"):
        manager.add_file("bad.docx", b"not a zip")


def test_csv_without_data_rows(manager):
    with pytest.raises(EmptyDocumentError):
        manager.add_file("headers.csv", b"a,b,c\n")
