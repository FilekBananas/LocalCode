"""Text extraction: text/code/csv/json/html/pdf/docx, images and failures."""

from __future__ import annotations

import io
import zipfile

import pytest

from localcode.chat import extract as ex
from localcode.chat.extract import classify_file, extract_text, html_to_text, resolve_mime


def make_pdf(pages: list[str | None]) -> bytes:
    """Minimal valid PDF; ``None`` pages have no text operators (like a scan)."""
    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                       f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        stream = b"" if text is None else f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_docx(paragraphs: list[str], *, footnote: str | None = None, doctype: bool = False) -> bytes:
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(
        f"<w:p><w:r><w:t>{p.split('|')[0]}</w:t></w:r>"
        + (f"<w:r><w:tab/><w:t>{p.split('|')[1]}</w:t></w:r>" if "|" in p else "")
        + "</w:p>"
        for p in paragraphs
    )
    prolog = '<?xml version="1.0" encoding="UTF-8"?>'
    if doctype:
        prolog += '<!DOCTYPE lol [<!ENTITY a "aaaa">]>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f"{prolog}<w:document {ns}><w:body>{body}</w:body></w:document>")
        if footnote:
            z.writestr("word/footnotes.xml",
                       f"{prolog}<w:footnotes {ns}><w:footnote><w:p><w:r><w:t>{footnote}</w:t></w:r></w:p>"
                       f"</w:footnote></w:footnotes>")
    return buf.getvalue()


# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,mime,kind", [
    ("photo.JPG", "", "image"),
    ("x.bin", "image/png", "image"),
    ("doc.pdf", "application/octet-stream", "pdf"),
    ("raport.docx", "", "text"),
    ("main.py", "", "text"),
    ("Dockerfile", "", "text"),
    ("data.json", "application/json", "text"),
    ("archive.zip", "application/zip", "other"),
])
def test_classify_file(name: str, mime: str, kind: str) -> None:
    assert classify_file(name, mime) == kind


def test_resolve_mime_and_names() -> None:
    assert resolve_mime("", "a.md") == "text/markdown"
    assert resolve_mime("application/octet-stream", "a.pdf") == "application/pdf"
    assert resolve_mime("Text/Plain; charset=utf-8", "a.bin") == "text/plain"
    assert resolve_mime(None, "a.unknownext") == "application/octet-stream"
    assert ex.display_name("C:\\Users\\x\\plik.txt") == "plik.txt"
    assert ex.display_name("   ") == "plik"
    assert ex.safe_extension("a.tar.GZ") == ".gz"
    assert ex.safe_extension("a.verylongextension1") == ""
    assert ex.safe_extension("noext") == ""


def test_plain_text_code_csv_json() -> None:
    r = extract_text("Zażółć\r\ngęślą\x07 jaźń   \n\n\n\nkoniec".encode(), "a.txt", "text/plain")
    assert r.error is None and r.text == "Zażółć\ngęślą jaźń\n\nkoniec"
    assert extract_text(b"def f():\n    return 1\n", "m.py", "").text == "def f():\n    return 1"
    assert extract_text(b"a,b\n1,2\n", "t.csv", "text/csv").text == "a,b\n1,2"
    assert extract_text(b'{"k": "w"}', "d.json", "application/json").text == '{"k": "w"}'
    assert extract_text("\ufeffBOM tekst".encode(), "b.md", "").text == "BOM tekst"


def test_legacy_encodings() -> None:
    assert extract_text("Zażółć gęślą jaźń".encode("cp1250"), "old.txt", "text/plain").text == "Zażółć gęślą jaźń"
    assert extract_text("Łódź".encode("utf-16"), "u16.txt", "text/plain").text == "Łódź"


def test_html_is_stripped() -> None:
    html = ("<html><head><title>Tytuł</title><style>p{color:red}</style></head><body>"
            "<h1>Nagłówek</h1><p>Akapit &amp; <b>pogrubienie</b></p><script>alert(1)</script>"
            "<ul><li>jeden</li><li>dwa</li></ul>linia<br>druga</body></html>")
    r = extract_text(html.encode(), "page.html", "text/html")
    assert r.error is None
    assert "alert" not in r.text and "color:red" not in r.text and "<" not in r.text
    lines = [line.strip() for line in r.text.split("\n") if line.strip()]
    assert lines[0] == "Tytuł"
    assert "Nagłówek" in lines and "Akapit & pogrubienie" in lines and "jeden" in lines and "druga" in lines
    assert html_to_text("<p>a<p>b") .split() == ["a", "b"]


def test_pdf_extraction() -> None:
    r = extract_text(make_pdf(["Hello PDF world", "Second page text"]), "doc.pdf", "application/pdf")
    assert r.error is None
    assert "Hello PDF world" in r.text and "Second page text" in r.text


def test_pdf_without_text_and_broken_pdf() -> None:
    r = extract_text(make_pdf([None]), "scan.pdf", "application/pdf")
    assert r.text == "" and "skan" in r.error
    broken = extract_text(b"%PDF-1.4 garbage", "b.pdf", "application/pdf")
    assert broken.text == "" and broken.error


def test_pdf_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "MAX_PDF_PAGES", 1)
    r = extract_text(make_pdf(["Page one", "Page two"]), "d.pdf", "application/pdf")
    assert "Page one" in r.text and "Page two" not in r.text and r.truncated


def test_encrypted_pdf() -> None:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(make_pdf(["Secret"]))).pages:
        writer.add_page(page)
    writer.encrypt(user_password="haslo", owner_password="owner")
    buf = io.BytesIO()
    writer.write(buf)
    r = extract_text(buf.getvalue(), "secret.pdf", "application/pdf")
    assert r.text == "" and "hasłem" in r.error


def test_docx_extraction() -> None:
    data = make_docx(["Pierwszy akapit", "Kolumna|Wartość", "Ostatni"], footnote="Przypis dolny")
    r = extract_text(data, "raport.docx", "")
    assert r.error is None
    assert r.text.split("\n")[:3] == ["Pierwszy akapit", "Kolumna\tWartość", "Ostatni"]
    assert "Przypis dolny" in r.text


def test_docx_failures() -> None:
    assert "Word" in extract_text(b"PK not a zip", "x.docx", "").error
    empty = make_docx([""])
    assert extract_text(empty, "e.docx", "").error == "Dokument Word nie zawiera tekstu"
    assert "DTD" in extract_text(make_docx(["x"], doctype=True), "d.docx", "").error
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("other.xml", "<a/>")
    assert "poprawnym" in extract_text(buf.getvalue(), "n.docx", "").error


def test_images_empty_binary_and_unsupported() -> None:
    assert extract_text(b"\x89PNG\r\n", "a.png", "image/png").text == ""
    assert extract_text(b"\x89PNG\r\n", "a.png", "image/png").error is None
    assert extract_text(b"", "a.txt", "text/plain").error == "Plik jest pusty"
    assert "binarny" in extract_text(b"\x00\x01\x02abc", "a.txt", "text/plain").error
    assert extract_text(b"\x00\x01\x02abc", "a.bin", "application/octet-stream").error == "Nieobsługiwany format pliku"
    assert ".docx" in extract_text(b"abc", "old.doc", "").error
    assert ".rtf" in extract_text(b"{\\rtf1}", "a.rtf", "").error
    assert extract_text(b"   \n  ", "blank.txt", "text/plain").error == "Plik nie zawiera tekstu"
    # unknown extension that decodes cleanly is accepted
    assert extract_text(b"plain words", "notes.weird", "").text == "plain words"


def test_text_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "MAX_TEXT_CHARS", 10)
    r = extract_text(b"0123456789abcdef", "a.txt", "text/plain")
    assert r.text == "0123456789" and r.truncated
