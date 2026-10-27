"""Plain-text extraction for chat attachments and knowledge-base documents.

Port of ai_ux ``src/lib/extract.ts`` with the LocalCode differences:

* PDF through ``pypdf`` (page and character limits so a huge scan cannot pin
  a CPU for minutes), DOCX straight from the ZIP container
  (``word/document.xml`` + footnotes/endnotes, no extra dependency), HTML with
  tags stripped (scripts/styles dropped, block elements become line breaks).
* Text files are decoded as UTF-8 (BOM aware), then UTF-16 when a BOM says so,
  and finally as Windows-1250 — the usual encoding of old Polish files.
* Nothing here raises for content problems: :func:`extract_text` returns an
  :class:`ExtractResult` with a Polish ``error`` the UI can show verbatim.
  Everything is synchronous and CPU-bound; call it through
  ``asyncio.to_thread`` on request paths.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Literal
from xml.etree import ElementTree

AttachmentKind = Literal["image", "text", "pdf", "other"]

#: Extracted text longer than this is cut (about 1.2M tokens — far beyond any
#: local context; the knowledge base chunks it, the chat truncates it further).
MAX_TEXT_CHARS = 5_000_000
#: PDF pages read at most.
MAX_PDF_PAGES = 2_000
#: Uncompressed size limit of a DOCX XML part (zip-bomb guard).
MAX_DOCX_PART_BYTES = 64 * 1024 * 1024

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    # documents / data
    "txt", "text", "md", "markdown", "mdx", "rst", "adoc", "org", "tex", "bib",
    "log", "csv", "tsv", "json", "jsonc", "json5", "ndjson", "yaml", "yml",
    "toml", "ini", "cfg", "conf", "env", "properties", "xml", "srt", "vtt",
    "diff", "patch", "ics",
    # web
    "html", "htm", "xhtml", "css", "scss", "sass", "less", "vue", "svelte",
    "astro", "hbs", "ejs", "pug",
    # code
    "js", "mjs", "cjs", "jsx", "ts", "tsx", "mts", "cts", "py", "pyi", "rb",
    "php", "java", "kt", "kts", "swift", "c", "h", "cpp", "cc", "cxx", "hpp",
    "hh", "cs", "go", "rs", "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd",
    "sql", "graphql", "gql", "proto", "dart", "lua", "r", "jl", "pl", "pm",
    "ex", "exs", "erl", "hrl", "hs", "clj", "cljs", "edn", "scala", "groovy",
    "gradle", "vim", "asm", "makefile", "mk", "cmake", "dockerfile",
    "gitignore", "gitattributes", "editorconfig", "ipynb", "m", "mm", "zig",
    "nim", "ml", "mli", "fs", "fsx", "vb", "tf", "hcl", "nix", "cue",
})

TEXT_FILENAMES: frozenset[str] = frozenset({
    "dockerfile", "makefile", "cmakelists.txt", "procfile", "license", "licence",
    "readme", "changelog", "authors", "codeowners", "gemfile", "rakefile",
    "brewfile", "justfile", "caddyfile", "vagrantfile",
})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    "png", "jpg", "jpeg", "jfif", "gif", "webp", "avif", "bmp", "svg", "ico",
    "heic", "heif", "tif", "tiff",
})

HTML_EXTENSIONS: frozenset[str] = frozenset({"html", "htm", "xhtml"})

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

TEXT_MIME_HINTS: frozenset[str] = frozenset({
    "application/json", "application/ld+json", "application/xml",
    "application/xhtml+xml", "application/javascript", "application/ecmascript",
    "application/x-javascript", "application/typescript", "application/x-sh",
    "application/x-shellscript", "application/x-yaml", "application/yaml",
    "application/toml", "application/x-ndjson", "application/csv",
    "application/sql", "application/x-httpd-php",
})

MIME_BY_EXTENSION: dict[str, str] = {
    "txt": "text/plain", "text": "text/plain", "log": "text/plain",
    "md": "text/markdown", "markdown": "text/markdown",
    "csv": "text/csv", "tsv": "text/tab-separated-values",
    "json": "application/json", "ndjson": "application/x-ndjson",
    "yaml": "application/yaml", "yml": "application/yaml", "toml": "application/toml",
    "xml": "application/xml", "html": "text/html", "htm": "text/html", "css": "text/css",
    "js": "text/javascript", "mjs": "text/javascript",
    "ts": "text/plain", "tsx": "text/plain", "jsx": "text/plain",
    "py": "text/x-python", "sql": "application/sql", "sh": "application/x-sh",
    "pdf": "application/pdf", "docx": DOCX_MIME,
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "jfif": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp", "avif": "image/avif", "bmp": "image/bmp",
    "ico": "image/x-icon", "svg": "image/svg+xml", "heic": "image/heic",
    "heif": "image/heif", "tif": "image/tiff", "tiff": "image/tiff",
}


@dataclass
class ExtractResult:
    text: str
    #: Human-readable (Polish) reason why nothing could be extracted.
    error: str | None = None
    #: True when the text was cut at :data:`MAX_TEXT_CHARS` / :data:`MAX_PDF_PAGES`.
    truncated: bool = False


# --------------------------------------------------------------------------- #
# Name / type helpers
# --------------------------------------------------------------------------- #


def base_name(name: str) -> str:
    cleaned = (name or "").replace("\\", "/")
    return cleaned.split("/")[-1].strip().lower()


def file_extension(name: str) -> str:
    """Lower-case extension without the dot, or ``""``."""
    base = base_name(name)
    dot = base.rfind(".")
    if dot <= 0 or dot == len(base) - 1:
        return ""
    return base[dot + 1:]


def display_name(name: str, fallback: str = "plik") -> str:
    """File name without any directory part a client may have sent."""
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    return base or fallback


def safe_extension(name: str) -> str:
    """Short, filesystem-safe ``.ext`` for stored files (or ``""``)."""
    ext = re.sub(r"[^a-z0-9]", "", file_extension(name))
    if not ext or len(ext) > 12:
        return ""
    return f".{ext}"


def normalize_mime(mime: str | None) -> str:
    return (mime or "").split(";")[0].strip().lower()


def resolve_mime(reported: str | None, name: str) -> str:
    """Browsers often send an empty or generic type — fall back to the extension."""
    mime = normalize_mime(reported)
    if mime and mime != "application/octet-stream":
        return mime
    return MIME_BY_EXTENSION.get(file_extension(name), "application/octet-stream")


def _is_pdf(ext: str, mime: str) -> bool:
    return mime in ("application/pdf", "application/x-pdf") or ext == "pdf"


def _is_docx(ext: str, mime: str) -> bool:
    return mime == DOCX_MIME or ext in ("docx", "docm")


def _is_html(ext: str, mime: str) -> bool:
    return mime in ("text/html", "application/xhtml+xml") or ext in HTML_EXTENSIONS


def _is_text(name: str, ext: str, mime: str) -> bool:
    if mime.startswith("text/"):
        return True
    if mime in TEXT_MIME_HINTS:
        return True
    if mime.endswith("+json") or mime.endswith("+xml"):
        return True
    if ext and ext in TEXT_EXTENSIONS:
        return True
    if not ext and base_name(name) in TEXT_FILENAMES:
        return True
    return False


def classify_file(name: str, mime: str | None) -> AttachmentKind:
    """Coarse attachment kind stored in the database."""
    m = normalize_mime(mime)
    ext = file_extension(name)
    if m.startswith("image/") or ext in IMAGE_EXTENSIONS:
        return "image"
    if _is_pdf(ext, m):
        return "pdf"
    if _is_docx(ext, m):
        return "text"          # Word documents end up as plain text
    if _is_text(name, ext, m):
        return "text"
    return "other"


def looks_binary(data: bytes) -> bool:
    """NUL bytes or a high ratio of control characters means binary."""
    sample = data[:8000]
    if not sample:
        return False
    if b"\x00" in sample:
        # UTF-16 text has NULs too; a BOM tells us it is text after all.
        return not (sample.startswith(b"\xff\xfe") or sample.startswith(b"\xfe\xff"))
    suspicious = sum(1 for b in sample if b < 32 and b not in (9, 10, 12, 13, 27))
    return suspicious / len(sample) > 0.1


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_WS_RE = re.compile(r"[^\S\n]+\n")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_extracted_text(raw: str) -> str:
    """Normalise line endings, strip the BOM and control junk."""
    text = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def decode_text(data: bytes) -> str:
    """UTF-8 (BOM aware) → UTF-16 with BOM → Windows-1250 (old Polish files)."""
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1250")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    return text[:MAX_TEXT_CHARS], True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def extract_text(data: bytes, name: str, mime: str | None) -> ExtractResult:
    """Plain text of an uploaded file. Never raises for content problems."""
    m = normalize_mime(mime)
    ext = file_extension(name)

    if not data:
        return ExtractResult("", "Plik jest pusty")

    # Images carry no text; vision models would get the raw bytes.
    if classify_file(name, m) == "image":
        return ExtractResult("")

    try:
        if _is_pdf(ext, m):
            return _extract_pdf(data)
        if _is_docx(ext, m):
            return _extract_docx(data)
        if ext == "doc":
            return ExtractResult("", "Nieobsługiwany format pliku (.doc) — zapisz dokument jako .docx")
        if ext == "rtf":
            return ExtractResult("", "Nieobsługiwany format pliku (.rtf)")

        if _is_text(name, ext, m) or not looks_binary(data):
            if looks_binary(data):
                return ExtractResult("", "Plik wygląda na binarny, nie udało się odczytać tekstu")
            decoded = decode_text(data)
            if _is_html(ext, m):
                decoded = html_to_text(decoded)
            text, truncated = _cap(normalize_extracted_text(decoded))
            if not text:
                return ExtractResult("", "Plik nie zawiera tekstu")
            return ExtractResult(text, None, truncated)

        return ExtractResult("", "Nieobsługiwany format pliku")
    except Exception as exc:  # defensive: third-party parsers raise anything
        return ExtractResult("", f"Nie udało się odczytać pliku: {_message_of(exc)}")


def _message_of(exc: BaseException) -> str:
    text = str(exc).strip()
    return text[:300] if text else type(exc).__name__


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
    "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "tr", "ul", "td", "th", "caption", "summary", "details",
})
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head", "iframe", "object"})


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._title: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            return
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag in ("td", "th"):
            self.parts.append("\t")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("br", "hr"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
            return
        if self._skip:
            return
        self.parts.append(data)

    def text(self) -> str:
        body = "".join(self.parts)
        body = re.sub(r"[^\S\n]+", " ", body)
        body = re.sub(r" ?\n ?", "\n", body)
        title = " ".join("".join(self._title).split())
        if title and title not in body[: len(title) + 200]:
            body = f"{title}\n\n{body}"
        return body


def html_to_text(source: str) -> str:
    """Visible text of an HTML document (no tags, scripts or styles)."""
    parser = _HTMLText()
    try:
        parser.feed(source)
        parser.close()
    except Exception:
        # Pathological markup: fall back to a blunt tag strip.
        return unescape(re.sub(r"<[^>]+>", " ", source))
    return parser.text()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def _extract_pdf(data: bytes) -> ExtractResult:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:  # pragma: no cover - pypdf is a declared dependency
        return ExtractResult("", "Obsługa PDF jest niedostępna (brak biblioteki pypdf)")

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except PdfReadError as exc:
        return ExtractResult("", f"Nie udało się odczytać pliku PDF: {_message_of(exc)}")
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                return ExtractResult("", "Plik PDF jest zabezpieczony hasłem")
        except Exception:
            return ExtractResult("", "Plik PDF jest zabezpieczony hasłem")

    pages: list[str] = []
    total = 0
    truncated = False
    for index, page in enumerate(reader.pages):
        if index >= MAX_PDF_PAGES or total >= MAX_TEXT_CHARS:
            truncated = True
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""  # one broken page must not lose the whole document
        pages.append(text)
        total += len(text)

    text, cut = _cap(normalize_extracted_text("\n\n".join(pages)))
    if not text:
        return ExtractResult("", "Nie znaleziono tekstu w pliku PDF (prawdopodobnie skan lub same obrazy)")
    return ExtractResult(text, None, truncated or cut)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_PARTS = ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml")


def _docx_part_text(xml_bytes: bytes) -> str:
    head = xml_bytes[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in xml_bytes[:65536].upper():
        raise ValueError("dokument zawiera niedozwolone deklaracje DTD")
    root = ElementTree.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for para in root.iter(f"{_W_NS}p"):
        pieces: list[str] = []
        for node in para.iter():
            tag = node.tag
            if tag == f"{_W_NS}t" and node.text:
                pieces.append(node.text)
            elif tag == f"{_W_NS}tab":
                pieces.append("\t")
            elif tag in (f"{_W_NS}br", f"{_W_NS}cr"):
                pieces.append("\n")
            elif tag == f"{_W_NS}noBreakHyphen":
                pieces.append("-")
        paragraphs.append("".join(pieces))
    return "\n".join(paragraphs)


def _extract_docx(data: bytes) -> ExtractResult:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return ExtractResult("", "Plik nie jest poprawnym dokumentem Word (.docx)")
    with archive:
        names = set(archive.namelist())
        if "word/document.xml" not in names:
            return ExtractResult("", "Plik nie jest poprawnym dokumentem Word (.docx)")
        texts: list[str] = []
        for part in _DOCX_PARTS:
            if part not in names:
                continue
            info = archive.getinfo(part)
            if info.file_size > MAX_DOCX_PART_BYTES:
                return ExtractResult("", "Dokument Word jest zbyt duży do przetworzenia")
            texts.append(_docx_part_text(archive.read(part)))
    text, truncated = _cap(normalize_extracted_text("\n\n".join(t for t in texts if t.strip())))
    if not text:
        return ExtractResult("", "Dokument Word nie zawiera tekstu")
    return ExtractResult(text, None, truncated)
