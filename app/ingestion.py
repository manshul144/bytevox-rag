"""
Document ingestion: load PDF / Markdown / plain text files into raw text,
tagged with their source filename.
"""
import os
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


@dataclass
class RawDocument:
    source: str          # filename, used for citation in API responses
    text: str            # full extracted text


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def _load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        content = page.extract_text() or ""
        pages.append(content)
    return "\n\n".join(pages)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path: str) -> RawDocument:
    """Load a single file based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    filename = os.path.basename(path)

    if ext == ".pdf":
        text = _load_pdf(path)
    elif ext in (".md", ".markdown", ".txt"):
        text = _load_text(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return RawDocument(source=filename, text=text)


def load_documents_from_dir(directory: str) -> List[RawDocument]:
    """Walk a directory and load every supported file."""
    docs: List[RawDocument] = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, fname)
                try:
                    docs.append(load_document(full_path))
                except Exception as e:
                    print(f"[ingestion] Skipped {fname}: {e}")
    return docs
