"""
Chunking strategy.

We use a recursive, paragraph-aware splitter:
1. Split on double newlines (paragraphs) first, since that respects natural
   document structure (sections, list items, markdown headers).
2. Greedily pack paragraphs into chunks up to `chunk_size` characters.
3. If a single paragraph is itself larger than `chunk_size` (e.g. a dense
   PDF paragraph with no breaks), fall back to a sliding window split.
4. Apply a character-based overlap between consecutive chunks so that
   information sitting on a chunk boundary is not lost to retrieval.

Character-based (not token-based) sizing keeps the implementation dependency
-free; see DESIGN_DECISIONS.md for why ~800 chars / 120 overlap was chosen.
"""
import re
from dataclasses import dataclass
from typing import List

from app.ingestion import RawDocument


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def _split_paragraphs(text: str) -> List[str]:
    # Normalize whitespace, then split on blank lines.
    text = re.sub(r"\r\n", "\n", text)
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_document(doc: RawDocument, chunk_size: int = 800, overlap: int = 120) -> List[Chunk]:
    paragraphs = _split_paragraphs(doc.text)
    if not paragraphs:
        return []

    raw_chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush whatever is buffered, then window-split the long paragraph.
            if buffer:
                raw_chunks.append(buffer)
                buffer = ""
            raw_chunks.extend(_sliding_window(para, chunk_size, overlap))
            continue

        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            raw_chunks.append(buffer)
            buffer = para

    if buffer:
        raw_chunks.append(buffer)

    # Apply overlap between consecutive chunks (carry tail of previous chunk
    # forward) so context isn't lost at boundaries.
    overlapped: List[str] = []
    for i, c in enumerate(raw_chunks):
        if i == 0 or overlap == 0:
            overlapped.append(c)
        else:
            tail = raw_chunks[i - 1][-overlap:]
            overlapped.append(f"{tail}\n\n{c}")

    chunks = [
        Chunk(chunk_id=f"{doc.source}::{i}", source=doc.source, text=c)
        for i, c in enumerate(overlapped)
    ]
    return chunks


def chunk_documents(docs: List[RawDocument], chunk_size: int = 800, overlap: int = 120) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
