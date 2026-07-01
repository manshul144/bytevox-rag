"""
CLI ingestion script: builds (or rebuilds) the BM25 + Chroma indexes from
DOCS_DIR. Equivalent to calling POST /ingest, but runnable standalone.

Usage:
    python scripts/ingest.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.ingestion import load_documents_from_dir
from app.chunking import chunk_documents
from app.vectorstore import VectorStore
from app.retrieval import HybridRetriever


def main():
    print(f"Loading documents from {settings.docs_dir} ...")
    docs = load_documents_from_dir(settings.docs_dir)
    print(f"Loaded {len(docs)} documents: {[d.source for d in docs]}")

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    print(f"Created {len(chunks)} chunks (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    vector_store = VectorStore()
    vector_store.reset()
    print("Embedding and writing chunks to ChromaDB ...")
    vector_store.add_chunks(chunks)

    retriever = HybridRetriever(vector_store)
    print("Building BM25 sparse index ...")
    retriever.build_bm25_index(chunks)

    print(f"Done. {vector_store.count()} vectors indexed in Chroma collection.")


if __name__ == "__main__":
    main()
