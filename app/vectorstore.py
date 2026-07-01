"""
Thin wrapper around ChromaDB for persistent dense vector storage.
"""
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.chunking import Chunk

COLLECTION_NAME = "bytevox_docs"

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64):
        embedder = get_embedder()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{"source": c.source} for c in batch],
            )

    def count(self) -> int:
        return self.collection.count()

    def dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        embedder = get_embedder()
        query_emb = embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)

        hits = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance; convert to a similarity score in [0, 1]
            similarity = 1 - dist
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "score": similarity,
                }
            )
        return hits
