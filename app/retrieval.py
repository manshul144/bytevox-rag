"""
Retrieval layer: Hybrid Search via Reciprocal Rank Fusion (RRF).

Why hybrid instead of pure semantic similarity?
- Dense embeddings are great at *meaning* (paraphrases, synonyms) but
  routinely under-perform on exact-match needs: product names, SKU codes,
  numbers, acronyms, error codes -- tokens that are rare in the embedding
  model's training distribution.
- BM25 (sparse, term-frequency based) is the inverse: excellent at exact
  keyword/phrase matches, weak at paraphrase/synonym understanding.
- Reciprocal Rank Fusion combines the *rankings* (not raw scores, which
  live on incompatible scales) from both retrievers into a single ranked
  list: score(d) = sum over retrievers of 1 / (k + rank_in_that_retriever).
  RRF is simple, requires no score normalization/calibration, and is a
  well-established baseline (used in Elasticsearch, Azure AI Search, etc.)
  for combining heterogeneous retrievers.

This module also re-ranks the fused list with a lightweight keyword-overlap
boost (a cheap proxy for cross-encoder reranking) before truncating to
TOP_K_FINAL -- see DESIGN_DECISIONS.md for trade-offs vs. a real reranker.
"""
import pickle
import os
import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from app.config import settings
from app.chunking import Chunk
from app.vectorstore import VectorStore
from app.reranker import CrossEncoderReranker


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25: BM25Okapi | None = None
        self.bm25_chunks: List[Chunk] = []
        self.reranker = CrossEncoderReranker()

    # ---------- Index building ----------

    def build_bm25_index(self, chunks: List[Chunk]):
        self.bm25_chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._persist_bm25()

    def _persist_bm25(self):
        os.makedirs(os.path.dirname(settings.bm25_index_path), exist_ok=True)
        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump(self.bm25_chunks, f)

    def load_bm25_index(self) -> bool:
        if not os.path.exists(settings.bm25_index_path):
            return False
        with open(settings.bm25_index_path, "rb") as f:
            self.bm25_chunks = pickle.load(f)
        tokenized = [_tokenize(c.text) for c in self.bm25_chunks]
        self.bm25 = BM25Okapi(tokenized)
        return True

    # ---------- Search ----------

    def sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.bm25_chunks:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for idx in ranked_idx:
            c = self.bm25_chunks[idx]
            hits.append(
                {"chunk_id": c.chunk_id, "text": c.text, "source": c.source, "score": float(scores[idx])}
            )
        return hits

    def hybrid_search(self, query: str, top_k_final: int = None) -> List[Dict[str, Any]]:
        top_k_final = top_k_final or settings.top_k_final

        dense_hits = self.vector_store.dense_search(query, settings.top_k_dense)
        sparse_hits = self.sparse_search(query, settings.top_k_sparse)

        # --- Reciprocal Rank Fusion ---
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            rrf_scores[hit["chunk_id"]] = rrf_scores.get(hit["chunk_id"], 0) + 1 / (settings.rrf_k + rank + 1)
            chunk_lookup[hit["chunk_id"]] = hit

        for rank, hit in enumerate(sparse_hits):
            rrf_scores[hit["chunk_id"]] = rrf_scores.get(hit["chunk_id"], 0) + 1 / (settings.rrf_k + rank + 1)
            chunk_lookup.setdefault(hit["chunk_id"], hit)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        query_terms = set(_tokenize(query))
        for chunk_id, rrf_score in fused[: top_k_final * 2]:
            hit = dict(chunk_lookup[chunk_id])
            # Lightweight lexical-overlap boost as a cheap reranking signal
            # (a stand-in for a cross-encoder reranker; see DESIGN_DECISIONS.md)
            overlap = len(query_terms & set(_tokenize(hit["text"]))) / max(len(query_terms), 1)
            hit["rrf_score"] = rrf_score
            hit["final_score"] = rrf_score * (1 + 0.25 * overlap)
            results.append(hit)

        
        results.sort(key=lambda h: h["final_score"], reverse=True)

# Take more candidates from hybrid retrieval
        candidate_chunks = results[: top_k_final * 2]

# CrossEncoder reranking
        reranked = self.reranker.rerank(
    query=query,
    chunks=candidate_chunks,
    top_k=top_k_final,
)

        return reranked
