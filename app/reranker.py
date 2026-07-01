"""
CrossEncoder reranker.

This module reranks the candidate chunks returned by the hybrid retriever.
Unlike embeddings, which encode the query and document separately, a
CrossEncoder jointly processes the query-document pair and produces a
relevance score.

Model:
    cross-encoder/ms-marco-MiniLM-L-6-v2

The reranker is applied AFTER hybrid retrieval (BM25 + Dense + RRF).
"""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int):
        if not chunks:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return chunks[:top_k]