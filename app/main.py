"""
ByteVox RAG API.

Endpoints:
  POST /ingest  -> (re)indexes all documents in DOCS_DIR
  POST /query   -> answers a question using hybrid retrieval + LLM generation
  GET  /health  -> basic liveness + index size check
"""
import time
import logging

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.ingestion import load_documents_from_dir
from app.chunking import chunk_documents
from app.vectorstore import VectorStore
from app.retrieval import HybridRetriever
from app.llm import generate_answer
from app.observability import log_query, fetch_recent_logs
from app.schemas import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
    IngestResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bytevox")

app = FastAPI(title="ByteVox RAG API", version="1.0.0")

vector_store = VectorStore()
retriever = HybridRetriever(vector_store)

# Try to warm-start the BM25 index from disk if it already exists
retriever.load_bm25_index()


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """(Re)index every document found in DOCS_DIR."""
    docs = load_documents_from_dir(settings.docs_dir)
    if not docs:
        raise HTTPException(status_code=400, detail=f"No supported documents found in {settings.docs_dir}")

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)

    vector_store.reset()
    vector_store.add_chunks(chunks)
    retriever.build_bm25_index(chunks)

    logger.info(f"Ingested {len(docs)} documents -> {len(chunks)} chunks")

    return IngestResponse(
        documents_processed=len(docs),
        chunks_indexed=len(chunks),
        files=[d.source for d in docs],
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.perf_counter()

    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Call POST /ingest first.",
        )

    top_k = request.top_k or settings.top_k_final
    hits = retriever.hybrid_search(request.question, top_k_final=top_k)

    if not hits:
        return QueryResponse(
            answer="I don't have enough information in the indexed documents to answer that.",
            sources=[],
            retrieved_chunks=[],
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    answer = generate_answer(request.question, hits)
    sources = sorted({h["source"] for h in hits})

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"query='{request.question}' latency_ms={latency_ms} sources={sources}")

    log_query(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieved_chunks=[{"chunk_id": h["chunk_id"], "score": h.get("final_score", 0)} for h in hits],
        latency_ms=latency_ms,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=[
            SourceChunk(
                source=h["source"],
                chunk_id=h["chunk_id"],
                text=h["text"][:400],
                score=round(h.get("final_score", h.get("score", 0)), 4),
            )
            for h in hits
        ],
        latency_ms=latency_ms,
    )


@app.get("/logs")
def logs(limit: int = 50):
    """Bonus Option A - inspect recent query logs (question, chunks, latency)."""
    return fetch_recent_logs(limit)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", documents_indexed=vector_store.count())
