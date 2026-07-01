# ByteVox RAG System — Technical Assignment Submission

**Candidate:** Manshul  
**Stack:** FastAPI · ChromaDB · BM25 (rank-bm25) · sentence-transformers · Groq / Anthropic · SQLite observability

---

## What's Implemented

| Part | Status |
|---|---|
| Part 1 — RAG API (ingest / query / health) | ✅ Complete |
| Part 2 — Design Decisions | ✅ `docs/DESIGN_DECISIONS.md` |
| Part 3 — Production Architecture | ✅ `docs/PRODUCTION_ARCHITECTURE.md` + `docs/PRODUCTION_DIAGRAM.md` |
| Part 4 — Reflection | ✅ `docs/REFLECTION.md` |
| Bonus A — Observability (SQLite query logs) | ✅ `/logs` endpoint |
| Bonus B — Containerization | ✅ `Dockerfile` + `docker-compose.yml` |
| Bonus C — Hybrid Retrieval (BM25 + Dense + RRF) | ✅ Core retrieval layer |

---

## Quick Start (local, no Docker)

### 1. Clone and install dependencies

```bash
git clone <your-repo>
cd bytevox-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key.
# GROQ_API_KEY is recommended (fast, generous free tier at console.groq.com)
```

### 3. Add your documents (or use the included samples)

Sample documents are already in `data/docs/`:
- `product_overview.md` — product description
- `pricing.md` — pricing plans
- `faq.txt` — frequently asked questions
- `security_and_compliance.pdf` — compliance policy

To add your own: drop any `.pdf`, `.md`, `.markdown`, or `.txt` file into `data/docs/`.

### 4. Ingest documents

```bash
python scripts/ingest.py
```

This embeds and indexes all documents into ChromaDB (stored in `storage/chroma/`)
and builds the BM25 sparse index (stored in `storage/bm25_index.pkl`).

### 5. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

Docs available at: http://localhost:8000/docs (Swagger UI)

### 6. Run a query

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What pricing plan includes API access?"}' | python3 -m json.tool
```

### 7. Run the evaluation suite

```bash
python scripts/evaluate.py
```

---

## Quick Start (Docker)

```bash
cp .env.example .env  # fill in your API key

# Step 1: ingest documents
docker compose --profile ingest up ingest

# Step 2: start the API (keeps running)
docker compose up api
```

---

## API Reference

### `POST /query`
Answer a question using the indexed document corpus.

**Request:**
```json
{ "question": "What pricing plan includes API access?", "top_k": 4 }
```

**Response:**
```json
{
  "answer": "API access is available on the Growth plan ($299/month) and above...",
  "sources": ["pricing.md"],
  "retrieved_chunks": [
    {
      "source": "pricing.md",
      "chunk_id": "pricing.md::1",
      "text": "...",
      "score": 0.0312
    }
  ],
  "latency_ms": 743.2
}
```

### `POST /ingest`
(Re)index all documents in `DOCS_DIR`. Safe to call multiple times — resets the index each time.

**Response:**
```json
{
  "documents_processed": 4,
  "chunks_indexed": 15,
  "files": ["faq.txt", "pricing.md", "product_overview.md", "security_and_compliance.pdf"]
}
```

### `GET /health`
```json
{ "status": "ok", "documents_indexed": 15 }
```

### `GET /logs?limit=50`
Returns the 50 most recent query log entries from SQLite, including
retrieved chunk IDs, answer, latency, and timestamp.

---

## Project Structure

```
bytevox-rag/
├── app/
│   ├── config.py         # Pydantic settings (reads .env)
│   ├── ingestion.py      # PDF / MD / TXT document loading
│   ├── chunking.py       # Paragraph-aware recursive chunker
│   ├── vectorstore.py    # ChromaDB wrapper (dense search)
│   ├── retrieval.py      # BM25 sparse search + RRF hybrid fusion
│   ├── llm.py            # Groq / Anthropic generation
│   ├── observability.py  # SQLite query logging (Bonus A)
│   ├── schemas.py        # Pydantic request/response models
│   └── main.py           # FastAPI app (/query /ingest /health /logs)
│
├── data/docs/            # Drop your documents here
│   ├── product_overview.md
│   ├── pricing.md
│   ├── faq.txt
│   └── security_and_compliance.pdf
│
├── scripts/
│   ├── ingest.py         # CLI ingestion (builds Chroma + BM25 indexes)
│   └── evaluate.py       # Benchmark evaluation (7 questions, hit rate)
│
├── docs/
│   ├── DESIGN_DECISIONS.md       # Part 2
│   ├── PRODUCTION_ARCHITECTURE.md # Part 3 (explanation)
│   ├── PRODUCTION_DIAGRAM.md      # Part 3 (Mermaid diagram)
│   └── REFLECTION.md             # Part 4
│
├── storage/              # Created at runtime (gitignored)
│   ├── chroma/           # ChromaDB persistent files
│   ├── bm25_index.pkl    # Pickled BM25 index
│   └── query_logs.db     # SQLite observability logs
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Retrieval Design (summary)

**Hybrid Search via Reciprocal Rank Fusion:**

1. Dense search — question embedded with `all-MiniLM-L6-v2`, cosine ANN
   search over ChromaDB HNSW index.
2. Sparse search — BM25 (Okapi BM25) over tokenized chunk corpus.
3. RRF fusion — `score(d) = Σ 1/(k + rank)` across both ranked lists,
   combining complementary strengths (semantics vs. exact keyword match)
   without requiring score-scale normalization.
4. Lexical-overlap boost — lightweight re-ranking signal applied post-RRF,
   a cost-effective stand-in for a full cross-encoder reranker.

See `docs/DESIGN_DECISIONS.md` for full rationale and trade-offs.

---

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `anthropic` |
| `GROQ_API_KEY` | — | Get from console.groq.com (free) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | |
| `ANTHROPIC_API_KEY` | — | Alternative provider |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Downloaded once from HuggingFace |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap characters between consecutive chunks |
| `TOP_K_FINAL` | `4` | Chunks fed to LLM as context |
