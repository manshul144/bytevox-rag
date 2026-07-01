# ByteVox RAG System — Technical Assignment Submission

**Candidate:** Manshul 

**Tech Stack:** FastAPI • ChromaDB • BM25 • Sentence Transformers • Groq / Anthropic • SQLite • Streamlit • Docker

---

# Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** assistant capable of ingesting enterprise documentation, indexing it using both dense and sparse retrieval techniques, and answering user questions grounded entirely in the indexed documents.

The system combines:

- Dense semantic retrieval using Sentence Transformers + ChromaDB
- Sparse keyword retrieval using BM25
- Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
- LLM-based answer generation using Groq (default) or Anthropic
- SQLite-based observability
- Streamlit evaluation dashboard
- Dockerized deployment

The repository uses the official **Nexus AI** knowledge base provided with the assignment.

---

# Features

## Core Features

- Document ingestion (.txt, .md, .pdf)
- Automatic chunking
- Dense embeddings using Sentence Transformers
- Persistent vector storage using ChromaDB
- BM25 sparse retrieval
- Hybrid Retrieval (Dense + Sparse)
- Reciprocal Rank Fusion (RRF)
- FastAPI REST API
- Source attribution
- Configurable LLM provider
- Persistent vector storage

---

## Bonus Features

### Bonus A — Observability

- SQLite query logging
- Response latency tracking
- Retrieved chunk logging
- `/logs` endpoint

### Bonus B — Containerization

- Dockerfile
- docker-compose support

### Bonus C — Advanced Retrieval

- Dense Retrieval
- BM25 Retrieval
- Reciprocal Rank Fusion
- Lightweight lexical reranking

### Bonus D — Evaluation Dashboard

Interactive Streamlit dashboard displaying:

- Ask questions
- Generated answers
- Retrieved chunks
- Sources
- Retrieval scores
- Query latency
- Recent query history

---

# Project Structure

```text
bytevox-rag/
│
├── app/
│   ├── chunking.py
│   ├── config.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── main.py
│   ├── observability.py
│   ├── retrieval.py
│   ├── schemas.py
│   └── vectorstore.py
│
├── data/
│   └── docs/
│       ├── 01_nexus_ai_overview.txt
│       ├── 02_nexus_api_reference.txt
│       ├── 03_nexus_architecture_internals.txt
│       ├── 04_nexus_troubleshooting_guide.txt
│       ├── 05_nexus_changelog.txt
│       ├── 06_nexus_sdk_guide.txt
│       ├── 07_nexus_security_compliance.txt
│       └── 08_nexus_ml_best_practices.txt
│
├── scripts/
│   ├── ingest.py
│   └── evaluate.py
│
├── storage/
│
├── dashboard.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/manshul144/bytevox-rag.git
cd bytevox-rag
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Copy

```bash
cp .env.example .env
```

Update your `.env`

```env
LLM_PROVIDER=groq

GROQ_API_KEY=your_groq_api_key_here

GROQ_MODEL=llama-3.1-8b-instant

EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

# Knowledge Base

The repository contains the official assignment documents:

```
data/docs/
```

```
01_nexus_ai_overview.txt
02_nexus_api_reference.txt
03_nexus_architecture_internals.txt
04_nexus_troubleshooting_guide.txt
05_nexus_changelog.txt
06_nexus_sdk_guide.txt
07_nexus_security_compliance.txt
08_nexus_ml_best_practices.txt
```

To use your own documents, simply replace or add supported `.txt`, `.md`, or `.pdf` files in `data/docs/` and run the ingestion process again.

---

# Ingest Documents

Run:

```bash
python scripts/ingest.py
```

or

```
POST /ingest
```

Example response:

```json
{
  "documents_processed": 8,
  "chunks_indexed": 156,
  "files": [
    "01_nexus_ai_overview.txt",
    "02_nexus_api_reference.txt",
    "03_nexus_architecture_internals.txt",
    "04_nexus_troubleshooting_guide.txt",
    "05_nexus_changelog.txt",
    "06_nexus_sdk_guide.txt",
    "07_nexus_security_compliance.txt",
    "08_nexus_ml_best_practices.txt"
  ]
}
```

---

# Start API

```bash
uvicorn app.main:app --reload
```

Swagger UI

```
http://localhost:8000/docs
```

---

# API Endpoints

## POST /query

Example Request

```json
{
  "question": "What is Nexus AI?",
  "top_k": 4
}
```

Example Response

```json
{
  "answer": "Nexus AI is a cloud-native machine learning platform developed by Meridian Labs, designed to accelerate the deployment and management of production ML workloads.",
  "sources": [
    "01_nexus_ai_overview.txt"
  ],
  "retrieved_chunks": [
    {
      "source": "01_nexus_ai_overview.txt",
      "chunk_id": "01_nexus_ai_overview.txt::1",
      "score": 0.0375
    }
  ],
  "latency_ms": 1382.83
}
```

---

## POST /ingest

Indexes all documents in `data/docs`.

---

## GET /health

Example

```json
{
  "status":"ok",
  "documents_indexed":156
}
```

---

## GET /logs

Returns recent query logs including

- User Question
- Generated Answer
- Retrieved Chunks
- Latency
- Timestamp

---

# Streamlit Dashboard

Launch

```bash
streamlit run dashboard.py
```

Dashboard Features

- Ask questions
- View generated answers
- View retrieved chunks
- Retrieval scores
- Sources used
- Recent queries
- Total queries
- Average latency

---

# Docker Deployment

## Build

```bash
docker compose build
```

## Ingest

```bash
docker compose --profile ingest up ingest
```

## Start API

```bash
docker compose up api
```

---

# Retrieval Pipeline

```
User Question
        │
        ▼
Sentence Transformer Embedding
        │
        ▼
Dense Search (ChromaDB)

        +

Sparse Search (BM25)

        │
        ▼

Reciprocal Rank Fusion (RRF)

        │
        ▼

Lexical Reranking

        │
        ▼

Top-K Chunks

        │
        ▼

Groq / Anthropic LLM

        │
        ▼

Grounded Answer
```

---

# Design Decisions

The retrieval system combines dense semantic retrieval with sparse lexical retrieval.

**Dense Retrieval**

- Sentence Transformers
- ChromaDB
- Cosine Similarity

**Sparse Retrieval**

- BM25 (Okapi)

**Fusion**

- Reciprocal Rank Fusion (RRF)

**Reranking**

- Lightweight lexical overlap boost

This hybrid approach improves recall for both semantic and exact keyword queries.

---

# Environment Variables

| Variable | Default |
|----------|----------|
| LLM_PROVIDER | groq |
| GROQ_MODEL | llama-3.1-8b-instant |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 |
| CHUNK_SIZE | 800 |
| CHUNK_OVERLAP | 120 |
| TOP_K_DENSE | 8 |
| TOP_K_SPARSE | 8 |
| TOP_K_FINAL | 4 |
| RRF_K | 60 |

---

# Documentation

The following documents accompany this submission:

- Design Decisions
- Production Architecture
- Architecture Diagram
- Reflection Write-up

These are submitted separately as requested.

---

# Future Improvements

- Cross-Encoder reranking
- Authentication & Authorization
- Streaming responses
- Redis caching
- CI/CD pipeline
- Kubernetes deployment
- Multi-user support
- Feedback-driven retrieval optimization

---

# Assignment Completion

| Requirement | Status |
|------------|---------|
| Part 1 – RAG API | ✅ |
| Part 2 – Design Decisions | ✅ |
| Part 3 – Production Architecture | ✅ |
| Part 4 – Reflection | ✅ |
| Bonus A – Observability | ✅ |
| Bonus B – Docker | ✅ |
| Bonus C – Hybrid Retrieval | ✅ |
| Bonus D – Dashboard | ✅ |

---

# Author

**Manshul**

B.Tech CSE (AI & ML)

FastAPI • Machine Learning • NLP • RAG • LLM Applications