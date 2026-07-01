# ByteVox RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that ingests enterprise documents, performs hybrid retrieval using dense semantic search and sparse keyword search, and generates grounded answers with source attribution through Large Language Models.

---

## 🛠️ Tech Stack & Key Design Choices

### FastAPI
Chosen for building a high-performance REST API with automatic request validation, interactive Swagger documentation, and asynchronous request handling.

### ChromaDB
Used as the persistent vector database for storing semantic embeddings and performing efficient cosine similarity search.

### Sentence Transformers (`all-MiniLM-L6-v2`)
Generates lightweight 384-dimensional embeddings that provide fast semantic retrieval while maintaining high retrieval quality.

### Rank-BM25
Implements sparse lexical retrieval to accurately capture exact keyword matches, technical terms, and domain-specific queries.

### Hybrid Retrieval (Reciprocal Rank Fusion)
Combines dense vector search from ChromaDB with BM25 keyword search using Reciprocal Rank Fusion (RRF), improving both semantic understanding and exact-match retrieval.

### Groq / Anthropic LLM
Supports Groq (Llama 3.1) as the default inference provider with Anthropic as an alternative. The LLM generates answers strictly grounded in the retrieved document context to minimize hallucinations.

### SQLite
Stores query logs, retrieved chunks, and response latency for observability and evaluation.

---

# 🚀 Getting Started

## 1. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Update your API credentials:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Ingest Documents

Place supported documents (`.txt`, `.md`, `.pdf`) inside:

```
data/docs/
```

Then build the vector and BM25 indexes:

```bash
python scripts/ingest.py
```

or via the API:

```
POST /ingest
```

---

## 4. Start the API Server

```bash
uvicorn app.main:app --reload
```

Swagger Documentation:

```
http://localhost:8000/docs
```

---

## 5. Launch the Evaluation Dashboard

```bash
streamlit run dashboard.py
```

The dashboard allows you to:

- Ask questions over indexed documents
- View retrieved source chunks
- Inspect retrieval scores
- Monitor response latency
- Review recent query history

---

## 📊 Evaluation & System Highlights

The system combines semantic and lexical retrieval to improve answer quality across diverse enterprise documentation.

**Key capabilities include:**

- Hybrid Retrieval using ChromaDB + BM25
- Reciprocal Rank Fusion (RRF) for result fusion
- Source-grounded answer generation
- Automatic document chunking and indexing
- Persistent vector storage
- Query observability with SQLite
- Interactive evaluation dashboard

To reduce hallucinations, the LLM is instructed to answer **only from the retrieved document context**. If the required information is not present in the indexed documents, the system responds that it does not have sufficient information rather than generating unsupported answers.