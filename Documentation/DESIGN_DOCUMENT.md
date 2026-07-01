# Engineering Design Document: ByteVox RAG System

---

# 1. Vector Database Selection: ChromaDB

For this implementation, **ChromaDB** was selected as the persistent vector database for semantic retrieval over alternatives such as **FAISS**, **Qdrant**, and **PostgreSQL with pgvector**.

## Alternatives Considered & Trade-offs

### FAISS

FAISS offers extremely fast Approximate Nearest Neighbor (ANN) search and is well suited for research workloads. However, it lacks built-in persistent storage and metadata management, requiring additional infrastructure for production-ready document retrieval.

### Qdrant

Qdrant provides excellent vector search performance with advanced filtering capabilities and distributed deployment options. While highly scalable, it introduces additional deployment complexity for a relatively small document corpus and requires running a dedicated vector database service.

### PostgreSQL + pgvector

Using **pgvector** enables vector search within an existing relational database. Although convenient for applications already dependent on PostgreSQL, its vector search performance and scalability are generally lower than specialized vector databases for retrieval-heavy workloads.

## Selected Choice Justification

**ChromaDB** was chosen because it provides:

- Persistent local vector storage
- Simple integration with Sentence Transformers
- Native metadata support
- Efficient cosine similarity search
- Minimal deployment overhead

These characteristics make ChromaDB well suited for lightweight RAG systems while keeping the architecture simple, reproducible, and easy to deploy.

---

# 2. Embedding Model Selection: all-MiniLM-L6-v2

The system uses the **all-MiniLM-L6-v2** embedding model through the **Sentence Transformers** library.

## Strengths

- Lightweight 384-dimensional embeddings
- Fast embedding generation
- Low memory consumption
- Strong semantic similarity performance
- Well suited for technical documentation

## Limitations

- Smaller embedding dimensions than larger transformer models
- Limited ability to capture very long-range document dependencies in a single embedding

## Mitigation

To preserve contextual continuity, documents are divided into overlapping chunks before embedding. This ensures that information spanning chunk boundaries remains retrievable while keeping each chunk within the embedding model's effective input length.

---

# 3. Chunking Strategy

## Chunk Size

**800 characters**

## Chunk Overlap

**120 characters**

## Design Rationale

The selected chunk size provides an effective balance between semantic completeness and retrieval precision.

- Preserves sufficient context for the language model.
- Improves retrieval accuracy by reducing irrelevant information.
- Prevents important entities and concepts from being split across chunk boundaries.
- Enhances both dense retrieval and BM25 keyword matching while minimizing duplicate information.

---

# 4. Retrieval Strategy

The retrieval pipeline combines **semantic search** with **lexical search** to improve answer quality and retrieval robustness.

---

## Dense Retrieval

Semantic retrieval is performed using **ChromaDB** and cosine similarity over Sentence Transformer embeddings.

### Strengths

- Captures semantic similarity
- Handles paraphrased questions effectively
- Robust to wording variations

### Limitations

- Can occasionally miss exact keywords, identifiers, or technical terminology.

---

## Sparse Retrieval (BM25)

BM25 retrieves documents using keyword frequency and inverse document frequency scoring.

### Strengths

- Excellent for exact keyword matching
- Performs well for technical terms, API names, version numbers, and configuration parameters

### Limitations

- Does not capture semantic similarity between related phrases.

---

## Hybrid Retrieval using Reciprocal Rank Fusion (RRF)

Rather than relying solely on semantic or lexical retrieval, the system combines both ranked result lists using **Reciprocal Rank Fusion (RRF)**.

### Benefits

- Leverages the strengths of both dense and sparse retrieval
- Eliminates the need for score normalization
- Improves recall while maintaining retrieval precision
- Produces more robust results across diverse query types

---

# 5. Lightweight Re-ranking

Following RRF, retrieved chunks receive a lightweight lexical overlap boost based on shared query terms.

Although less sophisticated than transformer-based Cross-Encoder rerankers, this strategy offers several advantages:

- Improves final ranking quality
- Adds negligible computational overhead
- Maintains low response latency

This provides a practical balance between retrieval accuracy and inference speed while keeping the system lightweight.

---

# 6. Language Model Selection

The generation layer supports two interchangeable providers:

- **Groq (Llama 3.1 8B Instant)** *(Default)*
- **Anthropic Claude** *(Optional)*

## Why Groq?

Groq was selected as the default provider because it offers:

- Low inference latency
- Competitive response quality
- Simple API integration
- Suitable free-tier availability for development

## Grounded Response Generation

The system prompt explicitly instructs the language model to:

- Answer **only** using the retrieved document context.
- Avoid generating unsupported information.
- Return grounded, source-backed responses.

This significantly reduces hallucinations while ensuring reliable and document-based answers.

---

# Summary

The ByteVox RAG System combines:

- **ChromaDB** for persistent vector storage
- **Sentence Transformers** for semantic embeddings
- **BM25** for sparse lexical retrieval
- **Reciprocal Rank Fusion (RRF)** for hybrid search
- **Groq / Anthropic** for grounded answer generation
- **SQLite** for observability and query logging

This architecture balances retrieval accuracy, response latency, deployment simplicity, and scalability while remaining flexible enough to incorporate advanced reranking techniques and larger embedding models in future iterations.