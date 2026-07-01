Engineering Design Document: ByteVox RAG System
1. Vector Database Selection: ChromaDB

For this implementation, ChromaDB was selected as the persistent vector database for semantic retrieval over alternatives such as FAISS, Qdrant, and PostgreSQL with pgvector.

Alternatives Considered & Trade-offs
FAISS

FAISS offers extremely fast approximate nearest neighbor (ANN) search and is well suited for research workloads. However, it lacks built-in persistent storage and metadata management, requiring additional infrastructure for production-ready document retrieval.

Qdrant

Qdrant provides excellent vector search performance with advanced filtering capabilities and distributed deployment options. While highly scalable, it introduces additional deployment complexity for a relatively small document corpus and requires running a dedicated vector database service.

PostgreSQL + pgvector

Using pgvector enables vector search within an existing relational database. Although convenient for applications already dependent on PostgreSQL, its vector search performance and scalability are generally lower than specialized vector databases for retrieval-heavy workloads.

Selected Choice Justification

ChromaDB was chosen because it provides:

Persistent local vector storage
Simple integration with Sentence Transformers
Native metadata support
Efficient cosine similarity search
Minimal deployment overhead

These characteristics make ChromaDB well suited for lightweight RAG systems while keeping the architecture simple and reproducible.

2. Embedding Model Selection: all-MiniLM-L6-v2

The system uses the all-MiniLM-L6-v2 embedding model through the Sentence Transformers library.

Strengths
Lightweight 384-dimensional embeddings
Fast embedding generation
Low memory consumption
Strong semantic similarity performance
Well suited for technical documentation
Limitations
Smaller embedding dimensions than larger transformer models
Limited ability to capture very long-range document dependencies in a single embedding
Mitigation

To preserve contextual continuity, documents are divided into overlapping chunks before embedding. This ensures that information spanning chunk boundaries remains retrievable while keeping each chunk within the embedding model's effective input length.

3. Chunking Strategy
Chunk Size

800 characters

Chunk Overlap

120 characters

Design Rationale

The chosen chunk size provides a balance between semantic completeness and retrieval precision.

Larger chunks preserve sufficient context for the language model.
Smaller chunks improve retrieval accuracy by reducing irrelevant information.
A 120-character overlap ensures important entities and concepts are not split across chunk boundaries.

This strategy improves both dense retrieval quality and BM25 keyword matching while minimizing duplicate information.

4. Retrieval Strategy

The retrieval layer combines semantic search with lexical search to improve answer quality.

Dense Retrieval

Semantic search is performed using ChromaDB and cosine similarity over Sentence Transformer embeddings.

Strengths:

Captures semantic similarity
Handles paraphrases effectively
Robust to wording differences

Limitations:

Can miss exact keywords, identifiers, or technical terminology.
Sparse Retrieval (BM25)

BM25 retrieves documents based on keyword frequency and inverse document frequency.

Strengths:

Excellent for exact keyword matching
Performs well for technical terms, version numbers, API names, and configuration parameters

Limitations:

Does not understand semantic similarity.
Hybrid Retrieval using Reciprocal Rank Fusion (RRF)

Instead of relying solely on semantic or lexical retrieval, the system combines both ranked result lists using Reciprocal Rank Fusion (RRF).

RRF improves retrieval robustness by leveraging the complementary strengths of dense and sparse retrieval without requiring score normalization.

This approach increases recall while maintaining retrieval precision across different query types.

5. Lightweight Re-ranking

After RRF, retrieved chunks receive a small lexical overlap boost based on shared query terms.

Although not as powerful as transformer-based cross-encoder rerankers, this lightweight strategy:

Improves ranking quality
Adds negligible computational overhead
Keeps response latency low

This provides a practical balance between retrieval quality and inference speed.

6. Language Model Selection

The generation layer supports two interchangeable providers:

Groq (Llama 3.1 8B Instant) (default)
Anthropic Claude (optional)

Groq was selected as the default because it provides:

Low inference latency
Competitive response quality
Easy API integration
Suitable free-tier availability for development

The system prompt explicitly instructs the model to answer only using the retrieved document context, reducing hallucinations and ensuring grounded responses.

