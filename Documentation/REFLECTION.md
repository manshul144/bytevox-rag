# Reflection

---

# 1. Most Challenging Engineering Decision

The most significant engineering decision in this project was designing a **hybrid retrieval pipeline** by combining **dense semantic retrieval** with **BM25-based lexical retrieval** using **Reciprocal Rank Fusion (RRF)**.

While dense retrieval is effective at understanding semantic meaning and paraphrased queries, it may overlook exact keywords, API names, or technical identifiers. Conversely, BM25 performs exceptionally well for exact keyword matching but lacks semantic understanding. Balancing these complementary approaches required careful consideration to maximize retrieval quality.

Implementing **RRF** provided an efficient and robust mechanism for merging the ranked outputs of both retrieval methods without requiring score normalization. Additionally, a lightweight lexical overlap boost was introduced after fusion to further improve ranking quality while maintaining low computational overhead and fast response times.

---

# 2. Improvements with Additional Development Time

Given additional development time, I would focus on improving both **retrieval accuracy** and **production readiness**.

From a retrieval perspective, I would integrate a **Cross-Encoder reranker** (such as **BAAI's bge-reranker-large**) to replace the current lightweight lexical reranking. This would provide more accurate relevance scoring before passing context to the language model.

I would also implement **parent-child chunking**, where smaller chunks are indexed for precise retrieval while larger parent chunks are supplied to the LLM. This approach would improve contextual continuity and reduce fragmented responses.

For production deployment, I would further enhance the system by introducing:

- **Authentication and authorization**
- **Redis-based response caching**
- **Streaming LLM responses**
- **CI/CD pipelines**
- **Container orchestration using Kubernetes**

These enhancements would improve scalability, reliability, and operational efficiency.

---

# 3. Technology Area I Want to Explore Further

This project significantly strengthened my understanding of **Retrieval-Augmented Generation (RAG)** systems. Moving forward, I would like to explore the internal mechanics of **large-scale vector search systems** and production-grade retrieval infrastructure.

Areas I am particularly interested in include:

- **Distributed vector databases**
- **HNSW graph optimization**
- **Vector compression techniques (Product Quantization)**
- **Transformer-based Cross-Encoder reranking**
- **Large-scale retrieval evaluation**
- **Agentic and multi-step retrieval workflows**

I am especially interested in understanding how enterprise RAG systems efficiently manage millions of documents while maintaining low latency, high retrieval accuracy, and reliable scalability.

---

# 4. How AI Tools Assisted Development

AI-assisted development tools played an important role in accelerating the implementation of this project.

They were particularly valuable for:

- Understanding unfamiliar library documentation
- Resolving dependency and package compatibility issues
- Generating initial FastAPI boilerplate
- Troubleshooting integration problems
- Exploring alternative implementation approaches

Rather than replacing the engineering process, these tools served as productivity accelerators, allowing me to focus more on designing the retrieval pipeline, implementing hybrid search, optimizing chunking strategies, and validating the overall RAG workflow.

Throughout development, I carefully reviewed, adapted, and tested AI-generated suggestions to ensure they aligned with the project requirements and my own design decisions. This combination of AI-assisted development and manual engineering helped produce a reliable, well-structured, and maintainable solution.

---

# Key Takeaways

- Successfully implemented a **hybrid Retrieval-Augmented Generation (RAG)** system using **FastAPI**, **ChromaDB**, **BM25**, and **Groq/Anthropic**.
- Improved retrieval quality by combining **dense semantic search** with **lexical retrieval** through **Reciprocal Rank Fusion (RRF)**.
- Built an end-to-end pipeline covering **document ingestion**, **chunking**, **embedding generation**, **hybrid retrieval**, **LLM-based answer generation**, and **observability**.
- Gained practical experience with **vector databases**, **retrieval pipelines**, **REST API development**, **Docker containerization**, and **production-oriented system design**.
- Developed a stronger understanding of building scalable, document-grounded AI applications while balancing retrieval quality, latency, and maintainability.