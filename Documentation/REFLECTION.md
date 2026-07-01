Reflection 
1. Most Challenging Engineering Decision

The most significant engineering decision was implementing a hybrid retrieval pipeline by combining dense semantic retrieval with BM25-based keyword retrieval using Reciprocal Rank Fusion (RRF). Dense retrieval performs well for semantic similarity but can miss exact keywords, while BM25 excels at exact matches but lacks semantic understanding. Designing a retrieval strategy that effectively leveraged the strengths of both approaches required careful consideration.

Implementing RRF provided a simple yet effective way to merge the ranked results from both retrieval methods without requiring score normalization. Additionally, introducing a lightweight lexical overlap boost improved the final ranking quality while keeping the retrieval pipeline computationally efficient.

2. Improvements with Additional Development Time

Given additional time, I would focus on enhancing retrieval quality and production readiness.

The first improvement would be integrating a Cross-Encoder reranker (such as BAAI's bge-reranker-large) to replace the current lightweight lexical reranking. This would provide more accurate relevance ranking before passing context to the language model.

I would also implement parent-child chunking, where smaller chunks are indexed for retrieval while larger parent chunks are supplied to the LLM. This would improve contextual continuity and reduce the chances of fragmented answers.

From a production perspective, I would add authentication, Redis-based response caching, streaming responses, and CI/CD deployment pipelines to improve scalability and operational efficiency.

3. Technology Area I Want to Explore Further

This project strengthened my understanding of Retrieval-Augmented Generation, but I would like to explore distributed vector databases and large-scale retrieval systems in greater depth.

Areas of interest include:

Distributed vector indexing techniques
HNSW optimization
Vector compression methods such as Product Quantization (PQ)
Advanced reranking architectures
Large-scale retrieval evaluation
Multi-agent retrieval workflows

I am particularly interested in understanding how production RAG systems efficiently handle millions of documents while maintaining low latency and high retrieval accuracy.

4. How AI Tools Assisted Development

AI-assisted development tools significantly improved development productivity throughout the project.

They were particularly helpful for:

Understanding unfamiliar library documentation
Resolving dependency and package compatibility issues
Generating initial FastAPI boilerplate
Troubleshooting integration problems
Exploring alternative implementation approaches

Rather than replacing the engineering process, these tools accelerated routine development tasks, allowing me to spend more time designing the retrieval pipeline, implementing hybrid search, optimizing chunking strategies, and validating the overall RAG workflow.

Throughout the implementation, I verified generated code, adapted it to the project requirements, and ensured that the final solution aligned with the assignment objectives and my own design decisions.