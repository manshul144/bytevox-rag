                           ByteVox RAG System Architecture

                                      Users
                          (Web / API / Streamlit Dashboard)
                                          │
                                          ▼
                                 FastAPI REST API
                           (/query, /ingest, /health, /logs)
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 │                        │                        │
                 ▼                        ▼                        ▼
          Document Ingestion        Query Processing         Observability
                 │                        │                        │
                 ▼                        ▼                        ▼
        PDF / TXT / MD Loader      User Question           SQLite Query Logs
                 │                        │                (Queries, Latency,
                 ▼                        ▼                 Sources, Answers)
             Chunking              Query Embedding
       (800 chars, 120 overlap)          │
                 │                        ▼
                 ▼               ┌─────────────────────┐
         Sentence Transformers   │  Hybrid Retrieval   │
        (all-MiniLM-L6-v2)       └─────────────────────┘
                 │                 │               │
                 ▼                 ▼               ▼
            ChromaDB          Dense Search     BM25 Search
         (Vector Store)        (Semantic)      (Keyword)
                 │                 │               │
                 └─────────────────┴───────────────┘
                               │
                               ▼
                  Reciprocal Rank Fusion (RRF)
                               │
                               ▼
                  Lightweight Lexical Reranking
                               │
                               ▼
                     Top-K Relevant Chunks
                               │
                               ▼
                Groq (Llama 3.1) / Anthropic LLM
                               │
                               ▼
                     Grounded Answer Generation
                               │
                               ▼
                    JSON Response with Sources
                               │
                               ▼
                        Streamlit Dashboard