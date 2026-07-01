"""
Centralized configuration loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Storage
    chroma_persist_dir: str = "./storage/chroma"
    bm25_index_path: str = "./storage/bm25_index.pkl"
    docs_dir: str = "./data/docs"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 4
    rrf_k: int = 60


settings = Settings()
