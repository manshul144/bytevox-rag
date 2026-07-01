from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's natural language question")
    top_k: Optional[int] = Field(None, description="Override number of chunks to use as context")


class SourceChunk(BaseModel):
    source: str
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_chunks: List[SourceChunk]
    latency_ms: float


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_indexed: int
    files: List[str]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
