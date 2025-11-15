"""
Pydantic schemas for API request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, max_length=500, description="User's question")
    k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    filter_type: Optional[str] = Field(None, description="Filter by type: 'part', 'blog', or 'repair'")
    include_examples: Optional[bool] = Field(False, description="Include few-shot examples in prompt")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I fix my ice maker?",
                "k": 5,
                "filter_type": None,
                "include_examples": False
            }
        }


class Source(BaseModel):
    """Source citation model."""
    type: str
    score: float
    part_id: Optional[str] = None
    part_name: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[str] = None
    symptom: Optional[str] = None
    appliance: Optional[str] = None
    difficulty: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class ChatMetadata(BaseModel):
    """Metadata about the chat response."""
    retrieved_docs: int
    tokens_used: int
    response_time_seconds: float
    model: str
    filter_type: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    status_code: int
    status: str
    query: str
    answer: str
    sources: List[Source]
    metadata: ChatMetadata
    
    class Config:
        json_schema_extra = {
            "example": {
                "status_code": 200,
                "status": "success",
                "query": "How do I fix my ice maker?",
                "answer": "Based on your question...",
                "sources": [
                    {
                        "type": "part",
                        "score": 0.92,
                        "part_id": "PS11752778",
                        "part_name": "Ice Maker Assembly",
                        "brand": "GE",
                        "price": "$89.99"
                    }
                ],
                "metadata": {
                    "retrieved_docs": 5,
                    "tokens_used": 450,
                    "response_time_seconds": 2.5,
                    "model": "google/gemma-3-27b-it:free",
                    "filter_type": None
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status_code: int
    status: str
    vector_docs: Optional[int] = None
    llm_model: Optional[str] = None
    reason: Optional[str] = None


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    queries_processed: int
    average_response_time: float
    vector_store_docs: int
    llm_model: str
    default_k: int


class ErrorResponse(BaseModel):
    """Response model for errors."""
    status_code: int
    status: str
    message: str
    error_type: Optional[str] = None