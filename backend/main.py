"""
FastAPI Backend for PartSelect Chat Agent
Exposes RAG pipeline as REST API endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from services.ingestion_pipeline import IngestionPipeline
from services.llm_service import LLMService
from services.rag_service import RAGService
from models.schemas import (
    ChatRequest, ChatResponse, HealthResponse, 
    StatsResponse, ErrorResponse
)
from utils.logger import setup_logger, log_success, log_error, log_warning

# Setup logger
logger = setup_logger(__name__)

# Global service instances
rag_service = None
ingestion_pipeline = None
llm_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic for the FastAPI app.
    """
    global rag_service, ingestion_pipeline, llm_service
    
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting PartSelect Chat Agent API")
    logger.info("="*60 + "\n")
    
    try:
        # Initialize services
        logger.info("📦 Initializing services...")
        
        # Initialize ingestion pipeline
        ingestion_pipeline = IngestionPipeline(
            collection_name="partselect_production",
            persist_directory="data/vector_store"
        )
        
        # Check if vector store is empty and needs ingestion
        status = ingestion_pipeline.get_status()
        if status['total_documents'] == 0:
            log_warning(logger, "Vector store is empty. Running ingestion pipeline...")
            result = ingestion_pipeline.run_pipeline(data_dir="data/raw")
            if result['status_code'] != 200:
                raise Exception(f"Ingestion failed: {result['message']}")
            log_success(logger, f"Ingestion complete: {result['total_in_collection']} documents loaded")
        else:
            log_success(logger, f"Vector store loaded: {status['total_documents']} documents")
        
        # Initialize LLM service
        llm_service = LLMService(model="google/gemma-3-27b-it:free")
        
        # Initialize RAG service
        rag_service = RAGService(
            ingestion_pipeline=ingestion_pipeline,
            llm_service=llm_service
        )
        
        logger.info("\n" + "="*60)
        log_success(logger, "All services initialized successfully!")
        logger.info("📍 API running at: http://localhost:8000")
        logger.info("📚 Docs available at: http://localhost:8000/docs")
        logger.info("="*60 + "\n")
        
        yield  # App runs here
        
    except Exception as e:
        log_error(logger, f"Startup failed: {e}")
        raise
    
    finally:
        # Cleanup on shutdown
        logger.info("\n🛑 Shutting down PartSelect Chat Agent API\n")


# Initialize FastAPI app
app = FastAPI(
    title="PartSelect Chat Agent API",
    description="RAG-powered chat agent for refrigerator and dishwasher parts",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for custom errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "status_code": 500,
            "status": "error",
            "message": str(exc),
            "error_type": type(exc).__name__
        }
    )


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "PartSelect Chat Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "POST /api/chat",
            "health": "GET /api/health",
            "stats": "GET /api/stats",
            "docs": "GET /docs"
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user queries with RAG.
    
    Args:
        request: ChatRequest with query, k, filter_type, include_examples
    
    Returns:
        ChatResponse with answer, sources, and metadata
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized"
        )
    
    try:
        logger.info(f"\n📨 New chat request: {request.query[:50]}...")
        start_time = time.time()
        
        # Process query
        result = rag_service.query(
            user_query=request.query,
            k=request.k,
            filter_type=request.filter_type,
            include_examples=request.include_examples
        )
        
        # Handle errors
        if result['status_code'] != 200:
            if result['status_code'] == 404:
                raise HTTPException(status_code=404, detail=result.get('message'))
            else:
                raise HTTPException(status_code=500, detail=result.get('message'))
        
        elapsed = time.time() - start_time
        log_success(logger, f"Chat request completed in {elapsed:.2f}s\n")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, f"Chat request failed: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint - verifies all services are operational.
    
    Returns:
        HealthResponse with system status
    """
    if rag_service is None:
        return {
            "status_code": 503,
            "status": "unhealthy",
            "reason": "Services not initialized"
        }
    
    try:
        result = rag_service.health_check()
        return result
    except Exception as e:
        return {
            "status_code": 500,
            "status": "error",
            "reason": str(e)
        }


@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    """
    Statistics endpoint - returns usage metrics.
    
    Returns:
        StatsResponse with query stats and system info
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized"
        )
    
    try:
        result = rag_service.get_stats()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)