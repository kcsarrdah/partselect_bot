"""
RAG Service
Orchestrates the full RAG workflow: Retrieve → Augment → Generate
Combines vector search, prompt building, and LLM generation.
"""

import time
from typing import Dict, Any, List, Optional
from services.ingestion_pipeline import IngestionPipeline
from services.llm_service import LLMService
from prompts import build_rag_prompt, PARTSELECT_SYSTEM_PROMPT
from utils.logger import setup_logger, log_pipeline_step, log_success, log_error, log_warning, log_metric
from services.query_cache import QueryCache
# Setup logger
logger = setup_logger(__name__)


class RAGService:
    """Service for handling RAG queries end-to-end."""
    
    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        llm_service: LLMService,
        system_prompt: str = PARTSELECT_SYSTEM_PROMPT,
        default_k: int = 5,
        enable_cache: bool = True
    ):
        """
        Initialize RAG service with dependencies.
        
        Args:
            ingestion_pipeline: For vector search
            llm_service: For LLM generation
            system_prompt: Default system prompt
            default_k: Default number of docs to retrieve
        """
        self.pipeline = ingestion_pipeline
        self.llm = llm_service
        self.system_prompt = system_prompt
        self.default_k = default_k

            # Initialize query cache
        self.cache_enabled = enable_cache
        if self.cache_enabled:
            self.cache = QueryCache()
            log_success(logger, "Query cache enabled")
        else:
            self.cache = None
            log_warning(logger, "Query cache disabled")
        
        # Stats tracking
        self.queries_processed = 0
        self.total_response_time = 0.0
        
        log_success(logger, "RAG Service initialized")
    
    def query(
        self,
        user_query: str,
        k: Optional[int] = None,
        filter_type: Optional[str] = None,
        include_examples: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query end-to-end with RAG.
        
        Args:
            user_query: User's question
            k: Number of documents to retrieve (default: 5)
            filter_type: Filter by doc type ("part", "blog", "repair")
            include_examples: Include few-shot examples in prompt
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        k = k or self.default_k
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"RAG Query: {user_query}")
            logger.info(f"{'='*60}")
            if self.cache_enabled:
                logger.info("🔍 Checking cache...")
                cached = self.cache.get(user_query)
                if cached:
                    # Update stats
                    self.queries_processed += 1
                    response_time = time.time() - start_time
                    
                    tokens_saved = cached['metadata'].get('tokens_used', 0)
                    log_success(logger, f"Cache hit! (saved {tokens_saved} tokens, {cached.get('cache_type', 'unknown')} match)")
                    
                    # Add response time for cache hit
                    cached['metadata']['cache_response_time_seconds'] = round(response_time, 3)
                    
                    logger.info(f"\n{'='*60}")
                    log_success(logger, f"Query Complete from Cache ({response_time:.3f}s)")
                    logger.info(f"{'='*60}\n")
                    
                    return cached
            
            # Step 1: Retrieve context from vector store
            log_pipeline_step(logger, 1, "Retrieving context")
            logger.info(f"🔍 Searching for: '{user_query}' (k={k})")
            context_docs = self._retrieve_context(user_query, k, filter_type)
            
            if not context_docs:
                return self._handle_no_context(user_query)
            
            log_success(logger, f"Retrieved {len(context_docs)} documents")
            
            # Step 2: Build prompt with context
            log_pipeline_step(logger, 2, "Building prompt")
            prompt = self._build_prompt(user_query, context_docs, include_examples)
            log_metric(logger, "Prompt size", f"{len(prompt)} chars")
            
            # Step 3: Generate LLM response
            log_pipeline_step(logger, 3, "Generating response")
            llm_result = self.llm.generate(prompt)
            
            if llm_result['status_code'] != 200:
                return self._handle_llm_error(llm_result)
            
            log_success(logger, "Response generated")
            
            # Step 4: Extract and format sources
            log_pipeline_step(logger, 4, "Extracting sources")
            sources = self._extract_sources(context_docs)
            log_success(logger, f"Extracted {len(sources)} sources")
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update stats
            self.queries_processed += 1
            self.total_response_time += response_time
            
            # Build final response
            result = {
                "status_code": 200,
                "status": "success",
                "query": user_query,
                "answer": llm_result['response'],
                "sources": sources,
                "metadata": {
                    "retrieved_docs": len(context_docs),
                    "tokens_used": llm_result['usage']['total_tokens'],
                    "response_time_seconds": round(response_time, 2),
                    "model": llm_result['model'],
                    "filter_type": filter_type
                }
            }
            # CACHE THE RESULT
            if self.cache_enabled:
                self.cache.set(user_query, result)
            
            logger.info(f"\n{'='*60}")
            log_success(logger, f"Query Complete ({response_time:.2f}s, {llm_result['usage']['total_tokens']} tokens)")
            logger.info(f"{'='*60}\n")
            
            return result
        
        except Exception as e:
            log_error(logger, f"RAG Query failed: {e}")
            return {
                "status_code": 500,
                "status": "error",
                "query": user_query,
                "message": str(e),
                "error_type": type(e).__name__
            }
    
    def _retrieve_context(
        self,
        query: str,
        k: int,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            query: Search query
            k: Number of documents
            filter_type: Optional doc type filter
        
        Returns:
            List of retrieved documents
        """
        try:
            if filter_type:
                result = self.pipeline.search_by_type(query, doc_type=filter_type, k=k)
            else:
                result = self.pipeline.search(query, k=k)
            
            if result['status_code'] != 200:
                log_warning(logger, f"Search returned status {result['status_code']}")
                return []
            
            return result.get('results', [])
        
        except Exception as e:
            log_error(logger, f"Retrieval error: {e}")
            return []
    
    def _build_prompt(
        self,
        user_query: str,
        context_docs: List[Dict[str, Any]],
        include_examples: bool = False
    ) -> str:
        """
        Build complete RAG prompt.
        
        Args:
            user_query: User's question
            context_docs: Retrieved documents
            include_examples: Include few-shot examples
        
        Returns:
            Complete prompt string
        """
        return build_rag_prompt(
            user_query=user_query,
            context_docs=context_docs,
            system_prompt=self.system_prompt,
            include_examples=include_examples
        )
    
    def _extract_sources(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract clean source citations from context docs.
        
        Args:
            context_docs: Retrieved documents with metadata
        
        Returns:
            List of source citations
        """
        sources = []
        
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('type', 'unknown')
            
            source = {
                "type": doc_type,
                "score": doc.get('score', 0.0)
            }
            
            # Add type-specific fields
            if doc_type == 'part':
                source.update({
                    "part_id": metadata.get('part_id'),
                    "part_name": metadata.get('part_name'),
                    "brand": metadata.get('brand'),
                    "price": metadata.get('price'),
                    "product_url": metadata.get('product_url'),
                    "install_video_url": metadata.get('install_video_url')
                })
            elif doc_type == 'repair':
                source.update({
                    "symptom": metadata.get('symptom'),
                    "appliance": metadata.get('appliance'),
                    "difficulty": metadata.get('difficulty'),
                    "video_url": metadata.get('video_url'),
                    "detail_url": metadata.get('detail_url')
                })
            elif doc_type == 'blog':
                source.update({
                    "title": metadata.get('title'),
                    "url": metadata.get('url')
                })
            
            sources.append(source)
        
        return sources
    
    def _handle_no_context(self, user_query: str) -> Dict[str, Any]:
        """Handle case where no relevant context is found."""
        log_warning(logger, "No relevant context found")
        
        return {
            "status_code": 404,
            "status": "no_context",
            "query": user_query,
            "answer": "I couldn't find relevant information in our database to answer your question. Please try rephrasing or contact PartSelect support for assistance.",
            "sources": [],
            "metadata": {
                "retrieved_docs": 0,
                "reason": "No relevant documents found"
            }
        }
    
    def _handle_llm_error(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM generation errors."""
        log_error(logger, f"LLM error: {llm_result.get('message')}")
        
        return {
            "status_code": llm_result['status_code'],
            "status": "llm_error",
            "message": llm_result.get('message', 'LLM generation failed'),
            "error_type": llm_result.get('error_type')
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG service statistics.
        
        Returns:
            Dictionary with service stats
        """
        avg_response_time = (
            self.total_response_time / self.queries_processed
            if self.queries_processed > 0
            else 0.0
        )
        
        # Get vector store stats
        vector_stats = self.pipeline.get_status()
        
        stats = {
            "queries_processed": self.queries_processed,
            "average_response_time": round(avg_response_time, 2),
            "vector_store_docs": vector_stats.get('total_documents', 0),
            "llm_model": self.llm.model,
            "default_k": self.default_k
        }
        
        # 🆕 ADD CACHE STATS
        if self.cache_enabled and self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if RAG system is healthy.
        
        Returns:
            Health status dictionary
        """
        try:
            # Check vector store
            vector_status = self.pipeline.health_check()
            if vector_status['status_code'] != 200:
                return {
                    "status_code": 503,
                    "status": "unhealthy",
                    "reason": "Vector store unavailable"
                }
            
            # Check LLM
            llm_info = self.llm.get_model_info()
            if not llm_info.get('api_key_set'):
                return {
                    "status_code": 503,
                    "status": "unhealthy",
                    "reason": "LLM API key not set"
                }
            
            return {
                "status_code": 200,
                "status": "healthy",
                "vector_docs": vector_status.get('total_documents', 0),
                "llm_model": llm_info.get('model')
            }
        
        except Exception as e:
            return {
                "status_code": 500,
                "status": "error",
                "message": str(e)
            }