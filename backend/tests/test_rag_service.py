"""
Unit tests for RAG Service (including cache integration)
"""
import pytest
from services.ingestion_pipeline import IngestionPipeline
from services.llm_service import LLMService
from services.rag_service import RAGService


class TestRAGServiceCache:
    @pytest.fixture
    def rag_service(self):
        """Create RAG service with test dependencies"""
        pipeline = IngestionPipeline(
            collection_name="partselect_test",
            persist_directory="tests/test_data/vector_store_test"
        )
        llm = LLMService(model="google/gemma-3-27b-it:free")
        return RAGService(
            ingestion_pipeline=pipeline,
            llm_service=llm,
            enable_cache=True
        )
    
    def test_cache_enabled_by_default(self, rag_service):
        """TEST: RAG service has cache enabled"""
        assert rag_service.cache_enabled == True
        assert rag_service.cache is not None
    
    def test_first_query_runs_rag_pipeline(self, rag_service):
    # Clear cache to ensure fresh start
        if rag_service.cache_enabled:
            rag_service.cache.clear()
        
        query = "test query unique 12345"
        result = rag_service.query(query, k=3)
        
        assert result['status_code'] == 200
        assert 'cached' not in result or result.get('cached') == False
    
    def test_duplicate_query_uses_cache(self, rag_service):
        """TEST: Duplicate query returns cached result"""
        query = "what is an ice maker"
        
        # First query (cache miss)
        result1 = rag_service.query(query, k=3)
        
        # Second query (should hit cache)
        result2 = rag_service.query(query, k=3)
        
        assert result2.get('cached') == True
        assert result2['cache_type'] in ['exact', 'semantic']
    
    def test_cache_stats_in_get_stats(self, rag_service):
        """TEST: get_stats() includes cache statistics"""
        # Run a query to populate cache
        rag_service.query("test query for stats", k=3)
        
        stats = rag_service.get_stats()
        
        assert 'cache_stats' in stats
        assert 'total_cached_queries' in stats['cache_stats']
        assert 'cache_hit_rate' in stats['cache_stats']
    
    def test_cache_disabled_mode(self):
        """TEST: RAG can run with cache disabled"""
        pipeline = IngestionPipeline(
            collection_name="partselect_test",
            persist_directory="tests/test_data/vector_store_test"
        )
        llm = LLMService(model="google/gemma-3-27b-it:free")
        rag = RAGService(
            ingestion_pipeline=pipeline,
            llm_service=llm,
            enable_cache=False
        )
        
        assert rag.cache_enabled == False
        assert rag.cache is None