"""
Unit tests for Query Cache Service
"""
import pytest
import os
from services.query_cache import QueryCache

class TestQueryCacheBasics:
    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache with temp database"""
        db_path = tmp_path / "test_cache.db"
        return QueryCache(db_path=str(db_path), similarity_threshold=0.90)
    
    def test_cache_initialization(self, cache):
        """TEST: Cache initializes with empty database"""
        stats = cache.get_stats()
        assert stats['total_cached_queries'] == 0
        assert stats['total_cache_hits'] == 0
    
    def test_exact_match_caching(self, cache):
        """TEST: Exact query match returns cached result"""
        # Cache a query
        query = "my dishwasher is not working"
        response = {
            "answer": "Test answer",
            "sources": [],
            "metadata": {"tokens_used": 100, "response_time_seconds": 2.0}
        }
        cache.set(query, response)
        
        # Retrieve exact match
        cached = cache.get(query)
        assert cached is not None
        assert cached['answer'] == "Test answer"
        assert cached['cached'] == True
        assert cached['cache_type'] == "exact"
    
    def test_semantic_similarity_matching(self, cache):
        """TEST: Similar query returns cached result"""
        # Cache original query
        cache.set("my dishwasher is not working", {
            "answer": "Check the water valve",
            "sources": [],
            "metadata": {"tokens_used": 150, "response_time_seconds": 3.0}
        })
        
        # Try similar query (should match with 0.90 threshold)
        cached = cache.get("dishwasher not working")
        assert cached is not None
        assert cached['cache_type'] == "semantic"
        assert cached['similarity'] >= 0.90
    
    def test_cache_miss_for_different_query(self, cache):
        """TEST: Unrelated query doesn't hit cache"""
        cache.set("dishwasher problem", {"answer": "A", "sources": [], "metadata": {}})
        
        # Very different query
        cached = cache.get("refrigerator ice maker broken")
        assert cached is None
    
    def test_cache_statistics_tracking(self, cache):
        """TEST: Cache tracks hits and statistics"""
        # Add queries
        cache.set("query 1", {"answer": "A", "sources": [], "metadata": {"tokens_used": 100}})
        cache.set("query 2", {"answer": "B", "sources": [], "metadata": {"tokens_used": 200}})
        
        # Hit cache multiple times
        cache.get("query 1")
        cache.get("query 1")
        cache.get("query 2")
        
        stats = cache.get_stats()
        assert stats['total_cached_queries'] == 2
        assert stats['total_cache_hits'] >= 3  # 2 initial + 3 hits
    
    def test_access_count_increments(self, cache):
        """TEST: Access count increments on each cache hit"""
        cache.set("test query", {"answer": "A", "sources": [], "metadata": {}})
        
        # Hit multiple times
        cached1 = cache.get("test query")
        cached2 = cache.get("test query")
        cached3 = cache.get("test query")
        
        assert cached3['access_count'] == 3  # 3 hits (set() doesn't increment)


class TestQueryCacheIntegration:
    """Integration tests with RAG service"""
    
    def test_rag_uses_cache_on_duplicate_query(self):
        """TEST: RAG returns cached answer for duplicate query"""
        # TODO: Integration test with RAG service
        pass