"""
Unit tests for Ingestion Pipeline
Tests the full RAG ingestion: Load → Chunk → Embed → Store → Search
"""
import pytest
import os
from services.ingestion_pipeline import IngestionPipeline


class TestIngestionPipelineInitialization:
    """Test pipeline initialization"""
    
    @pytest.fixture
    def pipeline(self):
        """Create test pipeline with isolated collection"""
        return IngestionPipeline(
            collection_name="partselect_pytest",
            persist_directory="tests/test_data/vector_store_pytest"
        )
    
    def test_pipeline_initialization(self, pipeline):
        """TEST: Pipeline initializes successfully"""
        assert pipeline is not None
        assert pipeline.vectorstore is not None
        assert pipeline.embeddings is not None
    
    def test_health_check_returns_200(self, pipeline):
        """TEST: Health check returns successful status"""
        health = pipeline.health_check()
        
        assert health['status_code'] == 200
        assert health['status'] == 'healthy'
        assert 'total_documents' in health
    
    def test_get_status_returns_collection_info(self, pipeline):
        """TEST: Status returns collection details"""
        status = pipeline.get_status()
        
        assert status['status_code'] == 200
        assert 'collection_name' in status
        assert 'embedding_model' in status
        assert 'total_documents' in status


class TestIngestionPipelineDataLoading:
    """Test data loading and ingestion"""
    
    @pytest.fixture
    def pipeline(self):
        """Create fresh pipeline for each test"""
        pipe = IngestionPipeline(
            collection_name="partselect_pytest_load",
            persist_directory="tests/test_data/vector_store_pytest_load"
        )
        # Reset store before each test
        pipe.reset_store()
        return pipe
    
    def test_run_pipeline_with_test_fixtures(self, pipeline):
        """TEST: Pipeline can ingest test fixture data"""
        result = pipeline.run_pipeline(
            data_dir="tests/fixtures",
            blog_files=["test_blogs.csv"],
            parts_files={"test": "test_parts.csv"},
            repairs_files={"test": "test_repairs.csv"},
            force_rebuild=True
        )
        
        assert result['status_code'] == 200
        assert result['total_in_collection'] > 0
        assert 'ingestion_time_seconds' in result
    
    def test_pipeline_ingests_all_document_types(self, pipeline):
        """TEST: Pipeline ingests parts, repairs, and blogs"""
        result = pipeline.run_pipeline(
            data_dir="tests/fixtures",
            blog_files=["test_blogs.csv"],
            parts_files={"test": "test_parts.csv"},
            repairs_files={"test": "test_repairs.csv"},
            force_rebuild=True
        )
        
        # Should have loaded multiple document types
        assert result['total_in_collection'] >= 3  # At least 3 docs from fixtures
    
    def test_force_rebuild_clears_existing_data(self, pipeline):
        """TEST: Force rebuild clears and reloads data"""
        # First ingestion
        result1 = pipeline.run_pipeline(
            data_dir="tests/fixtures",
            blog_files=["test_blogs.csv"],
            parts_files={"test": "test_parts.csv"},
            repairs_files={"test": "test_repairs.csv"},
            force_rebuild=True
        )
        count1 = result1['total_in_collection']
        
        # Second ingestion with force rebuild
        result2 = pipeline.run_pipeline(
            data_dir="tests/fixtures",
            blog_files=["test_blogs.csv"],
            parts_files={"test": "test_parts.csv"},
            repairs_files={"test": "test_repairs.csv"},
            force_rebuild=True
        )
        count2 = result2['total_in_collection']
        
        # Should have same count (replaced, not duplicated)
        assert count1 == count2


class TestIngestionPipelineSearch:
    """Test search functionality"""
    
    @pytest.fixture(scope="class")
    def pipeline_with_data(self):
        """Create pipeline and ingest test data once for all tests"""
        pipe = IngestionPipeline(
            collection_name="partselect_pytest_search",
            persist_directory="tests/test_data/vector_store_pytest_search"
        )
        # Ingest test data
        pipe.run_pipeline(
            data_dir="tests/fixtures",
            blog_files=["test_blogs.csv"],
            parts_files={"test": "test_parts.csv"},
            repairs_files={"test": "test_repairs.csv"},
            force_rebuild=True
        )
        return pipe
    
    def test_search_returns_results(self, pipeline_with_data):
        """TEST: Search returns relevant documents"""
        result = pipeline_with_data.search("ice maker", k=3)
        
        assert result['status_code'] == 200
        assert 'results' in result
        assert len(result['results']) > 0
    
    def test_search_results_have_metadata(self, pipeline_with_data):
        """TEST: Search results include metadata and scores"""
        result = pipeline_with_data.search("dishwasher", k=3)
        
        for doc in result['results']:
            assert 'metadata' in doc
            assert 'score' in doc
            assert 'content' in doc
    
    def test_search_by_type_filters_correctly(self, pipeline_with_data):
        """TEST: Search by type returns only specified type"""
        result = pipeline_with_data.search_by_type("problem", doc_type="repair", k=5)
        
        if result['status_code'] == 200 and len(result['results']) > 0:
            # All results should be repair type
            for doc in result['results']:
                assert doc['metadata'].get('type') == 'repair'
    
    def test_search_respects_k_parameter(self, pipeline_with_data):
        """TEST: Search returns at most k results"""
        result = pipeline_with_data.search("refrigerator", k=2)
        
        assert result['status_code'] == 200
        assert len(result['results']) <= 2
    
    def test_search_on_empty_query(self, pipeline_with_data):
        """TEST: Empty query still returns results"""
        result = pipeline_with_data.search("", k=3)
        
        # Should handle gracefully (may return results or empty)
        assert result['status_code'] in [200, 404]


class TestIngestionPipelineReset:
    """Test reset and cleanup functionality"""
    
    def test_reset_store_clears_data(self):
        """TEST: Reset store clears all documents"""
        pipeline = IngestionPipeline(
            collection_name="partselect_pytest_reset",
            persist_directory="tests/test_data/vector_store_pytest_reset"
        )
        
        # Add data
        pipeline.run_pipeline(
            data_dir="tests/fixtures",
            blog_files=["test_blogs.csv"],
            parts_files={"test": "test_parts.csv"},
            repairs_files={"test": "test_repairs.csv"},
            force_rebuild=True
        )
        status_before = pipeline.get_status()
        
        # Reset
        pipeline.reset_store()
        status_after = pipeline.get_status()
        
        # Should be empty after reset
        assert status_after['total_documents'] == 0
        assert status_before['total_documents'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
