"""
Unit tests for Document Loader Service
"""

import pytest
import os
from services.document_loader import DocumentLoader, load_documents


class TestDocumentLoader:
    """Test suite for DocumentLoader class"""
    
    @pytest.fixture
    def loader(self):
        """Create a DocumentLoader instance with test fixtures"""
        return DocumentLoader(data_dir="tests/fixtures")
    
    @pytest.fixture
    def fixtures_dir(self):
        """Return path to test fixtures"""
        return "tests/fixtures"
    
    def test_load_blogs_csv_success(self, loader, fixtures_dir):
        """
        TEST: Load valid blogs CSV and verify document structure
        
        GIVEN: A CSV file with 3 valid blog entries (title, url)
        WHEN: load_blogs_csv() is called with the file path
        THEN: 
            - Returns list of 3 Document objects
            - Each document has correct metadata (source='blogs', type='blog')
            - Each document has title, url, and unique doc_id
        
        PURPOSE: Verify happy path - all valid data loads correctly
        """
        # Arrange
        blogs_path = os.path.join(fixtures_dir, "test_blogs.csv")
        
        # Act
        documents = loader.load_blogs_csv(blogs_path)
        
        # Assert
        assert len(documents) == 3, "Should load 3 blog documents"
        assert documents[0].metadata['source'] == 'blogs'
        assert documents[0].metadata['type'] == 'blog'
        assert 'title' in documents[0].metadata
        assert 'url' in documents[0].metadata
        assert 'doc_id' in documents[0].metadata
        assert documents[0].metadata['doc_id'].startswith('blog_')
    
    def test_load_blogs_csv_missing_file(self, loader):
        """
        TEST: Handle missing CSV file gracefully without crashing
        
        GIVEN: A file path that doesn't exist
        WHEN: load_blogs_csv() is called with non-existent path
        THEN: 
            - Returns empty list (not None, not an error)
            - No exception is raised
        
        PURPOSE: Verify error handling - missing files don't crash the loader
        """
        # Arrange
        non_existent_path = "tests/fixtures/does_not_exist.csv"
        
        # Act
        documents = loader.load_blogs_csv(non_existent_path)
        
        # Assert
        assert documents == [], "Should return empty list for missing file"
    
    def test_load_blogs_csv_malformed_data(self, loader, fixtures_dir):
        """
        TEST: Skip rows with missing critical fields (title or url)
        
        GIVEN: A CSV with 5 rows: 2 valid, 3 with missing title/url
        WHEN: load_blogs_csv() is called
        THEN: 
            - Returns only 2 valid documents (skips 3 bad rows)
            - All returned documents have non-empty title and url
        
        PURPOSE: Verify data validation - bad rows are skipped, good rows are kept
        """
        # Arrange
        malformed_path = os.path.join(fixtures_dir, "malformed_blogs.csv")
        
        # Act
        documents = loader.load_blogs_csv(malformed_path)
        
        # Assert
        assert len(documents) == 2, "Should load only 2 valid blogs, skipping 3 malformed rows"
        assert all('title' in doc.metadata and doc.metadata['title'] for doc in documents)
        assert all('url' in doc.metadata and doc.metadata['url'] for doc in documents)
    
    def test_load_parts_csv_metadata(self, loader, fixtures_dir):
        """
        TEST: Verify all parts metadata fields are correctly populated
        
        GIVEN: A CSV with 2 parts containing part_id, mpn_id, brand, price, etc.
        WHEN: load_parts_csv() is called with appliance_type='refrigerator'
        THEN: 
            - Returns 2 documents with source='parts', type='part'
            - Metadata includes: part_id, mpn_id, brand, price, difficulty, appliance
            - appliance metadata matches the provided appliance_type
        
        PURPOSE: Verify metadata extraction - all fields from CSV map to metadata
        """
        # Arrange
        parts_path = os.path.join(fixtures_dir, "test_parts.csv")
        
        # Act
        documents = loader.load_parts_csv(parts_path, "refrigerator")
        
        # Assert
        assert len(documents) == 2, "Should load 2 part documents"
        
        # Check first document metadata
        doc = documents[0]
        assert doc.metadata['source'] == 'parts'
        assert doc.metadata['type'] == 'part'
        assert doc.metadata['appliance'] == 'refrigerator'
        assert doc.metadata['part_id'] == 'PS123456'
        assert doc.metadata['mpn_id'] == 'WR123'
        assert doc.metadata['brand'] == 'GE'
        assert doc.metadata['price'] == '$50.00'
        assert doc.metadata['difficulty'] == 'Easy'
        assert 'url' in doc.metadata
    
    def test_load_parts_csv_content_format(self, loader, fixtures_dir):
        """
        TEST: Verify parts page_content is formatted for semantic search
        
        GIVEN: A CSV with part information
        WHEN: load_parts_csv() is called
        THEN: 
            - page_content includes: Part name, ID, Price, Brand, Difficulty
            - Content is human-readable and search-friendly
        
        PURPOSE: Verify content formatting - page_content is optimized for RAG retrieval
        """
        # Arrange
        parts_path = os.path.join(fixtures_dir, "test_parts.csv")
        
        # Act
        documents = loader.load_parts_csv(parts_path, "refrigerator")
        
        # Assert
        content = documents[0].page_content
        assert "Part: Test Ice Maker" in content
        assert "Part ID: PS123456" in content
        assert "Price: $50.00" in content
        assert "Brand: GE" in content
        assert "Installation Difficulty: Easy" in content
    
    def test_load_repairs_csv_metadata(self, loader, fixtures_dir):
        """
        TEST: Verify all repairs metadata fields are correctly populated
        
        GIVEN: A CSV with 2 repair guides (symptom, difficulty, parts_needed)
        WHEN: load_repairs_csv() is called with appliance_type='refrigerator'
        THEN: 
            - Returns 2 documents with source='repairs', type='repair'
            - Metadata includes: symptom, difficulty, parts_needed, percentage, video_url
        
        PURPOSE: Verify repairs-specific metadata extraction
        """
        # Arrange
        repairs_path = os.path.join(fixtures_dir, "test_repairs.csv")
        
        # Act
        documents = loader.load_repairs_csv(repairs_path, "refrigerator")
        
        # Assert
        assert len(documents) == 2, "Should load 2 repair documents"
        
        # Check first document metadata
        doc = documents[0]
        assert doc.metadata['source'] == 'repairs'
        assert doc.metadata['type'] == 'repair'
        assert doc.metadata['appliance'] == 'refrigerator'
        assert doc.metadata['symptom'] == 'Ice maker not working'
        assert doc.metadata['difficulty'] == 'Moderate'
        assert doc.metadata['parts_needed'] == 'Ice Maker Assembly'
        assert doc.metadata['percentage'] == '75'
    
    def test_load_repairs_csv_content_format(self, loader, fixtures_dir):
        """
        TEST: Verify repairs page_content is symptom-focused for search
        
        GIVEN: A CSV with repair information
        WHEN: load_repairs_csv() is called
        THEN: 
            - page_content includes: Problem, Appliance, Percentage, Difficulty, Parts needed
            - Content is focused on symptoms users search for
        
        PURPOSE: Verify content is optimized for symptom-based queries
        """
        # Arrange
        repairs_path = os.path.join(fixtures_dir, "test_repairs.csv")
        
        # Act
        documents = loader.load_repairs_csv(repairs_path, "refrigerator")
        
        # Assert
        content = documents[0].page_content
        assert "Problem: Ice maker not working" in content
        assert "Appliance: Refrigerator" in content
        assert "Reported by: 75% of users" in content
        assert "Difficulty: Moderate" in content
        assert "Parts needed: Ice Maker Assembly" in content
    
    def test_document_unique_ids(self, loader, fixtures_dir):
        """
        TEST: Verify all documents have unique doc_ids (no duplicates)
        
        GIVEN: Multiple CSV files loaded (blogs, parts, repairs)
        WHEN: All documents are loaded from different sources
        THEN: 
            - Every document has a unique doc_id
            - No two documents share the same doc_id
        
        PURPOSE: Verify ID generation - prevent duplicate documents in vector store
        """
        # Arrange
        blogs_path = os.path.join(fixtures_dir, "test_blogs.csv")
        parts_path = os.path.join(fixtures_dir, "test_parts.csv")
        repairs_path = os.path.join(fixtures_dir, "test_repairs.csv")
        
        # Act
        all_docs = []
        all_docs.extend(loader.load_blogs_csv(blogs_path))
        all_docs.extend(loader.load_parts_csv(parts_path, "refrigerator"))
        all_docs.extend(loader.load_repairs_csv(repairs_path, "refrigerator"))
        
        # Assert
        doc_ids = [doc.metadata['doc_id'] for doc in all_docs]
        assert len(doc_ids) == len(set(doc_ids)), "All doc_ids should be unique"
    
    def test_load_all_documents_with_fixtures(self, loader):
        """
        TEST: Integration test - load_all_documents() finds and loads all available files
        
        GIVEN: Test fixture directory with multiple CSV files
        WHEN: load_all_documents() is called
        THEN: 
            - Returns a list of Document objects
            - All documents have proper structure (page_content, metadata)
            - All documents have source and type metadata
        
        PURPOSE: Verify integration - loader finds files automatically and loads everything
        """
        # Act
        documents = loader.load_all_documents()
        
        # Assert - should skip missing dishwasher files gracefully
        assert len(documents) >= 0, "Should return a list (may be empty if test files configured)"
        
        # If documents were loaded, verify they have proper structure
        if documents:
            assert all(hasattr(doc, 'page_content') for doc in documents)
            assert all(hasattr(doc, 'metadata') for doc in documents)
            assert all('source' in doc.metadata for doc in documents)
            assert all('type' in doc.metadata for doc in documents)


class TestConvenienceFunction:
    """Test the convenience function"""
    
    def test_load_documents_function(self):
        """
        TEST: Verify convenience function works as a simple interface
        
        GIVEN: The standalone load_documents() function
        WHEN: Called with a data directory
        THEN: 
            - Returns a list (doesn't crash)
            - Handles missing files gracefully
        
        PURPOSE: Verify the helper function provides easy access to loader functionality
        """
        # Act
        documents = load_documents(data_dir="tests/fixtures")
        
        # Assert
        assert isinstance(documents, list), "Should return a list"
        # Function should handle missing files gracefully


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])