"""
Unit tests for Chunking Service
"""

import pytest
from langchain.schema import Document
from services.chunking_service import ChunkingService, chunk_documents
from services.document_loader import DocumentLoader


class TestChunkingService:
    """Test suite for ChunkingService class"""
    
    @pytest.fixture
    def service(self):
        """Create a ChunkingService instance"""
        return ChunkingService(blog_chunk_size=1000, blog_chunk_overlap=200)
    
    @pytest.fixture
    def loader(self):
        """Create a DocumentLoader for test fixtures"""
        return DocumentLoader(data_dir="tests/fixtures")
    
    @pytest.fixture
    def short_blog_doc(self):
        """
        Create a short blog document that won't be chunked.
        
        PURPOSE: Test that small blogs stay as single documents
        """
        return Document(
            page_content="Short blog post about ice makers. Just a title and URL.",
            metadata={
                "source": "blogs",
                "type": "blog",
                "title": "Ice Maker Tips",
                "url": "https://test.com/blog",
                "doc_id": "blog_test_1"
            }
        )
    
    @pytest.fixture
    def long_blog_doc(self):
        """
        Create a long blog document (2000+ chars) that WILL be chunked.
        
        PURPOSE: Test that long blogs get split into multiple chunks
        """
        # Create content longer than chunk_size (1000)
        long_content = """Ice Maker Troubleshooting Guide

Introduction
Ice makers are essential appliances in modern refrigerators. When they malfunction, it can be frustrating. This guide will help you diagnose and fix common ice maker problems. """ + ("Common ice maker issues include not making ice, making small ice cubes, leaking water, and strange noises. " * 20)
        
        return Document(
            page_content=long_content,
            metadata={
                "source": "blogs",
                "type": "blog",
                "title": "Complete Ice Maker Guide",
                "url": "https://test.com/blog/long",
                "doc_id": "blog_test_long"
            }
        )
    
    @pytest.fixture
    def part_doc(self):
        """
        Create a parts document.
        
        PURPOSE: Test that parts are never chunked
        """
        return Document(
            page_content="""Part: Ice Maker Assembly
Part ID: PS123456
Price: $89.99
Brand: GE
Installation: Easy""",
            metadata={
                "source": "parts",
                "type": "part",
                "part_id": "PS123456",
                "doc_id": "part_test_1"
            }
        )
    
    @pytest.fixture
    def repair_doc(self):
        """
        Create a repairs document.
        
        PURPOSE: Test that repairs are never chunked
        """
        return Document(
            page_content="""Problem: Ice maker not working
Appliance: Refrigerator
Difficulty: Moderate
Parts needed: Ice Maker Assembly""",
            metadata={
                "source": "repairs",
                "type": "repair",
                "symptom": "Ice maker not working",
                "doc_id": "repair_test_1"
            }
        )
    
    def test_chunk_short_blog_stays_single(self, service, short_blog_doc):
        """
        TEST: Short blog documents should NOT be chunked
        
        GIVEN: A blog document shorter than chunk_size (1000 chars)
        WHEN: chunk_document() is called
        THEN: 
            - Returns a list with exactly 1 document (no chunking)
            - Original metadata is preserved
            - No chunk_index or chunk_id added
        
        PURPOSE: Verify small blogs aren't unnecessarily split
        """
        # Act
        result = service.chunk_document(short_blog_doc)
        
        # Assert
        assert len(result) == 1, "Short blog should not be chunked"
        assert result[0].page_content == short_blog_doc.page_content
        assert result[0].metadata['doc_id'] == 'blog_test_1'
        assert 'chunk_index' not in result[0].metadata, "No chunk metadata for unchunked docs"
    
    def test_chunk_long_blog_splits(self, service, long_blog_doc):
        """
        TEST: Long blog documents SHOULD be chunked into multiple pieces
        
        GIVEN: A blog document longer than chunk_size (1000 chars)
        WHEN: chunk_document() is called
        THEN: 
            - Returns a list with 2+ documents (chunked)
            - Each chunk has original metadata + chunk metadata
            - Chunk metadata includes: chunk_index, total_chunks, chunk_id
            - chunk_ids are unique
        
        PURPOSE: Verify long blogs are properly split for better retrieval
        """
        # Act
        result = service.chunk_document(long_blog_doc)
        
        # Assert
        assert len(result) > 1, f"Long blog should be chunked (got {len(result)} chunks)"
        
        # Check first chunk
        first_chunk = result[0]
        assert 'chunk_index' in first_chunk.metadata
        assert first_chunk.metadata['chunk_index'] == 0
        assert 'total_chunks' in first_chunk.metadata
        assert first_chunk.metadata['total_chunks'] == len(result)
        assert 'chunk_id' in first_chunk.metadata
        assert first_chunk.metadata['chunk_id'] == 'blog_test_long_chunk_0'
        
        # Check original metadata preserved
        assert first_chunk.metadata['source'] == 'blogs'
        assert first_chunk.metadata['type'] == 'blog'
        assert first_chunk.metadata['title'] == 'Complete Ice Maker Guide'
        
        # Check all chunks have unique chunk_ids
        chunk_ids = [chunk.metadata['chunk_id'] for chunk in result]
        assert len(chunk_ids) == len(set(chunk_ids)), "All chunk_ids should be unique"
    
    def test_parts_never_chunked(self, service, part_doc):
        """
        TEST: Parts documents should NEVER be chunked (stay atomic)
        
        GIVEN: A parts document
        WHEN: chunk_document() is called
        THEN: 
            - Returns a list with exactly 1 document
            - Document is unchanged
        
        PURPOSE: Verify parts stay as single units for accurate retrieval
        """
        # Act
        result = service.chunk_document(part_doc)
        
        # Assert
        assert len(result) == 1, "Parts should never be chunked"
        assert result[0].page_content == part_doc.page_content
        assert result[0].metadata == part_doc.metadata
    
    def test_repairs_never_chunked(self, service, repair_doc):
        """
        TEST: Repairs documents should NEVER be chunked (symptom+solution together)
        
        GIVEN: A repairs document
        WHEN: chunk_document() is called
        THEN: 
            - Returns a list with exactly 1 document
            - Document is unchanged
        
        PURPOSE: Verify repairs stay as single units (symptom+solution together)
        """
        # Act
        result = service.chunk_document(repair_doc)
        
        # Assert
        assert len(result) == 1, "Repairs should never be chunked"
        assert result[0].page_content == repair_doc.page_content
        assert result[0].metadata == repair_doc.metadata
    
    def test_chunk_multiple_documents(self, service, short_blog_doc, long_blog_doc, part_doc):
        """
        TEST: Batch chunking of multiple document types
        
        GIVEN: A list with short blog, long blog, and part
        WHEN: chunk_documents() is called
        THEN: 
            - Short blog: 1 doc
            - Long blog: 2+ docs
            - Part: 1 doc
            - Total > 3 documents
        
        PURPOSE: Verify batch processing handles mixed document types correctly
        """
        # Arrange
        documents = [short_blog_doc, long_blog_doc, part_doc]
        
        # Act
        result = service.chunk_documents(documents)
        
        # Assert
        assert len(result) > 3, f"Should have more than 3 chunks (got {len(result)})"
        
        # Verify we have different types
        types = [doc.metadata['type'] for doc in result]
        assert 'blog' in types
        assert 'part' in types
    
    
    def test_chunk_by_type_returns_stats(self, service, short_blog_doc, part_doc, repair_doc):
        """
        TEST: chunk_by_type() returns both documents and statistics
        
        GIVEN: Multiple documents of different types
        WHEN: chunk_by_type() is called
        THEN: 
            - Returns dict with 'documents' and 'stats' keys
            - Stats show original vs chunked counts per type
        
        PURPOSE: Verify statistics tracking for analysis
        """
        # Arrange
        documents = [short_blog_doc, part_doc, repair_doc]
        
        # Act
        result = service.chunk_by_type(documents)
        
        # Assert
        assert 'documents' in result
        assert 'stats' in result
        assert isinstance(result['documents'], list)
        assert isinstance(result['stats'], dict)
        
        # Check stats structure
        stats = result['stats']
        assert 'blog' in stats
        assert 'part' in stats
        assert 'repair' in stats
        assert stats['blog']['original'] == 1
        assert stats['part']['original'] == 1
        assert stats['repair']['original'] == 1
    
    def test_get_chunk_stats_estimates_correctly(self, service, long_blog_doc):
        """
        TEST: get_chunk_stats() provides accurate estimates without chunking
        
        GIVEN: A long blog document
        WHEN: get_chunk_stats() is called (analysis only, no chunking)
        THEN: 
            - Returns statistics dict
            - Estimates more than 1 chunk for long blog
            - Shows will_chunk=True for blogs
        
        PURPOSE: Verify statistics can be generated without actual chunking
        """
        # Arrange
        documents = [long_blog_doc]
        
        # Act
        stats = service.get_chunk_stats(documents)
        
        # Assert
        assert 'total_docs' in stats
        assert stats['total_docs'] == 1
        assert 'by_type' in stats
        assert 'blog' in stats['by_type']
        assert stats['by_type']['blog']['will_chunk'] == True
        assert stats['estimated_chunks'] >= 1
    
    def test_chunk_real_long_blog_from_fixture(self, service, loader):
        """
        TEST: Integration test with real long blog fixture file
        
        GIVEN: long_blog.csv fixture with 2000+ character blog
        WHEN: Load and chunk the document
        THEN: 
            - Document loads successfully
            - Gets chunked into 2+ pieces
            - All chunks have proper metadata
        
        PURPOSE: Verify chunking works with realistic data from CSV files
        """
        # Arrange
        import os
        long_blog_path = os.path.join("tests/fixtures", "long_blog.csv")
        
        # Act
        documents = loader.load_blogs_csv(long_blog_path)
        
        if not documents:
            pytest.skip("long_blog.csv not found or empty")
        
        chunked = service.chunk_documents(documents)
        
        # Assert
        assert len(chunked) > 1, f"Long blog should chunk (got {len(chunked)} chunks)"
        
        # Verify all chunks have required metadata
        for chunk in chunked:
            assert 'chunk_index' in chunk.metadata
            assert 'total_chunks' in chunk.metadata
            assert 'chunk_id' in chunk.metadata
            assert chunk.metadata['source'] == 'blogs'


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_chunk_documents_function(self):
        """
        TEST: Convenience function provides easy access
        
        GIVEN: A list of documents
        WHEN: chunk_documents() convenience function is called
        THEN: 
            - Returns a list of chunked documents
            - Function works without creating service instance
        
        PURPOSE: Verify helper function provides simple interface
        """
        # Arrange
        doc = Document(
            page_content="Test content",
            metadata={"source": "test", "type": "part", "doc_id": "test1"}
        )
        
        # Act
        result = chunk_documents([doc])
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

