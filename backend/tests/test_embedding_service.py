"""
Unit tests for Embedding Service
"""
import pytest
from services.embedding_service import EmbeddingService
from langchain_core.documents import Document


class TestEmbeddingServiceInitialization:
    """Test embedding service initialization"""
    
    @pytest.fixture
    def service(self):
        """Create embedding service instance"""
        return EmbeddingService()
    
    def test_service_initialization(self, service):
        """TEST: Service initializes successfully"""
        assert service is not None
        assert service.embeddings is not None
    
    def test_get_model_info(self, service):
        """TEST: Model info returns correct details"""
    info = service.get_model_info()
        
        assert 'model_name' in info
        assert 'embedding_dimension' in info
        assert info['embedding_dimension'] == 384  # all-MiniLM-L6-v2 dimension


class TestEmbeddingServiceQueryEmbedding:
    """Test query embedding functionality"""
    
    @pytest.fixture
    def service(self):
        """Create embedding service instance"""
        return EmbeddingService()
    
    def test_embed_single_query(self, service):
        """TEST: Embed a single query string"""
    query = "ice maker not working"
        embedding = service.embed_query(query)
        
        assert embedding is not None
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding[:5])
    
    def test_embed_different_queries_differ(self, service):
        """TEST: Different queries produce different embeddings"""
        query1 = "ice maker not working"
        query2 = "dishwasher leaking water"
        
        emb1 = service.embed_query(query1)
        emb2 = service.embed_query(query2)
        
        # Embeddings should be different
        assert emb1 != emb2
    
    def test_embed_empty_query(self, service):
        """TEST: Empty query returns valid embedding"""
        embedding = service.embed_query("")
        
        assert embedding is not None
        assert len(embedding) == 384


class TestEmbeddingServiceDocumentEmbedding:
    """Test document embedding functionality"""
    
    @pytest.fixture
    def service(self):
        """Create embedding service instance"""
        return EmbeddingService()
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
        Document(
            page_content="The ice maker assembly needs to be replaced.",
            metadata={"type": "part", "doc_id": "test_1"}
        ),
        Document(
            page_content="Dishwasher spray arm is not rotating properly.",
            metadata={"type": "repair", "doc_id": "test_2"}
        ),
        Document(
            page_content="How to install a new refrigerator water filter.",
            metadata={"type": "blog", "doc_id": "test_3"}
        )
    ]
    
    def test_embed_multiple_documents(self, service, sample_documents):
        """TEST: Embed multiple documents at once"""
        embedded_docs = service.embed_documents(sample_documents)
        
        assert len(embedded_docs) == 3
        assert all('embedding' in doc for doc in embedded_docs)
        assert all('page_content' in doc for doc in embedded_docs)
        assert all('metadata' in doc for doc in embedded_docs)
    
    def test_embedded_documents_preserve_metadata(self, service, sample_documents):
        """TEST: Embeddings preserve original document metadata"""
        embedded_docs = service.embed_documents(sample_documents)
        
        for orig, embedded in zip(sample_documents, embedded_docs):
            assert embedded['page_content'] == orig.page_content
            assert embedded['metadata'] == orig.metadata
    
    def test_embedded_documents_have_correct_dimensions(self, service, sample_documents):
        """TEST: All embeddings have correct dimension"""
        embedded_docs = service.embed_documents(sample_documents)
        
        for doc in embedded_docs:
            assert len(doc['embedding']) == 384
    
    def test_embed_empty_document_list(self, service):
        """TEST: Empty document list returns empty result"""
        embedded_docs = service.embed_documents([])
        
        assert embedded_docs == []


class TestEmbeddingServiceConsistency:
    """Test embedding consistency and determinism"""
    
    @pytest.fixture
    def service(self):
        """Create embedding service instance"""
        return EmbeddingService()
    
    def test_same_query_produces_same_embedding(self, service):
        """TEST: Same query produces identical embedding"""
        query = "refrigerator not cooling"
        
        emb1 = service.embed_query(query)
        emb2 = service.embed_query(query)
        
        # Should be identical (deterministic)
        assert emb1 == emb2
    
    def test_similar_queries_have_high_similarity(self, service):
        """TEST: Similar queries produce similar embeddings"""
        query1 = "ice maker broken"
        query2 = "ice maker not working"
        
        emb1 = service.embed_query(query1)
        emb2 = service.embed_query(query2)
        
        # Calculate cosine similarity
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Similar queries should have high similarity (>0.7)
        assert similarity > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
