"""
Embedding Service
Converts text chunks into vector embeddings.
Pure transformation service - no database handling.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


class EmbeddingService:
    """Service for converting text into vector embeddings."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model name for embeddings
                       Options:
                       - "all-MiniLM-L6-v2" (384 dim, fast, recommended)
                       - "all-mpnet-base-v2" (768 dim, more accurate, slower)
            device: 'cpu' or 'cuda' (if GPU available)
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading embedding model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}  # For cosine similarity
        )
        print(f"✓ Embedding model loaded ({model_name})")
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Embed a list of Document objects.
        
        Args:
            documents: List of LangChain Document objects
        
        Returns:
            List of dicts containing:
                - page_content: Original text
                - metadata: Original metadata
                - embedding: Vector embedding (list of floats)
        """
        if not documents:
            return []
        
        print(f"Embedding {len(documents)} documents...")
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Combine documents with their embeddings
        embedded_docs = []
        for doc, embedding in zip(documents, embeddings):
            embedded_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding
            })
        
        print(f"✓ Embedded {len(embedded_docs)} documents")
        return embedded_docs
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query: Query text to embed
        
        Returns:
            Vector embedding as list of floats
        """
        return self.embeddings.embed_query(query)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        # Embed a dummy text to get dimension
        dummy_embedding = self.embed_query("test")
        return len(dummy_embedding)
    
    def batch_embed_documents(
        self,
        documents: List[Document],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Embed documents in batches for memory efficiency.
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents to process at once
        
        Returns:
            List of embedded documents
        """
        if not documents:
            return []
        
        print(f"Batch embedding {len(documents)} documents (batch_size={batch_size})...")
        
        all_embedded = []
        total = len(documents)
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            embedded_batch = self.embed_documents(batch)
            all_embedded.extend(embedded_batch)
            print(f"  Progress: {min(i + batch_size, total)}/{total}")
        
        print(f"✓ Batch embedding completed")
        return all_embedded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "normalized": True  # We use normalized embeddings for cosine similarity
        }


# Convenience function
def create_embedding_service(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu"
) -> EmbeddingService:
    """Create and return an EmbeddingService instance."""
    return EmbeddingService(model_name=model_name, device=device)
