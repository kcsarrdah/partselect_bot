"""
Ingestion Pipeline
Orchestrates the RAG data pipeline: Load → Chunk → Embed → Store in ChromaDB
"""

import os
from typing import Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import time
from services.document_loader import DocumentLoader
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService


class IngestionPipeline:
    """Orchestrates the full RAG ingestion pipeline."""
    
    def __init__(
        self,
        collection_name: str = "partselect_docs",
        persist_directory: str = "data/vector_store",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            embedding_model: HuggingFace embedding model name
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings
        print(f"Initializing embeddings ({embedding_model})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize or load ChromaDB
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing ChromaDB vector store."""
        try:
            print(f"Initializing ChromaDB collection: {self.collection_name}...")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Get current count
            count = self.vectorstore._collection.count()
            print(f"✓ ChromaDB initialized ({count} documents in collection)")
            
        except Exception as e:
            print(f"✗ Error initializing ChromaDB: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the vector store.
        
        Returns:
            Dictionary with status information and HTTP-like status code
        """
        try:
            if self.vectorstore is None:
                return {
                    "status_code": 503,
                    "status": "error",
                    "message": "Vector store not initialized"
                }
            
            # Get collection info
            collection = self.vectorstore._collection
            doc_count = collection.count()
            
            # Try to get sample metadata
            metadata_keys = set()
            doc_types = {}
            
            if doc_count > 0:
                sample = collection.peek(limit=min(doc_count, 100))
                if sample and 'metadatas' in sample:
                    for meta in sample['metadatas']:
                        if meta:
                            metadata_keys.update(meta.keys())
                            doc_type = meta.get('type', 'unknown')
                            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            return {
                "status_code": 200,
                "status": "ready",
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model_name,
                "total_documents": doc_count,
                "document_types": doc_types,
                "metadata_keys": list(metadata_keys),
                "is_empty": doc_count == 0
            }
            
        except Exception as e:
            return {
                "status_code": 500,
                "status": "error",
                "message": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Quick health check endpoint (lighter than get_status).
        
        Returns:
            Dictionary with basic health info
        """
        try:
            if self.vectorstore is None:
                return {
                    "status_code": 503,
                    "status": "unhealthy",
                    "message": "Vector store not initialized"
                }
            
            doc_count = self.vectorstore._collection.count()
            
            return {
                "status_code": 200,
                "status": "healthy",
                "total_documents": doc_count,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            return {
                "status_code": 500,
                "status": "unhealthy",
                "message": str(e)
            }

    def run_pipeline(
        self,
        data_dir: str = "data/raw",
        batch_size: int = 32,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Run the full ingestion pipeline: Load → Chunk → Embed → Store
        
        Args:
            data_dir: Directory containing CSV files to ingest
            batch_size: Number of documents to process at once
            force_rebuild: If True, delete existing collection and rebuild
        
        Returns:
            Dictionary with pipeline execution statistics
        """
        
        start_time = time.time()
        
        try:
            print(f"\n{'='*60}")
            print("Starting RAG Ingestion Pipeline")
            print(f"{'='*60}\n")
            
            # Step 0: Reset if requested
            if force_rebuild:
                print("🔄 Force rebuild enabled - clearing existing collection...")
                self.reset_store()
            
            # Step 1: Load documents from CSVs
            print(f"📂 Step 1: Loading documents from {data_dir}...")
            loader = DocumentLoader(data_dir=data_dir)
            documents = loader.load_all_documents()
            
            if not documents:
                return {
                    "status_code": 400,
                    "status": "error",
                    "message": f"No documents found in {data_dir}"
                }
            
            print(f"   ✓ Loaded {len(documents)} documents")
            
            # Step 2: Chunk documents
            print(f"\n✂️  Step 2: Chunking documents...")
            chunker = ChunkingService()
            chunks = chunker.chunk_documents(documents)
            print(f"   ✓ Created {len(chunks)} chunks")
            
            # Step 3: Embed chunks (handled by ChromaDB via add_documents)
            print(f"\n🧮 Step 3: Embedding and storing in ChromaDB...")
            print(f"   Processing {len(chunks)} chunks in batches of {batch_size}...")
            
            # Add documents to ChromaDB in batches
            total_added = 0
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # ChromaDB will automatically embed these using self.embeddings
                self.vectorstore.add_documents(batch)
                
                total_added += len(batch)
                print(f"   Progress: {total_added}/{len(chunks)} chunks stored")
            
            # Persist to disk
            print(f"\n💾 Step 4: Persisting to disk...")
            # Note: Chroma in newer versions auto-persists, but we'll be explicit
            # self.vectorstore.persist()  # Uncomment if using older Chroma version
            
            # Calculate stats
            end_time = time.time()
            time_taken = end_time - start_time
            
            # Get final collection stats
            final_count = self.vectorstore._collection.count()
            
            print(f"\n{'='*60}")
            print("✅ Pipeline Complete!")
            print(f"{'='*60}")
            print(f"Documents Loaded:  {len(documents)}")
            print(f"Chunks Created:    {len(chunks)}")
            print(f"Chunks Stored:     {total_added}")
            print(f"Total in DB:       {final_count}")
            print(f"Time Taken:        {time_taken:.2f} seconds")
            print(f"{'='*60}\n")
            
            return {
                "status_code": 200,
                "status": "success",
                "documents_loaded": len(documents),
                "documents_chunked": len(chunks),
                "documents_stored": total_added,
                "total_in_collection": final_count,
                "time_taken_seconds": round(time_taken, 2),
                "collection_name": self.collection_name,
                "data_dir": data_dir
            }
        
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status_code": 500,
                "status": "error",
                "message": str(e),
                "time_taken_seconds": round(time.time() - start_time, 2)
            }


    def reset_store(self):
        """
        Clear all documents from the collection.
        Useful for rebuilding the index from scratch.
        """
        try:
            print(f"🗑️  Clearing collection: {self.collection_name}...")
            
            # Delete all documents
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            
            try:
                client.delete_collection(name=self.collection_name)
                print(f"   ✓ Deleted collection")
            except:
                print(f"   Collection doesn't exist yet (first run)")
            
            # Reinitialize
            self._initialize_vectorstore()
            print(f"   ✓ Collection reset complete")
            
        except Exception as e:
            print(f"   ✗ Error resetting collection: {e}")
            raise
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search the vector store for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"type": "part"})
        
        Returns:
            Dictionary with search results and scores
        """
        try:
            if self.vectorstore is None:
                return {
                    "status_code": 503,
                    "status": "error",
                    "message": "Vector store not initialized"
                }
            
            # Check if collection is empty
            doc_count = self.vectorstore._collection.count()
            if doc_count == 0:
                return {
                    "status_code": 404,
                    "status": "error",
                    "message": "No documents in collection. Run pipeline first."
                }
            
            print(f"\n🔍 Searching for: '{query}'")
            if filter_metadata:
                print(f"   Filters: {filter_metadata}")
            
            # Perform similarity search with scores
            if filter_metadata:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k
                )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)  # Convert to float for JSON serialization
                })
            
            print(f"   ✓ Found {len(formatted_results)} results\n")
            
            return {
                "status_code": 200,
                "status": "success",
                "query": query,
                "num_results": len(formatted_results),
                "results": formatted_results
            }
        
        except Exception as e:
            print(f"   ✗ Search failed: {e}")
            return {
                "status_code": 500,
                "status": "error",
                "message": str(e)
            }


    def search_by_type(
        self,
        query: str,
        doc_type: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Convenience method to search within a specific document type.
        
        Args:
            query: Search query
            doc_type: Document type ('blog', 'part', or 'repair')
            k: Number of results
        
        Returns:
            Dictionary with search results
        """
        return self.search(
            query,
            k=k,
            filter_metadata={"type": doc_type}
        )

# Test code
if __name__ == "__main__":
    print("\n=== Testing Ingestion Pipeline - Step 1: Setup ===\n")
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        collection_name="partselect_test",
        persist_directory="data/vector_store_test"
    )
    
    # Test health check
    print("\n1. Health Check:")
    health = pipeline.health_check()
    print(f"   Status Code: {health['status_code']}")
    print(f"   Status: {health['status']}")
    print(f"   Documents: {health.get('total_documents', 0)}")
    
    # Test full status
    print("\n2. Full Status:")
    status = pipeline.get_status()
    print(f"   Status Code: {status['status_code']}")
    print(f"   Status: {status['status']}")
    print(f"   Collection: {status.get('collection_name', 'N/A')}")
    print(f"   Model: {status.get('embedding_model', 'N/A')}")
    print(f"   Total Docs: {status.get('total_documents', 0)}")
    print(f"   Is Empty: {status.get('is_empty', True)}")
    
    if status['status_code'] == 200:
        print("\n✓ ChromaDB setup successful!")
    else:
        print(f"\n✗ Setup failed: {status.get('message', 'Unknown error')}")