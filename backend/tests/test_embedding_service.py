"""
Test/Demo script for Embedding Service
Run with: python3 -m tests.test_embedding_service
"""

from services.embedding_service import EmbeddingService
from langchain_core.documents import Document


# Test/Demo code
if __name__ == "__main__":
    print("\n=== Testing Embedding Service ===\n")
    
    # Test 1: Initialize service
    print("Test 1: Initialize embedding service")
    service = EmbeddingService()
    info = service.get_model_info()
    print(f"Model info: {info}\n")
    
    # Test 2: Embed a single query
    print("Test 2: Embed a single query")
    query = "ice maker not working"
    query_embedding = service.embed_query(query)
    print(f"Query: '{query}'")
    print(f"Embedding dimension: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}\n")
    
    # Test 3: Embed documents
    print("Test 3: Embed sample documents")
    
    sample_docs = [
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
    
    embedded_docs = service.embed_documents(sample_docs)
    
    for i, emb_doc in enumerate(embedded_docs, 1):
        print(f"\nDocument {i}:")
        print(f"  Content: {emb_doc['page_content'][:50]}...")
        print(f"  Type: {emb_doc['metadata']['type']}")
        print(f"  Embedding dimension: {len(emb_doc['embedding'])}")
        print(f"  First 3 values: {emb_doc['embedding'][:3]}")
    
    # Test 4: Test with real data (if available)
    print("\n\nTest 4: Embed real documents (if available)")
    try:
        from services.document_loader import DocumentLoader
        from services.chunking_service import ChunkingService
        
        loader = DocumentLoader(data_dir="data/raw")
        documents = loader.load_all_documents()
        
        if documents:
            # Take a small sample
            sample = documents[:5]
            print(f"Loaded {len(documents)} documents, testing with {len(sample)}")
            
            # Chunk them
            chunker = ChunkingService()
            chunks = chunker.chunk_documents(sample)
            print(f"Created {len(chunks)} chunks")
            
            # Embed them
            embedded = service.embed_documents(chunks)
            print(f"✓ Successfully embedded {len(embedded)} chunks")
            
            # Show stats
            avg_content_length = sum(len(e['page_content']) for e in embedded) / len(embedded)
            print(f"Average content length: {avg_content_length:.0f} characters")
        else:
            print("No documents found in data/raw/")
    
    except Exception as e:
        print(f"Could not test with real data: {e}")
    
    print("\n✓ All tests completed!")