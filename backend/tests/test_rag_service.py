"""
Test script for RAG Service
Tests end-to-end RAG workflow.
"""

from services.ingestion_pipeline import IngestionPipeline
from services.llm_service import LLMService
from services.rag_service import RAGService


if __name__ == "__main__":
    print("\n=== Testing RAG Service ===\n")
    
    # Initialize dependencies
    print("Step 1: Loading dependencies...")
    pipeline = IngestionPipeline(
        collection_name="partselect_test",
        persist_directory="tests/test_data/vector_store_test"
    )
    
    llm = LLMService(model="google/gemma-3-27b-it:free")
    
    # Initialize RAG service
    print("\nStep 2: Initializing RAG service...")
    rag = RAGService(ingestion_pipeline=pipeline, llm_service=llm)
    
    # Health check
    print("\nStep 3: Health check...")
    health = rag.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Docs in store: {health.get('vector_docs', 0)}")
    
    if health['status_code'] != 200:
        print(f"\n❌ System not healthy: {health.get('reason')}")
        print("   Make sure to run ingestion pipeline first!")
        exit(1)
    
    # Test query
    print("\nStep 4: Test RAG query...")
    result = rag.query("What should I do if my ice maker isn't working?", k=3)
    
    if result['status_code'] == 200:
        print(f"\n✅ Query succeeded!")
        print(f"   Answer: {result['answer'][:150]}...")
        print(f"   Sources: {len(result['sources'])}")
        print(f"   Tokens: {result['metadata']['tokens_used']}")
        print(f"   Time: {result['metadata']['response_time_seconds']}s")
        
        # Show sources
        print(f"\n   Sources retrieved:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"     {i}. Type: {source['type']}, Score: {source['score']:.3f}")
    else:
        print(f"\n❌ Query failed: {result.get('message')}")
    
    # Stats
    print("\nStep 5: Service stats...")
    stats = rag.get_stats()
    print(f"   Queries processed: {stats['queries_processed']}")
    print(f"   Avg response time: {stats['average_response_time']}s")
    print(f"   Vector store docs: {stats['vector_store_docs']}")
    print(f"   LLM model: {stats['llm_model']}")
    
    print("\n=== RAG Service Test Complete ===\n")