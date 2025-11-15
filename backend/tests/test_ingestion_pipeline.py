"""
Test/Demo script for Ingestion Pipeline
Tests the full RAG ingestion: Load → Chunk → Embed → Store → Search
"""

from services.ingestion_pipeline import IngestionPipeline


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Ingestion Pipeline - Full RAG Workflow")
    print("="*70 + "\n")
    
    # Initialize pipeline with test collection
    print("Step 1: Initializing pipeline...")
    pipeline = IngestionPipeline(
        collection_name="partselect_test",
        persist_directory="tests/test_data/vector_store_test"  # ✅ Good
    )
    
    # Test 1: Health check (before ingestion)
    print("\n" + "-"*70)
    print("Test 1: Health Check (Empty Collection)")
    print("-"*70)
    health = pipeline.health_check()
    print(f"Status Code: {health['status_code']}")
    print(f"Status: {health['status']}")
    print(f"Documents: {health.get('total_documents', 0)}")
    assert health['status_code'] == 200, "Health check should return 200"
    assert health['total_documents'] == 0, "Should start with 0 documents"
    print("✓ Test 1 passed")
    
    # Test 2: Get status (detailed)
    print("\n" + "-"*70)
    print("Test 2: Get Full Status")
    print("-"*70)
    status = pipeline.get_status()
    print(f"Status Code: {status['status_code']}")
    print(f"Collection: {status.get('collection_name')}")
    print(f"Model: {status.get('embedding_model')}")
    print(f"Is Empty: {status.get('is_empty')}")
    assert status['status_code'] == 200, "Status should return 200"
    assert status['is_empty'] == True, "Collection should be empty"
    print("✓ Test 2 passed")
    
    # Test 3: Search on empty collection (should return 404)
    print("\n" + "-"*70)
    print("Test 3: Search on Empty Collection (Should Fail)")
    print("-"*70)
    search_result = pipeline.search("ice maker", k=3)
    print(f"Status Code: {search_result['status_code']}")
    print(f"Message: {search_result.get('message')}")
    assert search_result['status_code'] == 404, "Should return 404 for empty collection"
    print("✓ Test 3 passed")
    
    # Test 4: Run the full pipeline
    print("\n" + "-"*70)
    print("Test 4: Run Full Ingestion Pipeline")
    print("-"*70)
    result = pipeline.run_pipeline(
        data_dir="tests/fixtures",  # ✅ Using test CSV files
        batch_size=32,
        force_rebuild=True
    )
    print(f"\nPipeline Result:")
    print(f"  Status Code: {result['status_code']}")
    print(f"  Documents Loaded: {result.get('documents_loaded', 0)}")
    print(f"  Documents Chunked: {result.get('documents_chunked', 0)}")
    print(f"  Documents Stored: {result.get('documents_stored', 0)}")
    print(f"  Total in Collection: {result.get('total_in_collection', 0)}")
    print(f"  Time Taken: {result.get('time_taken_seconds', 0)} seconds")
    
    assert result['status_code'] == 200, "Pipeline should succeed"
    assert result['documents_loaded'] > 0, "Should load documents"
    assert result['total_in_collection'] > 0, "Should store documents"
    print("✓ Test 4 passed")
    
    # Test 5: Health check after ingestion
    print("\n" + "-"*70)
    print("Test 5: Health Check (After Ingestion)")
    print("-"*70)
    health = pipeline.health_check()
    print(f"Status Code: {health['status_code']}")
    print(f"Documents: {health.get('total_documents', 0)}")
    assert health['status_code'] == 200, "Health check should return 200"
    assert health['total_documents'] > 0, "Should have documents now"
    print("✓ Test 5 passed")
    
    # Test 6: Search - General query
    print("\n" + "-"*70)
    print("Test 6: Search - General Query")
    print("-"*70)
    queries = [
        "ice maker not working",
        "dishwasher parts",
        "how to install refrigerator"
    ]
    
    for query in queries:
        result = pipeline.search(query, k=3)
        print(f"\nQuery: '{query}'")
        print(f"  Status: {result['status_code']}")
        print(f"  Results: {result.get('num_results', 0)}")
        
        if result['status_code'] == 200 and result.get('results'):
            for i, res in enumerate(result['results'][:2], 1):  # Show first 2
                print(f"  {i}. Score: {res['score']:.4f}")
                print(f"     Type: {res['metadata'].get('type', 'unknown')}")
                print(f"     Content: {res['content'][:60]}...")
        
        assert result['status_code'] == 200, f"Search for '{query}' should succeed"
        assert result['num_results'] > 0, f"Should find results for '{query}'"
    
    print("\n✓ Test 6 passed")
    
    # Test 7: Search by type (parts only)
    print("\n" + "-"*70)
    print("Test 7: Search by Type - Parts Only")
    print("-"*70)
    result = pipeline.search_by_type("refrigerator", doc_type="part", k=3)
    print(f"Query: 'refrigerator' (parts only)")
    print(f"  Status: {result['status_code']}")
    print(f"  Results: {result.get('num_results', 0)}")
    
    if result['status_code'] == 200 and result.get('results'):
        for i, res in enumerate(result['results'], 1):
            print(f"  {i}. Type: {res['metadata'].get('type')}")
            print(f"     Part ID: {res['metadata'].get('part_id', 'N/A')}")
            print(f"     Score: {res['score']:.4f}")
            # Verify all results are parts
            assert res['metadata'].get('type') == 'part', "Should only return parts"
    
    print("✓ Test 7 passed")
    
    # Test 8: Get final stats
    print("\n" + "-"*70)
    print("Test 8: Final Collection Stats")
    print("-"*70)
    status = pipeline.get_status()
    print(f"Collection: {status.get('collection_name')}")
    print(f"Total Documents: {status.get('total_documents')}")
    print(f"Document Types: {status.get('document_types')}")
    print(f"Metadata Keys: {status.get('metadata_keys')}")
    print("✓ Test 8 passed")
    
    # Summary
    print("\n" + "="*70)
    print("✅ All Tests Passed!")
    print("="*70)
    print("\nIngestion Pipeline is working correctly:")
    print("  ✓ ChromaDB initialization")
    print("  ✓ Health checks and status")
    print("  ✓ Full pipeline execution (load → chunk → embed → store)")
    print("  ✓ Similarity search")
    print("  ✓ Filtered search by document type")
    print("\n" + "="*70 + "\n")