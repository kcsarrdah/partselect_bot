# Query Cache System

## Overview

The PartSelect RAG backend now includes a **hybrid query caching system** that significantly reduces LLM costs and improves response times by caching previously answered questions.

## Architecture

### Two-Layer Caching Strategy

**Layer 1: Exact Hash Match (0.001s)**
- Fast MD5 hash lookup in SQLite
- Catches identical queries (case-insensitive)
- Instant response from cache

**Layer 2: Semantic Similarity (0.1s)**
- Embedding-based similarity search
- Catches paraphrased questions
- Configurable threshold (default: 0.95)

## How It Works

```
User Query: "my dishwasher is not working"
    ↓
Layer 1: Hash lookup (0.001s)
    ↓
    ├─ Exact match? → Return cached response ✅
    └─ No match
        ↓
    Layer 2: Semantic search (0.1s)
        ↓
        ├─ Similar query found (>0.95)? → Return cached response ✅
        └─ No similar query
            ↓
        Run full RAG pipeline (3-6s)
            ↓
        Cache result for future queries
            ↓
        Return response
```

## Features

### Automatic Caching
- Every RAG response is automatically cached
- No manual intervention required
- Transparent to API consumers

### Smart Matching
```python
# Exact matches (Layer 1)
"my dishwasher is not working" == "my dishwasher is not working" ✅
"My Dishwasher Is Not Working" == "my dishwasher is not working" ✅

# Semantic matches (Layer 2)
"dishwasher won't start" ≈ "my dishwasher is not working" (similarity: 0.97) ✅
"refrigerator broken" ≈ "my dishwasher is not working" (similarity: 0.42) ❌
```

### Analytics
- Track cache hits vs misses
- Calculate tokens and cost saved
- Identify popular queries
- Monitor cache performance

## API Integration

### Cache Metadata in Responses

When a query hits the cache, the response includes:

```json
{
  "status_code": 200,
  "status": "success",
  "query": "my dishwasher is not working",
  "answer": "Based on the symptoms...",
  "sources": [...],
  "metadata": {
    "tokens_used": 747,
    "cache_response_time_seconds": 0.002,
    "model": "google/gemma-3-27b-it:free"
  },
  "cached": true,
  "cache_type": "exact",
  "access_count": 5
}
```

### Stats Endpoint

`GET /api/stats` now includes cache statistics:

```json
{
  "queries_processed": 150,
  "average_response_time": 1.8,
  "vector_store_docs": 224,
  "llm_model": "google/gemma-3-27b-it:free",
  "default_k": 5,
  "cache_stats": {
    "total_cached_queries": 45,
    "total_cache_hits": 150,
    "cache_hit_rate": 0.70,
    "estimated_tokens_saved": 52000,
    "estimated_cost_saved_usd": 0.104,
    "popular_queries": [
      {"query": "how to fix ice maker", "hits": 12},
      {"query": "dishwasher not draining", "hits": 8}
    ],
    "similarity_threshold": 0.95
  }
}
```

## Database

### Location
`data/query_cache.db` (SQLite)

### Schema
```sql
CREATE TABLE query_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding BLOB NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,
    metadata TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    response_time REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 1
);
```

## Configuration

### Enable/Disable Cache

```python
# In main.py or rag_service initialization
rag_service = RAGService(
    ingestion_pipeline=pipeline,
    llm_service=llm,
    enable_cache=True  # Set to False to disable
)
```

### Adjust Similarity Threshold

```python
# In services/query_cache.py __init__
cache = QueryCache(
    similarity_threshold=0.95  # Higher = stricter matching
)
```

## Performance Impact

| Scenario | Without Cache | With Cache (Hit) | Savings |
|----------|---------------|------------------|---------|
| **Response Time** | 3-6 seconds | 0.002 seconds | **99.9%** |
| **Tokens Used** | 500-1500 | 0 | **100%** |
| **API Calls** | 1 per query | 0 | **100%** |
| **Cost** | ~$0.001/query | $0 | **100%** |

### Real-World Impact
- **100 queries, 30% cache hit rate:**
  - Tokens saved: ~15,000
  - Time saved: ~90 seconds
  - Cost saved: ~$0.03

- **1000 queries, 60% cache hit rate:**
  - Tokens saved: ~450,000
  - Time saved: ~30 minutes
  - Cost saved: ~$0.90

## Testing

### Manual Test
```bash
# Start backend
cd backend
python3 -m uvicorn main:app --reload

# Query 1 (cache miss)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "how to fix ice maker"}'
# Response time: ~4s

# Query 2 (exact cache hit)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "how to fix ice maker"}'
# Response time: ~0.002s ✅

# Query 3 (semantic cache hit)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "ice maker not working"}'
# Response time: ~0.1s ✅
```

### Check Stats
```bash
curl http://localhost:8000/api/stats
```

## Logging

Cache operations are logged for debugging:

```
2024-11-16 20:15:30 | INFO | services.rag_service | 🔍 Checking cache...
2024-11-16 20:15:30 | INFO | services.query_cache | ✓ Layer 1: Exact hash match found (accessed 5 times)
2024-11-16 20:15:30 | INFO | services.rag_service | ✓ Cache hit! (saved 747 tokens, exact match)
```

## Future Enhancements

1. **TTL (Time-To-Live)**: Auto-expire old cache entries
2. **Cache Invalidation**: Manual refresh for outdated answers
3. **Distributed Cache**: Redis for multi-instance deployments
4. **A/B Testing**: Compare cached vs fresh responses
5. **Query Normalization**: Better handling of typos and variations

## Cost Savings Calculator

Estimated savings for Deepseek API ($0.002 per 1K tokens):

| Daily Queries | Cache Hit Rate | Monthly Savings |
|---------------|----------------|-----------------|
| 100 | 30% | $0.90 |
| 500 | 50% | $15.00 |
| 1000 | 60% | $54.00 |
| 5000 | 70% | $315.00 |

*Based on average 750 tokens per query*

## Summary

✅ **Fast**: 99.9% faster for cached queries  
✅ **Cost-effective**: 100% token savings on cache hits  
✅ **Smart**: Semantic matching catches paraphrased questions  
✅ **Transparent**: Works automatically without code changes  
✅ **Observable**: Full logging and analytics  

The query cache is production-ready and will significantly reduce operational costs! 🚀

