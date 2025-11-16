"""
Query Cache Service
===================
Hybrid caching system with exact hash matching and semantic similarity search.

Features:
- Layer 1: Fast exact match via MD5 hash lookup
- Layer 2: Semantic similarity search using embeddings
- Automatic caching of all RAG responses
- Cache analytics and statistics

"""

import sqlite3
import json
import hashlib
import pickle
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
from numpy import dot
from numpy.linalg import norm

from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.logger import setup_logger, log_success, log_warning, log_error


logger = setup_logger(__name__)


class QueryCache:
    """
    Hybrid query cache with exact and semantic matching.
    
    Usage:
        cache = QueryCache()
        
        # Check cache
        result = cache.get("my dishwasher is not working")
        if result:
            return result  # Cache hit!
        
        # ... run RAG pipeline ...
        
        # Store result
        cache.set(query, rag_response)
    """
    
    def __init__(
        self,
        db_path: str = "data/query_cache.db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.90
    ):
        """
        Initialize query cache.
        
        Args:
            db_path: Path to SQLite database file
            embedding_model: HuggingFace model for embeddings
            similarity_threshold: Minimum cosine similarity for semantic match (0-1)
        """
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize embeddings (same model as RAG pipeline)
        logger.info(f"📦 Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize database
        self._init_db()
        log_success(logger, f"Cache initialized at {db_path}")
    
    def _init_db(self):
        """Create cache database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
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
            )
        """)
        
        # Indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_hash 
            ON query_cache(query_hash)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON query_cache(created_at DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def _hash_query(self, query: str) -> str:
        """
        Generate MD5 hash of normalized query for exact matching.
        
        Args:
            query: User query string
            
        Returns:
            MD5 hash as hex string
        """
        # Normalize: lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(query.lower().strip().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        return float(dot(vec1, vec2) / (norm(vec1) * norm(vec2)))
    
    def _get_exact(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Layer 1: Check for exact hash match.
        
        Args:
            query_hash: MD5 hash of query
            
        Returns:
            Cached response or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query_text, answer, sources, metadata, access_count
            FROM query_cache
            WHERE query_hash = ?
        """, (query_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            query_text, answer, sources, metadata, access_count = result
            logger.info(f"✓ Layer 1: Exact hash match found (accessed {access_count} times)")
            return {
                "status_code": 200,  # ADD THIS
                "status": "success",  # ADD THIS
                "query": query_text,
                "answer": answer,
                "sources": json.loads(sources),
                "metadata": json.loads(metadata),
                "cached": True,
                "cache_type": "exact",
                "access_count": access_count
            }
        
        return None
    
    def _get_similar(self, query: str, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Layer 2: Check for semantically similar queries.
        
        Args:
            query: User query string
            query_embedding: Embedding vector of query
            
        Returns:
            Cached response if similar query found, else None
        """
        logger.info("🔍 Layer 2: Searching for similar queries...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all cached queries with embeddings
        cursor.execute("""
            SELECT id, query_text, query_embedding, answer, sources, metadata, access_count
            FROM query_cache
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            logger.info("No cached queries to compare against")
            return None
        
        # Find most similar query
        best_match = None
        best_similarity = 0.0
        
        for row in results:
            cache_id, cached_query, embedding_blob, answer, sources, metadata, access_count = row
            
            # Deserialize embedding
            try:
                cached_embedding = pickle.loads(embedding_blob)
            except Exception as e:
                log_warning(logger, f"Failed to deserialize embedding for query {cache_id}: {e}")
                continue
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "id": cache_id,
                    "query": cached_query,
                    "answer": answer,
                    "sources": sources,
                    "metadata": metadata,
                    "access_count": access_count,
                    "similarity": similarity
                }
        
        # Check if best match exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            log_success(logger, f"Similar query found! Similarity: {best_similarity:.3f}")
            logger.info(f"  Original: '{best_match['query']}'")
            logger.info(f"  New:      '{query}'")
            
            return {
                "status_code": 200,  # ADD THIS
                "status": "success",  # ADD THIS
                "query": query,  # Return original query, not cached one
                "answer": best_match["answer"],
                "sources": json.loads(best_match["sources"]),
                "metadata": json.loads(best_match["metadata"]),
                "cached": True,
                "cache_type": "semantic",
                "similarity": best_similarity,
                "matched_query": best_match["query"],
                "access_count": best_match["access_count"]
            }
        
        if best_match:
            logger.info(f"Best similarity: {best_similarity:.3f} (below threshold {self.similarity_threshold})")
        
        return None
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query (tries exact then semantic matching).
        
        Args:
            query: User query string
            
        Returns:
            Cached response dict or None if no match found
        """
        query_hash = self._hash_query(query)
        
        # Layer 1: Try exact match first (fastest)
        cached = self._get_exact(query_hash)
        if cached:
            self._update_access_stats(query_hash)
            return cached
        
        # Layer 2: Try semantic similarity
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array(query_embedding)
        
        cached = self._get_similar(query, query_embedding_np)
        if cached:
            # Update stats for the matched query
            matched_hash = self._hash_query(cached['matched_query'])
            self._update_access_stats(matched_hash)
            return cached
        
        logger.info("❌ No cache match found")
        return None
    
    def set(self, query: str, response: Dict[str, Any]):
        """
        Cache a query-response pair.
        
        Args:
            query: User query string
            response: RAG response dict with answer, sources, metadata
        """
        query_hash = self._hash_query(query)
        
        # Generate embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array(query_embedding)
        embedding_blob = pickle.dumps(query_embedding_np)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO query_cache 
                (query_hash, query_text, query_embedding, answer, sources, metadata, 
                 tokens_used, response_time, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_hash,
                query,
                embedding_blob,
                response['answer'],
                json.dumps(response['sources']),
                json.dumps(response['metadata']),
                response['metadata'].get('tokens_used', 0),
                response['metadata'].get('response_time_seconds', 0.0),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()
            log_success(logger, f"Cached query: '{query[:50]}...'")
            
        except sqlite3.IntegrityError:
            # Query already cached (hash collision or duplicate)
            log_warning(logger, "Query already in cache")
        except Exception as e:
            log_error(logger, f"Failed to cache query: {e}")
        finally:
            conn.close()
    
    def _update_access_stats(self, query_hash: str):
        """Update access count and timestamp for cached query."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE query_cache
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE query_hash = ?
        """, (datetime.now().isoformat(), query_hash))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total cached queries
        cursor.execute("SELECT COUNT(*) FROM query_cache")
        total_cached = cursor.fetchone()[0]
        
        # Total cache hits (sum of access counts)
        cursor.execute("SELECT SUM(access_count) FROM query_cache")
        total_hits = cursor.fetchone()[0] or 0
        
        # Total tokens that would have been used
        cursor.execute("SELECT SUM(tokens_used) FROM query_cache")
        total_tokens = cursor.fetchone()[0] or 0
        
        # Calculate tokens saved (hits beyond initial query)
        tokens_saved = total_tokens * (total_hits - total_cached) if total_cached > 0 else 0
        
        # Estimate cost saved (rough estimate: $0.002 per 1K tokens for Deepseek)
        cost_saved = (tokens_saved / 1000) * 0.002
        
        # Cache hit rate
        cache_hit_rate = (total_hits / total_hits) if total_hits > 0 else 0
        
        # Most popular queries
        cursor.execute("""
            SELECT query_text, access_count
            FROM query_cache
            ORDER BY access_count DESC
            LIMIT 5
        """)
        popular_queries = [
            {"query": row[0], "hits": row[1]} 
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            "total_cached_queries": total_cached,
            "total_cache_hits": total_hits,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "estimated_tokens_saved": int(tokens_saved),
            "estimated_cost_saved_usd": round(cost_saved, 4),
            "popular_queries": popular_queries,
            "similarity_threshold": self.similarity_threshold
        }
    
    def get_conversation_history(self, limit: int = 20) -> List[Dict[str, str]]:
        """
        Retrieve the last N query-response pairs for conversation context.
        
        Args:
            limit: Number of recent interactions to retrieve (default 20)
            
        Returns:
            List of dicts with 'query' and 'answer' keys, ordered from oldest to newest
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the most recent interactions
            cursor.execute(f"""
                SELECT query_text, answer
                FROM query_cache
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Return in chronological order (oldest to newest for better context)
            history = [
                {"query": row[0], "answer": row[1]}
                for row in reversed(rows)
            ]
            
            logger.info(f"📚 Retrieved {len(history)} conversation history items")
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def clear(self):
        """Clear all cached queries (useful for testing)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM query_cache")
        conn.commit()
        conn.close()
        log_success(logger, "Cache cleared")

