"""
Query Cache System for FRC RAG
Implements multi-level caching to speed up similar queries
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class QueryCache:
    """
    Multi-level cache for RAG queries with semantic similarity matching
    
    Features:
    - Exact query matching (hash-based)
    - Semantic similarity matching (embedding-based)
    - LRU eviction policy
    - Configurable TTL
    - Hit/miss statistics
    """
    
    def __init__(
        self, 
        max_size: int = 1000,
        similarity_threshold: float = 0.92,
        ttl_seconds: Optional[int] = 3600,
        enable_semantic_cache: bool = True
    ):
        """
        Initialize the query cache
        
        Args:
            max_size: Maximum number of cached entries (LRU eviction when exceeded)
            similarity_threshold: Cosine similarity threshold for semantic cache hits (0.9-0.95 recommended)
            ttl_seconds: Time-to-live for cache entries in seconds (None for no expiration)
            enable_semantic_cache: Enable semantic similarity matching (slower but better recall)
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.enable_semantic_cache = enable_semantic_cache
        
        # Exact match cache (hash -> cached response)
        self.exact_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Semantic cache (query embedding -> cached response)
        # Stores: {query_hash: (embedding, response_data, timestamp, original_query)}
        self.semantic_cache: OrderedDict[str, Tuple[np.ndarray, Dict[str, Any], float, str]] = OrderedDict()
        
        # Statistics
        self.stats = {
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'total_queries': 0,
            'evictions': 0,
            'ttl_expirations': 0
        }
    
    def _generate_query_hash(self, query: str, k: int = 10) -> str:
        """Generate a hash for exact query matching"""
        # Normalize query: lowercase, strip whitespace
        normalized = query.lower().strip()
        # Include k parameter in hash
        cache_key = f"{normalized}|k={k}"
        return hashlib.sha256(cache_key.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self.ttl_seconds
    
    def _evict_oldest(self, cache: OrderedDict):
        """Evict the oldest entry from a cache"""
        if cache:
            cache.popitem(last=False)  # Remove oldest (FIFO)
            self.stats['evictions'] += 1
    
    def _ensure_cache_size(self):
        """Ensure cache doesn't exceed max_size"""
        total_entries = len(self.exact_cache) + len(self.semantic_cache)
        
        while total_entries > self.max_size:
            # Evict from the larger cache first
            if len(self.exact_cache) >= len(self.semantic_cache):
                self._evict_oldest(self.exact_cache)
            else:
                self._evict_oldest(self.semantic_cache)
            total_entries -= 1
    
    def get(
        self, 
        query: str, 
        k: int = 10,
        query_embedding: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a query
        
        Args:
            query: The user query
            k: Number of results requested
            query_embedding: Pre-computed query embedding (for semantic matching)
        
        Returns:
            Cached response dict or None if not found
        """
        self.stats['total_queries'] += 1
        query_hash = self._generate_query_hash(query, k)
        
        # 1. Try exact match cache first (fastest)
        if query_hash in self.exact_cache:
            entry = self.exact_cache[query_hash]
            
            # Check expiration
            if self._is_expired(entry['timestamp']):
                del self.exact_cache[query_hash]
                self.stats['ttl_expirations'] += 1
            else:
                # Move to end (mark as recently used)
                self.exact_cache.move_to_end(query_hash)
                self.stats['exact_hits'] += 1
                
                # Return a copy to prevent mutation
                return self._create_cache_response(entry['data'], 'exact')
        
        # 2. Try semantic similarity cache (if enabled and embedding provided)
        if self.enable_semantic_cache and query_embedding is not None:
            best_match = None
            best_similarity = 0.0
            best_hash = None
            
            # Find most similar cached query
            for cached_hash, (cached_embedding, cached_data, timestamp, original_query) in self.semantic_cache.items():
                # Check expiration first
                if self._is_expired(timestamp):
                    continue
                
                # Compute cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_data
                    best_hash = cached_hash
            
            # Check if similarity exceeds threshold
            if best_similarity >= self.similarity_threshold:
                # Move to end (mark as recently used)
                self.semantic_cache.move_to_end(best_hash)
                self.stats['semantic_hits'] += 1
                
                return self._create_cache_response(best_match, 'semantic', best_similarity)
        
        # 3. Cache miss
        self.stats['misses'] += 1
        return None
    
    def set(
        self,
        query: str,
        response_data: Dict[str, Any],
        k: int = 10,
        query_embedding: Optional[np.ndarray] = None
    ):
        """
        Cache a query response
        
        Args:
            query: The user query
            response_data: The response data to cache
            k: Number of results requested
            query_embedding: Query embedding (for semantic cache)
        """
        query_hash = self._generate_query_hash(query, k)
        timestamp = time.time()
        
        # Store in exact match cache
        self.exact_cache[query_hash] = {
            'data': response_data,
            'timestamp': timestamp,
            'query': query
        }
        
        # Move to end (mark as most recent)
        self.exact_cache.move_to_end(query_hash)
        
        # Store in semantic cache if embedding provided
        if self.enable_semantic_cache and query_embedding is not None:
            self.semantic_cache[query_hash] = (
                query_embedding,
                response_data,
                timestamp,
                query
            )
            self.semantic_cache.move_to_end(query_hash)
        
        # Ensure we don't exceed max size
        self._ensure_cache_size()
    
    def _create_cache_response(
        self, 
        data: Dict[str, Any], 
        cache_type: str,
        similarity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a response with cache metadata"""
        response = data.copy()
        response['_cache_hit'] = True
        response['_cache_type'] = cache_type
        
        if similarity is not None:
            response['_cache_similarity'] = float(similarity)
        
        return response
    
    def clear(self):
        """Clear all cached entries"""
        self.exact_cache.clear()
        self.semantic_cache.clear()
    
    def remove_expired(self):
        """Remove all expired entries"""
        if self.ttl_seconds is None:
            return
        
        current_time = time.time()
        
        # Remove from exact cache
        expired_exact = [
            key for key, value in self.exact_cache.items()
            if (current_time - value['timestamp']) > self.ttl_seconds
        ]
        for key in expired_exact:
            del self.exact_cache[key]
            self.stats['ttl_expirations'] += 1
        
        # Remove from semantic cache
        expired_semantic = [
            key for key, (_, _, timestamp, _) in self.semantic_cache.items()
            if (current_time - timestamp) > self.ttl_seconds
        ]
        for key in expired_semantic:
            del self.semantic_cache[key]
            self.stats['ttl_expirations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.stats['exact_hits'] + self.stats['semantic_hits']
        hit_rate = (total_hits / self.stats['total_queries'] * 100) if self.stats['total_queries'] > 0 else 0
        
        return {
            **self.stats,
            'total_hits': total_hits,
            'hit_rate_percent': round(hit_rate, 2),
            'exact_cache_size': len(self.exact_cache),
            'semantic_cache_size': len(self.semantic_cache),
            'total_cache_size': len(self.exact_cache) + len(self.semantic_cache)
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'total_queries': 0,
            'evictions': 0,
            'ttl_expirations': 0
        }


class ChunkCache:
    """
    Cache for vector database search results (chunks)
    Caches the actual retrieved chunks to avoid repeated similarity searches
    """
    
    def __init__(self, max_size: int = 500, ttl_seconds: Optional[int] = 1800):
        """
        Initialize chunk cache
        
        Args:
            max_size: Maximum number of cached chunk sets
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Cache: {embedding_hash: (chunks, timestamp)}
        self.cache: OrderedDict[str, Tuple[List, float]] = OrderedDict()
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }
    
    def _generate_embedding_hash(self, embedding: np.ndarray, k: int) -> str:
        """Generate hash for embedding + k parameter"""
        # Use first 100 dimensions + k for hash (balance between uniqueness and performance)
        hash_input = np.concatenate([embedding[:100], [k]]).tobytes()
        return hashlib.sha256(hash_input).hexdigest()
    
    def get(self, embedding: np.ndarray, k: int) -> Optional[List]:
        """Get cached chunks for an embedding"""
        cache_key = self._generate_embedding_hash(embedding, k)
        
        if cache_key in self.cache:
            chunks, timestamp = self.cache[cache_key]
            
            # Check expiration
            if self.ttl_seconds and (time.time() - timestamp) > self.ttl_seconds:
                del self.cache[cache_key]
                self.stats['expirations'] += 1
                self.stats['misses'] += 1
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            self.stats['hits'] += 1
            return chunks
        
        self.stats['misses'] += 1
        return None
    
    def set(self, embedding: np.ndarray, k: int, chunks: List):
        """Cache chunks for an embedding"""
        cache_key = self._generate_embedding_hash(embedding, k)
        
        self.cache[cache_key] = (chunks, time.time())
        self.cache.move_to_end(cache_key)
        
        # Evict oldest if needed
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self.cache)
        }
