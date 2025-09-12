"""
Cache Management System for K-POP Dashboard
==========================================

Advanced caching strategies for real-time data processing optimization:
- Multi-layer caching (Memory, Redis, Database)
- Cache invalidation and consistency management
- Performance monitoring and metrics
- Intelligent cache warming and prefetching
- TTL-based expiration policies

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
import redis
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, OrderedDict
import weakref
import pickle
import zlib
from functools import wraps, lru_cache

from ..config import Config

# Configure module logger
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache level hierarchy."""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"

class CacheStrategy(Enum):
    """Cache strategy types."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    READ_THROUGH = "read_through"

@dataclass
class CacheConfig:
    """Configuration for cache settings."""
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    redis_ttl: int = 3600  # 1 hour
    memory_ttl: int = 300  # 5 minutes
    compression_threshold: int = 1024  # Compress data > 1KB
    enable_compression: bool = True
    enable_monitoring: bool = True
    prefetch_enabled: bool = True
    
@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total * 100) if total > 0 else 0.0

class MemoryCache:
    """Thread-safe in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items to store
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key in self._cache:
                # Check TTL
                if self._is_expired(key):
                    self._remove(key)
                    self._stats.misses += 1
                    return None
                
                # Move to end (mark as recently used)
                self._cache.move_to_end(key)
                self._stats.hits += 1
                self._stats.update_hit_rate()
                return self._cache[key]
            
            self._stats.misses += 1
            self._stats.update_hit_rate()
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                self._remove(key)
            
            # Check capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self._cache[key] = value
            self._timestamps[key] = time.time() + (ttl or self.ttl)
            self._stats.size = len(self._cache)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        return key in self._timestamps and time.time() > self._timestamps[key]
    
    def _remove(self, key: str):
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            if key in self._timestamps:
                del self._timestamps[key]
            self._stats.evictions += 1

class CacheManager:
    """
    Multi-layer cache management system.
    
    Provides intelligent caching with multiple tiers:
    1. Memory cache (fastest, smallest capacity)
    2. Redis cache (fast, medium capacity)
    3. Database cache (slower, large capacity)
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration settings
        """
        self.config = config or CacheConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize cache layers
        self._init_memory_cache()
        self._init_redis_cache()
        
        # Cache statistics
        self._stats = {
            CacheLevel.MEMORY: CacheStats(),
            CacheLevel.REDIS: CacheStats(),
            CacheLevel.DATABASE: CacheStats()
        }
        
        # Performance monitoring
        self._response_times = defaultdict(list)
        self._monitoring_enabled = self.config.enable_monitoring
        
        # Cache warming thread
        self._warming_thread = None
        self._stop_warming = threading.Event()
        
        if self.config.prefetch_enabled:
            self._start_cache_warming()
    
    def _init_memory_cache(self):
        """Initialize memory cache."""
        max_items = self.config.max_memory_size // 1024  # Rough estimate
        self._memory_cache = MemoryCache(max_items, self.config.memory_ttl)
    
    def _init_redis_cache(self):
        """Initialize Redis cache connection."""
        try:
            redis_config = getattr(Config, 'REDIS_CONFIG', {})
            if not redis_config:
                # Default Redis configuration
                redis_config = {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'decode_responses': False  # We'll handle encoding/decoding
                }
            
            self._redis_client = redis.Redis(**redis_config)
            self._redis_client.ping()  # Test connection
            self.logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Redis not available, disabling Redis cache: {e}")
            self._redis_client = None
    
    def get(self, key: str, fetch_function: Optional[Callable] = None) -> Optional[Any]:
        """
        Get value from multi-layer cache.
        
        Args:
            key: Cache key
            fetch_function: Function to fetch data on cache miss
            
        Returns:
            Cached value or None
        """
        start_time = time.time()
        
        try:
            # Try memory cache first
            value = self._get_from_memory(key)
            if value is not None:
                self._record_hit(CacheLevel.MEMORY, start_time)
                return value
            
            # Try Redis cache
            value = self._get_from_redis(key)
            if value is not None:
                # Store in memory cache for faster access
                self._set_in_memory(key, value)
                self._record_hit(CacheLevel.REDIS, start_time)
                return value
            
            # Cache miss - fetch from source if function provided
            if fetch_function:
                value = fetch_function()
                if value is not None:
                    # Store in all cache levels
                    self.set(key, value)
                    self._record_hit(CacheLevel.DATABASE, start_time)
                    return value
            
            # Complete miss
            self._record_miss(start_time)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            self._record_miss(start_time)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in multi-layer cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live override
            
        Returns:
            Success status
        """
        try:
            # Set in memory cache
            memory_ttl = ttl or self.config.memory_ttl
            self._set_in_memory(key, value, memory_ttl)
            
            # Set in Redis cache
            redis_ttl = ttl or self.config.redis_ttl
            self._set_in_redis(key, value, redis_ttl)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache levels."""
        success = True
        
        try:
            # Delete from memory
            self._memory_cache.delete(key)
            
            # Delete from Redis
            if self._redis_client:
                self._redis_client.delete(key)
                
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            success = False
        
        return success
    
    def clear_all(self):
        """Clear all cache levels."""
        try:
            self._memory_cache.clear()
            
            if self._redis_client:
                self._redis_client.flushdb()
                
        except Exception as e:
            self.logger.error(f"Error clearing caches: {e}")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        try:
            if self._redis_client:
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
                    self.logger.info(f"Invalidated {len(keys)} keys matching pattern: {pattern}")
                    
        except Exception as e:
            self.logger.error(f"Error invalidating pattern {pattern}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {}
        
        # Memory cache stats
        memory_stats = self._memory_cache.get_stats()
        stats['memory'] = {
            'hits': memory_stats.hits,
            'misses': memory_stats.misses,
            'hit_rate': memory_stats.hit_rate,
            'size': memory_stats.size,
            'evictions': memory_stats.evictions
        }
        
        # Redis stats
        if self._redis_client:
            try:
                redis_info = self._redis_client.info()
                stats['redis'] = {
                    'hits': redis_info.get('keyspace_hits', 0),
                    'misses': redis_info.get('keyspace_misses', 0),
                    'used_memory': redis_info.get('used_memory', 0),
                    'connected_clients': redis_info.get('connected_clients', 0)
                }
            except Exception as e:
                self.logger.error(f"Error getting Redis stats: {e}")
                stats['redis'] = {'error': str(e)}
        else:
            stats['redis'] = {'status': 'disabled'}
        
        # Performance stats
        stats['performance'] = self._get_performance_stats()
        
        return stats
    
    # Cache warming and prefetching
    
    def warm_cache(self, warming_functions: Dict[str, Callable]):
        """
        Warm cache with commonly accessed data.
        
        Args:
            warming_functions: Dictionary of cache keys to functions that fetch data
        """
        for key, func in warming_functions.items():
            try:
                if not self.get(key):  # Only warm if not already cached
                    value = func()
                    if value is not None:
                        self.set(key, value)
                        self.logger.debug(f"Warmed cache key: {key}")
            except Exception as e:
                self.logger.error(f"Error warming cache key {key}: {e}")
    
    def _start_cache_warming(self):
        """Start background cache warming thread."""
        if self._warming_thread is None or not self._warming_thread.is_alive():
            self._warming_thread = threading.Thread(
                target=self._cache_warming_worker,
                daemon=True
            )
            self._warming_thread.start()
    
    def _cache_warming_worker(self):
        """Background worker for cache warming."""
        while not self._stop_warming.is_set():
            try:
                # Warm commonly accessed data
                warming_functions = {
                    'dashboard:summary': self._fetch_dashboard_summary,
                    'artists:active': self._fetch_active_artists,
                    'metrics:latest': self._fetch_latest_metrics
                }
                
                self.warm_cache(warming_functions)
                
                # Wait before next warming cycle
                self._stop_warming.wait(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cache warming worker: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def stop_cache_warming(self):
        """Stop cache warming thread."""
        self._stop_warming.set()
        if self._warming_thread and self._warming_thread.is_alive():
            self._warming_thread.join(timeout=5)
    
    # Private helper methods
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        return self._memory_cache.get(key)
    
    def _set_in_memory(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in memory cache."""
        self._memory_cache.set(key, value, ttl)
    
    def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._redis_client:
            return None
        
        try:
            data = self._redis_client.get(key)
            if data:
                return self._deserialize(data)
        except Exception as e:
            self.logger.error(f"Error getting from Redis: {e}")
        
        return None
    
    def _set_in_redis(self, key: str, value: Any, ttl: int):
        """Set value in Redis cache."""
        if not self._redis_client:
            return
        
        try:
            serialized_data = self._serialize(value)
            self._redis_client.setex(key, ttl, serialized_data)
        except Exception as e:
            self.logger.error(f"Error setting in Redis: {e}")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Convert to JSON first for better compatibility
            json_str = json.dumps(value, default=str, ensure_ascii=False)
            data = json_str.encode('utf-8')
            
            # Compress if enabled and data is large enough
            if (self.config.enable_compression and 
                len(data) > self.config.compression_threshold):
                data = zlib.compress(data)
                data = b'COMPRESSED:' + data
            
            return data
            
        except Exception as e:
            # Fallback to pickle
            self.logger.warning(f"JSON serialization failed, using pickle: {e}")
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Check if compressed
            if data.startswith(b'COMPRESSED:'):
                data = zlib.decompress(data[11:])  # Remove 'COMPRESSED:' prefix
            
            # Try JSON first
            try:
                json_str = data.decode('utf-8')
                return json.loads(json_str)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Fallback to pickle
                return pickle.loads(data)
                
        except Exception as e:
            self.logger.error(f"Error deserializing data: {e}")
            return None
    
    def _record_hit(self, level: CacheLevel, start_time: float):
        """Record cache hit metrics."""
        if self._monitoring_enabled:
            response_time = time.time() - start_time
            self._response_times[level].append(response_time)
            
            # Keep only recent response times
            if len(self._response_times[level]) > 1000:
                self._response_times[level] = self._response_times[level][-500:]
            
            self._stats[level].hits += 1
            self._stats[level].update_hit_rate()
    
    def _record_miss(self, start_time: float):
        """Record cache miss metrics."""
        if self._monitoring_enabled:
            response_time = time.time() - start_time
            
            for level in CacheLevel:
                self._stats[level].misses += 1
                self._stats[level].update_hit_rate()
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        perf_stats = {}
        
        for level in CacheLevel:
            response_times = self._response_times[level]
            if response_times:
                perf_stats[level.value] = {
                    'avg_response_time': sum(response_times) / len(response_times),
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times),
                    'recent_samples': len(response_times)
                }
        
        return perf_stats
    
    # Cache warming data fetchers (placeholders)
    
    def _fetch_dashboard_summary(self) -> Dict[str, Any]:
        """Fetch dashboard summary data for warming."""
        # This would integrate with actual data fetching
        return {'summary': 'data'}
    
    def _fetch_active_artists(self) -> List[Dict[str, Any]]:
        """Fetch active artists data for warming."""
        # This would integrate with actual data fetching
        return [{'artist': 'data'}]
    
    def _fetch_latest_metrics(self) -> Dict[str, Any]:
        """Fetch latest metrics data for warming."""
        # This would integrate with actual data fetching
        return {'metrics': 'data'}

# Cache decorators for easy usage

def cached(cache_manager: CacheManager, ttl: int = 300, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        cache_manager: Cache manager instance
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            
            # Add args to key
            for arg in args:
                if isinstance(arg, (str, int, float)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
            
            # Add kwargs to key
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

def cache_invalidate(cache_manager: CacheManager, pattern: str):
    """
    Decorator for invalidating cache on function execution.
    
    Args:
        cache_manager: Cache manager instance
        pattern: Pattern for keys to invalidate
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            cache_manager.invalidate_pattern(pattern)
            return result
        
        return wrapper
    return decorator

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager