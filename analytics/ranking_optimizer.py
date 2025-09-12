"""
Ranking System Performance Optimizer
====================================

High-performance optimization layer for the K-Pop ranking system with advanced
batch processing, intelligent caching, and parallel computation capabilities.

Features:
- Multi-level caching with TTL and invalidation strategies
- Batch processing optimization for large datasets (1000+ artists)
- Parallel ranking computation with configurable worker pools
- Memory-efficient streaming data processing
- Performance monitoring and metrics tracking
- Cache warming and precomputation strategies

Author: Backend Development Team
Date: 2025-09-08
Version: 2.0.0
"""

import asyncio
import hashlib
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Iterator
from enum import Enum
import logging
import threading
from collections import OrderedDict, defaultdict
import weakref
import psutil
import gc

# Import from existing modules
from .ranking_system import (
    GrowthRankingEngine, RankingCategory, RankingPeriod, DebutCohort,
    ArtistMetrics, RankingResult, CompositeIndex
)
from .growth_rate_calculator import MetricDataPoint

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for hierarchical caching strategy."""
    MEMORY = "memory"          # In-memory LRU cache
    COMPUTATION = "computation"  # Pre-computed ranking results
    INCREMENTAL = "incremental" # Delta-based updates
    PERSISTENT = "persistent"   # Disk-based cache (future)


class OptimizationStrategy(Enum):
    """Optimization strategies for different workload patterns."""
    REAL_TIME = "real_time"     # Low latency, small batches
    BATCH_HEAVY = "batch_heavy"  # High throughput, large batches
    MEMORY_EFFICIENT = "memory_efficient"  # Minimal memory footprint
    CPU_INTENSIVE = "cpu_intensive"  # Maximum parallel processing


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent cache management."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 3600  # 1 hour default
    computation_cost: float = 1.0  # Relative cost of recomputation
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time: float = 0.0
    peak_memory_usage: float = 0.0
    concurrent_requests: int = 0
    batch_processing_time: float = 0.0
    parallel_efficiency: float = 1.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class IntelligentCache:
    """
    Multi-level intelligent cache with advanced features.
    
    Features:
    - LRU eviction with cost-based prioritization
    - TTL-based expiration
    - Memory pressure adaptation
    - Cache warming and precomputation
    - Hierarchical cache levels
    """
    
    def __init__(self, 
                 max_memory_mb: int = 512,
                 default_ttl: int = 3600,
                 cleanup_interval: int = 300):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Cache storage by level
        self._caches: Dict[CacheLevel, OrderedDict] = {
            level: OrderedDict() for level in CacheLevel
        }
        
        # Cache metadata
        self._metadata: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._total_size = 0
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        
        # Background cleanup
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"IntelligentCache initialized: {max_memory_mb}MB capacity, {default_ttl}s TTL")
    
    def get(self, key: str, level: CacheLevel = CacheLevel.MEMORY) -> Optional[Any]:
        """Get value from cache with intelligent access tracking."""
        with self._lock:
            self.metrics.total_requests += 1
            
            cache = self._caches[level]
            
            if key in cache:
                entry = self._metadata[key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key, level)
                    self.metrics.cache_misses += 1
                    return None
                
                # Update access metadata and move to end (most recent)
                entry.touch()
                cache.move_to_end(key)
                
                self.metrics.cache_hits += 1
                logger.debug(f"Cache hit for key: {key[:50]}...")
                return entry.value
            
            self.metrics.cache_misses += 1
            return None
    
    def set(self, 
            key: str, 
            value: Any, 
            level: CacheLevel = CacheLevel.MEMORY,
            ttl: Optional[int] = None,
            computation_cost: float = 1.0) -> bool:
        """Set value in cache with intelligent eviction."""
        with self._lock:
            cache = self._caches[level]
            ttl = ttl or self.default_ttl
            
            # Calculate entry size
            try:
                size_bytes = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Check if we need to evict entries
            while self._total_size + size_bytes > self.max_memory_bytes and cache:
                self._evict_lru_entry(level)
            
            # Remove existing entry if present
            if key in cache:
                old_entry = self._metadata[key]
                self._total_size -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                computation_cost=computation_cost,
                size_bytes=size_bytes
            )
            
            # Store in cache
            cache[key] = value
            self._metadata[key] = entry
            self._total_size += size_bytes
            
            logger.debug(f"Cached key: {key[:50]}... (size: {size_bytes} bytes)")
            return True
    
    def invalidate(self, pattern: str = None, level: Optional[CacheLevel] = None):
        """Invalidate cache entries by pattern or level."""
        with self._lock:
            if level:
                # Invalidate specific level
                cache = self._caches[level]
                keys_to_remove = list(cache.keys())
                for key in keys_to_remove:
                    if pattern is None or pattern in key:
                        self._remove_entry(key, level)
            else:
                # Invalidate across all levels
                for cache_level in CacheLevel:
                    cache = self._caches[cache_level]
                    keys_to_remove = list(cache.keys())
                    for key in keys_to_remove:
                        if pattern is None or pattern in key:
                            self._remove_entry(key, cache_level)
        
        logger.info(f"Invalidated cache entries matching pattern: {pattern}")
    
    def warm_cache(self, 
                   data_source: Callable[[], List[ArtistMetrics]], 
                   warm_functions: List[Callable]) -> int:
        """Pre-warm cache with commonly requested data."""
        logger.info("Starting cache warming process...")
        warmed_count = 0
        
        try:
            # Get fresh data
            artists_data = data_source()
            
            # Execute warming functions
            for warm_func in warm_functions:
                try:
                    results = warm_func(artists_data)
                    if results:
                        warmed_count += len(results) if isinstance(results, (list, dict)) else 1
                except Exception as e:
                    logger.error(f"Cache warming function failed: {e}")
        
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
        
        logger.info(f"Cache warming completed: {warmed_count} entries warmed")
        return warmed_count
    
    def _evict_lru_entry(self, level: CacheLevel):
        """Evict least recently used entry with cost consideration."""
        cache = self._caches[level]
        
        if not cache:
            return
        
        # Find entry with lowest priority (LRU + cost consideration)
        best_key = None
        best_priority = float('inf')
        
        for key in cache:
            entry = self._metadata[key]
            # Priority = access_count * computation_cost / age_hours
            age_hours = max(1, (datetime.now() - entry.last_accessed).total_seconds() / 3600)
            priority = (entry.access_count * entry.computation_cost) / age_hours
            
            if priority < best_priority:
                best_priority = priority
                best_key = key
        
        if best_key:
            self._remove_entry(best_key, level)
    
    def _remove_entry(self, key: str, level: CacheLevel):
        """Remove entry from cache and metadata."""
        cache = self._caches[level]
        
        if key in cache:
            del cache[key]
            
        if key in self._metadata:
            entry = self._metadata[key]
            self._total_size -= entry.size_bytes
            del self._metadata[key]
    
    def _background_cleanup(self):
        """Background thread for cache maintenance."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                
                with self._lock:
                    # Remove expired entries
                    expired_keys = []
                    for key, entry in self._metadata.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        # Find which level the key is in
                        for level in CacheLevel:
                            if key in self._caches[level]:
                                self._remove_entry(key, level)
                                break
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                # Memory pressure check
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                if memory_usage > self.max_memory_bytes / 1024 / 1024 * 1.2:  # 20% over limit
                    logger.warning(f"Memory pressure detected: {memory_usage:.1f}MB, forcing cleanup")
                    self._force_cleanup()
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    def _force_cleanup(self):
        """Force cleanup under memory pressure."""
        with self._lock:
            # Aggressively evict entries
            target_size = self.max_memory_bytes * 0.7  # Target 70% capacity
            
            for level in CacheLevel:
                cache = self._caches[level]
                while self._total_size > target_size and cache:
                    self._evict_lru_entry(level)
            
            # Force garbage collection
            gc.collect()


class BatchRankingProcessor:
    """
    High-performance batch processor for large-scale ranking operations.
    
    Features:
    - Parallel processing with configurable worker pools
    - Memory-efficient streaming processing
    - Adaptive batch sizing based on system resources
    - Progress tracking and error handling
    - Automatic workload balancing
    """
    
    def __init__(self,
                 max_workers: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 memory_limit_mb: int = 1024,
                 strategy: OptimizationStrategy = OptimizationStrategy.BATCH_HEAVY):
        
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.memory_limit_mb = memory_limit_mb
        self.strategy = strategy
        
        # Adaptive batch sizing
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        if batch_size is None:
            if strategy == OptimizationStrategy.MEMORY_EFFICIENT:
                self.batch_size = min(100, max(10, int(available_memory_mb / 100)))
            elif strategy == OptimizationStrategy.REAL_TIME:
                self.batch_size = min(50, max(5, int(available_memory_mb / 200)))
            else:
                self.batch_size = min(500, max(50, int(available_memory_mb / 50)))
        else:
            self.batch_size = batch_size
        
        self.metrics = PerformanceMetrics()
        self._active_tasks = 0
        self._lock = threading.Lock()
        
        logger.info(f"BatchProcessor initialized: {self.max_workers} workers, "
                   f"batch size {self.batch_size}, strategy {strategy.value}")
    
    def process_ranking_batch(self,
                             artists_data: List[ArtistMetrics],
                             ranking_engine: GrowthRankingEngine,
                             operations: List[Dict[str, Any]],
                             progress_callback: Optional[Callable] = None) -> Dict[str, List[Any]]:
        """
        Process multiple ranking operations in parallel batches.
        
        Args:
            artists_data: Full artist dataset
            ranking_engine: Ranking engine instance
            operations: List of ranking operations to perform
            progress_callback: Optional progress reporting callback
            
        Returns:
            Dictionary of operation results
        """
        start_time = time.time()
        self.metrics.concurrent_requests += 1
        
        try:
            logger.info(f"Starting batch processing: {len(artists_data)} artists, "
                       f"{len(operations)} operations")
            
            results = {}
            
            if self.strategy == OptimizationStrategy.CPU_INTENSIVE:
                # Use process pool for CPU-intensive work
                results = self._process_with_multiprocessing(
                    artists_data, ranking_engine, operations, progress_callback
                )
            else:
                # Use thread pool for I/O and memory operations
                results = self._process_with_threading(
                    artists_data, ranking_engine, operations, progress_callback
                )
            
            processing_time = time.time() - start_time
            self.metrics.batch_processing_time = processing_time
            
            logger.info(f"Batch processing completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
        finally:
            self.metrics.concurrent_requests -= 1
    
    def _process_with_threading(self,
                               artists_data: List[ArtistMetrics],
                               ranking_engine: GrowthRankingEngine,
                               operations: List[Dict[str, Any]],
                               progress_callback: Optional[Callable]) -> Dict[str, List[Any]]:
        """Process operations using thread pool."""
        results = {}
        completed_operations = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all operations
            future_to_op = {}
            
            for i, operation in enumerate(operations):
                future = executor.submit(
                    self._execute_ranking_operation,
                    artists_data,
                    ranking_engine,
                    operation
                )
                future_to_op[future] = (i, operation)
            
            # Collect results
            for future in as_completed(future_to_op):
                op_index, operation = future_to_op[future]
                
                try:
                    result = future.result()
                    op_key = operation.get('key', f'operation_{op_index}')
                    results[op_key] = result
                    
                    completed_operations += 1
                    
                    if progress_callback:
                        progress_callback(completed_operations, len(operations))
                        
                except Exception as e:
                    logger.error(f"Operation {op_index} failed: {e}")
                    op_key = operation.get('key', f'operation_{op_index}')
                    results[op_key] = []
        
        return results
    
    def _execute_ranking_operation(self,
                                  artists_data: List[ArtistMetrics],
                                  ranking_engine: GrowthRankingEngine,
                                  operation: Dict[str, Any]) -> List[Any]:
        """Execute a single ranking operation."""
        op_type = operation.get('type')
        
        try:
            if op_type == 'platform_ranking':
                return ranking_engine.rank_by_platform(
                    artists_data=artists_data,
                    platform=operation['platform'],
                    metric_type=operation['metric_type'],
                    category=operation.get('category', RankingCategory.GROWTH_RATE),
                    period=operation.get('period', RankingPeriod.MONTHLY)
                )
            
            elif op_type == 'composite_index':
                return ranking_engine.calculate_composite_index(
                    artists_data=artists_data,
                    platforms=operation['platforms'],
                    weights=operation.get('weights'),
                    include_analysis=operation.get('include_analysis', True)
                )
            
            elif op_type == 'company_ranking':
                result = ranking_engine.rank_within_company(
                    artists_data=artists_data,
                    company_id=operation['company_id'],
                    platform=operation['platform'],
                    metric_type=operation['metric_type'],
                    category=operation.get('category', RankingCategory.COMPOSITE)
                )
                return result.get('rankings', [])
            
            elif op_type == 'cohort_ranking':
                result = ranking_engine.rank_debut_cohort(
                    artists_data=artists_data,
                    cohort=operation['cohort'],
                    platform=operation['platform'],
                    metric_type=operation['metric_type'],
                    category=operation.get('category', RankingCategory.GROWTH_RATE)
                )
                return result.get('rankings', [])
            
            else:
                logger.warning(f"Unknown operation type: {op_type}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to execute {op_type} operation: {e}")
            return []
    
    def stream_process_artists(self,
                              artist_iterator: Iterator[ArtistMetrics],
                              ranking_function: Callable,
                              batch_size: Optional[int] = None) -> Iterator[Any]:
        """
        Stream process artists in batches for memory efficiency.
        
        Args:
            artist_iterator: Iterator of artist data
            ranking_function: Function to apply to each batch
            batch_size: Override default batch size
            
        Yields:
            Processing results for each batch
        """
        batch_size = batch_size or self.batch_size
        current_batch = []
        
        for artist in artist_iterator:
            current_batch.append(artist)
            
            if len(current_batch) >= batch_size:
                # Process current batch
                try:
                    result = ranking_function(current_batch)
                    yield result
                    current_batch = []
                    
                    # Memory cleanup
                    if len(current_batch) % 10 == 0:  # Every 10 batches
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    current_batch = []  # Skip problematic batch
        
        # Process final batch
        if current_batch:
            try:
                result = ranking_function(current_batch)
                yield result
            except Exception as e:
                logger.error(f"Final batch processing error: {e}")


class OptimizedRankingEngine(GrowthRankingEngine):
    """
    Performance-optimized ranking engine with advanced caching and batch processing.
    
    Features:
    - Intelligent multi-level caching
    - Parallel batch processing
    - Memory-efficient operations
    - Performance monitoring
    - Cache warming strategies
    """
    
    def __init__(self,
                 cache_size_mb: int = 512,
                 max_workers: Optional[int] = None,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BATCH_HEAVY,
                 enable_cache_warming: bool = True,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # Initialize cache and batch processor
        self.cache = IntelligentCache(max_memory_mb=cache_size_mb)
        self.batch_processor = BatchRankingProcessor(
            max_workers=max_workers,
            strategy=optimization_strategy
        )
        self.optimization_strategy = optimization_strategy
        self.enable_cache_warming = enable_cache_warming
        
        # Performance monitoring
        self.performance_metrics = PerformanceMetrics()
        self._request_times = []
        
        logger.info(f"OptimizedRankingEngine initialized with {optimization_strategy.value} strategy")
    
    def rank_by_platform_optimized(self,
                                  artists_data: List[ArtistMetrics],
                                  platform: str,
                                  metric_type: str,
                                  category: RankingCategory = RankingCategory.GROWTH_RATE,
                                  period: RankingPeriod = RankingPeriod.MONTHLY,
                                  use_cache: bool = True,
                                  **kwargs) -> List[RankingResult]:
        """Optimized platform ranking with caching."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            'platform_ranking', 
            platform=platform,
            metric_type=metric_type,
            category=category.value,
            period=period.value,
            artist_count=len(artists_data),
            data_hash=self._hash_artists_data(artists_data)
        )
        
        # Try cache first
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.performance_metrics.cache_hits += 1
                logger.debug(f"Cache hit for platform ranking: {platform}/{metric_type}")
                return cached_result
        
        # Compute ranking
        result = super().rank_by_platform(
            artists_data, platform, metric_type, category, period, **kwargs
        )
        
        # Cache result with appropriate TTL
        if use_cache and result:
            ttl = self._calculate_ttl(category, len(artists_data))
            computation_cost = self._estimate_computation_cost('platform_ranking', len(artists_data))
            
            self.cache.set(
                cache_key,
                result,
                ttl=ttl,
                computation_cost=computation_cost
            )
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_performance_metrics(processing_time)
        
        return result
    
    def calculate_composite_index_optimized(self,
                                          artists_data: List[ArtistMetrics],
                                          platforms: List[str],
                                          weights: Optional[Dict[str, float]] = None,
                                          use_cache: bool = True,
                                          **kwargs) -> List[CompositeIndex]:
        """Optimized composite index calculation with caching."""
        start_time = time.time()
        
        # Generate cache key
        weights = weights or self.default_weights
        cache_key = self._generate_cache_key(
            'composite_index',
            platforms=sorted(platforms),
            weights=weights,
            artist_count=len(artists_data),
            data_hash=self._hash_artists_data(artists_data)
        )
        
        # Try cache first
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.performance_metrics.cache_hits += 1
                logger.debug(f"Cache hit for composite index: {len(platforms)} platforms")
                return cached_result
        
        # Compute composite index
        result = super().calculate_composite_index(
            artists_data, platforms, weights, **kwargs
        )
        
        # Cache result
        if use_cache and result:
            ttl = self._calculate_ttl(RankingCategory.COMPOSITE, len(artists_data))
            computation_cost = self._estimate_computation_cost('composite_index', len(artists_data))
            
            self.cache.set(
                cache_key,
                result,
                ttl=ttl,
                computation_cost=computation_cost
            )
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_performance_metrics(processing_time)
        
        return result
    
    def batch_rank_multiple_operations(self,
                                     artists_data: List[ArtistMetrics],
                                     operations: List[Dict[str, Any]],
                                     use_cache: bool = True,
                                     progress_callback: Optional[Callable] = None) -> Dict[str, List[Any]]:
        """
        Execute multiple ranking operations in optimized batches.
        
        Args:
            artists_data: Artist dataset
            operations: List of operations with parameters
            use_cache: Enable caching for operations
            progress_callback: Progress reporting callback
            
        Returns:
            Dictionary mapping operation keys to results
        """
        logger.info(f"Executing {len(operations)} ranking operations in batch")
        
        # Check cache for each operation
        results = {}
        uncached_operations = []
        
        if use_cache:
            for operation in operations:
                cache_key = self._generate_operation_cache_key(operation, len(artists_data))
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    op_key = operation.get('key', f'operation_{len(results)}')
                    results[op_key] = cached_result
                    self.performance_metrics.cache_hits += 1
                else:
                    uncached_operations.append(operation)
        else:
            uncached_operations = operations
        
        # Process uncached operations in batch
        if uncached_operations:
            batch_results = self.batch_processor.process_ranking_batch(
                artists_data,
                self,
                uncached_operations,
                progress_callback
            )
            
            # Cache results and merge
            for op_key, result in batch_results.items():
                results[op_key] = result
                
                if use_cache and result:
                    # Find corresponding operation for cache key
                    operation = next((op for op in uncached_operations if op.get('key') == op_key), None)
                    if operation:
                        cache_key = self._generate_operation_cache_key(operation, len(artists_data))
                        ttl = self._calculate_ttl(
                            RankingCategory(operation.get('category', 'growth_rate')), 
                            len(artists_data)
                        )
                        self.cache.set(cache_key, result, ttl=ttl)
        
        return results
    
    def warm_cache_for_common_operations(self, artists_data: List[ArtistMetrics]) -> int:
        """Pre-warm cache with commonly requested ranking operations."""
        if not self.enable_cache_warming:
            return 0
        
        logger.info("Warming cache with common ranking operations...")
        
        warming_operations = [
            # Platform rankings for major platforms
            {'type': 'platform_ranking', 'platform': 'youtube', 'metric_type': 'subscribers', 
             'category': RankingCategory.GROWTH_RATE, 'key': 'youtube_growth'},
            {'type': 'platform_ranking', 'platform': 'youtube', 'metric_type': 'subscribers',
             'category': RankingCategory.ABSOLUTE_VALUE, 'key': 'youtube_absolute'},
            {'type': 'platform_ranking', 'platform': 'spotify', 'metric_type': 'monthly_listeners',
             'category': RankingCategory.GROWTH_RATE, 'key': 'spotify_growth'},
            
            # Composite indices
            {'type': 'composite_index', 'platforms': ['youtube', 'spotify'], 'key': 'composite_main'},
            
            # Cohort rankings for current generation
            {'type': 'cohort_ranking', 'cohort': DebutCohort.FOURTH_GEN, 'platform': 'youtube',
             'metric_type': 'subscribers', 'key': 'fourth_gen_youtube'}
        ]
        
        # Execute warming operations
        warmed_results = self.batch_rank_multiple_operations(
            artists_data,
            warming_operations,
            use_cache=True
        )
        
        warmed_count = sum(len(results) for results in warmed_results.values() if results)
        logger.info(f"Cache warming completed: {warmed_count} entries cached")
        
        return warmed_count
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cache_stats = {
            'cache_hit_rate': self.cache.metrics.cache_hit_rate,
            'total_requests': self.cache.metrics.total_requests,
            'cache_hits': self.cache.metrics.cache_hits,
            'cache_misses': self.cache.metrics.cache_misses,
            'cache_size_mb': self.cache._total_size / 1024 / 1024
        }
        
        batch_stats = {
            'batch_processing_time': self.batch_processor.metrics.batch_processing_time,
            'concurrent_requests': self.batch_processor.metrics.concurrent_requests,
            'parallel_efficiency': self.batch_processor.metrics.parallel_efficiency
        }
        
        system_stats = {
            'cpu_count': psutil.cpu_count(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
        
        return {
            'cache_performance': cache_stats,
            'batch_performance': batch_stats,
            'system_resources': system_stats,
            'optimization_strategy': self.optimization_strategy.value,
            'average_response_time': self.performance_metrics.average_response_time
        }
    
    def _generate_cache_key(self, operation: str, **params) -> str:
        """Generate deterministic cache key from operation parameters."""
        # Sort parameters for consistent key generation
        param_str = '|'.join(f"{k}={v}" for k, v in sorted(params.items()))
        key_content = f"{operation}|{param_str}"
        
        # Use hash for compact keys
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def _generate_operation_cache_key(self, operation: Dict[str, Any], artist_count: int) -> str:
        """Generate cache key for a ranking operation."""
        op_copy = operation.copy()
        op_copy['artist_count'] = artist_count
        
        # Remove non-deterministic fields
        op_copy.pop('key', None)
        
        return self._generate_cache_key('operation', **op_copy)
    
    def _hash_artists_data(self, artists_data: List[ArtistMetrics]) -> str:
        """Generate hash of artist data for cache invalidation."""
        # Use artist IDs and last_updated timestamps for efficient hashing
        hash_input = ''.join(
            f"{artist.artist_id}:{artist.last_updated.timestamp()}"
            for artist in artists_data
        )
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _calculate_ttl(self, category: RankingCategory, artist_count: int) -> int:
        """Calculate appropriate TTL based on operation characteristics."""
        base_ttl = 3600  # 1 hour
        
        # More volatile categories need shorter TTL
        if category == RankingCategory.MOMENTUM:
            base_ttl = 900  # 15 minutes
        elif category == RankingCategory.GROWTH_RATE:
            base_ttl = 1800  # 30 minutes
        elif category == RankingCategory.COMPOSITE:
            base_ttl = 2700  # 45 minutes
        
        # Larger datasets change less frequently
        if artist_count > 1000:
            base_ttl *= 2
        elif artist_count > 100:
            base_ttl *= 1.5
        
        return base_ttl
    
    def _estimate_computation_cost(self, operation_type: str, artist_count: int) -> float:
        """Estimate relative computation cost for cache prioritization."""
        base_costs = {
            'platform_ranking': 1.0,
            'composite_index': 2.5,
            'company_ranking': 1.5,
            'cohort_ranking': 1.8
        }
        
        base_cost = base_costs.get(operation_type, 1.0)
        
        # Scale with dataset size
        size_factor = 1.0 + (artist_count / 1000)
        
        return base_cost * size_factor
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics with new measurement."""
        self._request_times.append(processing_time)
        
        # Keep only recent measurements (last 100)
        if len(self._request_times) > 100:
            self._request_times.pop(0)
        
        # Update average response time
        self.performance_metrics.average_response_time = sum(self._request_times) / len(self._request_times)
        self.performance_metrics.total_requests += 1


# Export optimized components
__all__ = [
    'OptimizedRankingEngine',
    'IntelligentCache', 
    'BatchRankingProcessor',
    'CacheLevel',
    'OptimizationStrategy',
    'PerformanceMetrics'
]