"""
Cache Invalidation and Consistency Management System
===================================================

Advanced cache invalidation strategies for maintaining data consistency:
- Event-driven cache invalidation
- Smart dependency tracking
- Batch invalidation optimization
- Cross-system consistency management
- Cache warming strategies
- Invalidation audit trails

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re
import weakref

from .cache_manager import CacheManager

# Configure module logger
logger = logging.getLogger(__name__)

class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    IMMEDIATE = "immediate"
    BATCH = "batch"
    LAZY = "lazy"
    SCHEDULED = "scheduled"

class InvalidationScope(Enum):
    """Scope of cache invalidation."""
    SINGLE_KEY = "single_key"
    PATTERN = "pattern"
    DEPENDENCY = "dependency"
    GLOBAL = "global"

@dataclass
class InvalidationEvent:
    """Cache invalidation event."""
    event_id: str
    event_type: str  # data_update, schema_change, manual, etc.
    scope: InvalidationScope
    target: str  # key, pattern, or dependency
    strategy: InvalidationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False

@dataclass
class CacheDependency:
    """Cache dependency relationship."""
    cache_key: str
    depends_on: Set[str]
    invalidation_pattern: Optional[str] = None
    ttl_override: Optional[int] = None

class CacheInvalidationManager:
    """
    Advanced cache invalidation and consistency management.
    
    Provides intelligent cache invalidation with:
    - Dependency-based invalidation
    - Pattern-based bulk invalidation
    - Event-driven cache updates
    - Consistency verification
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize cache invalidation manager.
        
        Args:
            cache_manager: Cache manager to control
        """
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Dependency tracking
        self._dependencies: Dict[str, CacheDependency] = {}
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependency_lock = threading.RLock()
        
        # Event processing
        self._pending_events: deque = deque()
        self._event_history: deque = deque(maxlen=1000)
        self._event_lock = threading.Lock()
        
        # Processing control
        self._processing = False
        self._processor_thread = None
        self._batch_processor_thread = None
        
        # Invalidation patterns
        self._pattern_handlers: Dict[str, Callable[[str], List[str]]] = {}
        
        # Metrics
        self._invalidation_stats = {
            'total_invalidations': 0,
            'batch_invalidations': 0,
            'dependency_invalidations': 0,
            'pattern_invalidations': 0
        }
        
        self._setup_default_patterns()
    
    def start_processing(self):
        """Start invalidation event processing."""
        if self._processing:
            return
        
        self._processing = True
        
        # Start event processor
        self._processor_thread = threading.Thread(
            target=self._event_processor,
            daemon=True
        )
        self._processor_thread.start()
        
        # Start batch processor
        self._batch_processor_thread = threading.Thread(
            target=self._batch_processor,
            daemon=True
        )
        self._batch_processor_thread.start()
        
        self.logger.info("Cache invalidation processing started")
    
    def stop_processing(self):
        """Stop invalidation event processing."""
        self._processing = False
        
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)
        
        if self._batch_processor_thread and self._batch_processor_thread.is_alive():
            self._batch_processor_thread.join(timeout=5)
        
        self.logger.info("Cache invalidation processing stopped")
    
    def register_dependency(self, cache_key: str, depends_on: Set[str], invalidation_pattern: Optional[str] = None):
        """
        Register cache dependency.
        
        Args:
            cache_key: Cache key that depends on others
            depends_on: Set of keys this cache depends on
            invalidation_pattern: Pattern for invalidating dependent keys
        """
        with self._dependency_lock:
            dependency = CacheDependency(
                cache_key=cache_key,
                depends_on=depends_on,
                invalidation_pattern=invalidation_pattern
            )
            
            self._dependencies[cache_key] = dependency
            
            # Update reverse dependencies
            for dep_key in depends_on:
                self._reverse_dependencies[dep_key].add(cache_key)
        
        self.logger.debug(f"Registered dependency: {cache_key} depends on {depends_on}")
    
    def invalidate_key(self, key: str, strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE):
        """
        Invalidate specific cache key.
        
        Args:
            key: Cache key to invalidate
            strategy: Invalidation strategy to use
        """
        event = InvalidationEvent(
            event_id=f"key_{key}_{int(time.time())}",
            event_type="manual_key",
            scope=InvalidationScope.SINGLE_KEY,
            target=key,
            strategy=strategy
        )
        
        self._queue_invalidation_event(event)
    
    def invalidate_pattern(self, pattern: str, strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE):
        """
        Invalidate keys matching pattern.
        
        Args:
            pattern: Pattern to match keys against
            strategy: Invalidation strategy to use
        """
        event = InvalidationEvent(
            event_id=f"pattern_{hash(pattern)}_{int(time.time())}",
            event_type="manual_pattern",
            scope=InvalidationScope.PATTERN,
            target=pattern,
            strategy=strategy
        )
        
        self._queue_invalidation_event(event)
    
    def invalidate_dependencies(self, changed_key: str, strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE):
        """
        Invalidate all keys dependent on changed key.
        
        Args:
            changed_key: Key that changed
            strategy: Invalidation strategy to use
        """
        event = InvalidationEvent(
            event_id=f"deps_{changed_key}_{int(time.time())}",
            event_type="dependency_change",
            scope=InvalidationScope.DEPENDENCY,
            target=changed_key,
            strategy=strategy
        )
        
        self._queue_invalidation_event(event)
    
    def on_data_update(self, table_name: str, record_ids: Optional[List[Any]] = None):
        """
        Handle data update events.
        
        Args:
            table_name: Name of updated table
            record_ids: List of updated record IDs
        """
        # Generate cache invalidation patterns based on data changes
        patterns = self._generate_invalidation_patterns(table_name, record_ids)
        
        for pattern in patterns:
            event = InvalidationEvent(
                event_id=f"data_update_{table_name}_{int(time.time())}",
                event_type="data_update",
                scope=InvalidationScope.PATTERN,
                target=pattern,
                strategy=InvalidationStrategy.BATCH,  # Use batch for data updates
                metadata={
                    'table_name': table_name,
                    'record_ids': record_ids
                }
            )
            
            self._queue_invalidation_event(event)
    
    def warm_cache_after_invalidation(self, keys: List[str], warming_functions: Dict[str, Callable]):
        """
        Warm cache after invalidation.
        
        Args:
            keys: Keys to warm
            warming_functions: Functions to generate data for warming
        """
        for key in keys:
            if key in warming_functions:
                try:
                    data = warming_functions[key]()
                    if data is not None:
                        self.cache_manager.set(key, data)
                        self.logger.debug(f"Warmed cache key: {key}")
                except Exception as e:
                    self.logger.error(f"Failed to warm cache key {key}: {e}")
    
    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get cache invalidation statistics."""
        return {
            **self._invalidation_stats,
            'pending_events': len(self._pending_events),
            'dependencies_registered': len(self._dependencies),
            'reverse_dependencies': sum(len(deps) for deps in self._reverse_dependencies.values())
        }
    
    def verify_consistency(self) -> Dict[str, Any]:
        """
        Verify cache consistency.
        
        Returns:
            Dictionary with consistency check results
        """
        results = {
            'consistent': True,
            'issues': [],
            'checked_keys': 0,
            'dependency_violations': 0
        }
        
        with self._dependency_lock:
            for cache_key, dependency in self._dependencies.items():
                # Check if dependent key exists when dependencies exist
                if self.cache_manager.get(cache_key):
                    results['checked_keys'] += 1
                    
                    for dep_key in dependency.depends_on:
                        if not self.cache_manager.get(dep_key):
                            results['consistent'] = False
                            results['dependency_violations'] += 1
                            results['issues'].append(f"Key {cache_key} exists but dependency {dep_key} is missing")
        
        return results
    
    # Private methods
    
    def _setup_default_patterns(self):
        """Setup default invalidation patterns."""
        # Artist-related patterns
        self._pattern_handlers['artists'] = lambda table: [
            'dashboard:summary',
            'artists:*',
            'metrics:artist:*',
            'performance:artist:*'
        ]
        
        # Platform metrics patterns
        self._pattern_handlers['platform_metrics'] = lambda table: [
            'metrics:*',
            'dashboard:*',
            'performance:*'
        ]
        
        # Company-related patterns
        self._pattern_handlers['companies'] = lambda table: [
            'companies:*',
            'dashboard:company:*'
        ]
        
        # Events patterns
        self._pattern_handlers['events'] = lambda table: [
            'events:*',
            'dashboard:events'
        ]
    
    def _queue_invalidation_event(self, event: InvalidationEvent):
        """Queue invalidation event for processing."""
        with self._event_lock:
            self._pending_events.append(event)
        
        self.logger.debug(f"Queued invalidation event: {event.event_type} - {event.target}")
    
    def _event_processor(self):
        """Process invalidation events."""
        self.logger.info("Invalidation event processor started")
        
        while self._processing:
            try:
                # Process immediate events
                immediate_events = []
                with self._event_lock:
                    events_to_process = []
                    remaining_events = deque()
                    
                    while self._pending_events:
                        event = self._pending_events.popleft()
                        if event.strategy == InvalidationStrategy.IMMEDIATE:
                            immediate_events.append(event)
                        else:
                            remaining_events.append(event)
                    
                    self._pending_events = remaining_events
                
                # Process immediate events
                for event in immediate_events:
                    self._process_invalidation_event(event)
                
                time.sleep(0.1)  # Short sleep to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in event processor: {e}")
                time.sleep(1)
        
        self.logger.info("Invalidation event processor stopped")
    
    def _batch_processor(self):
        """Process batch invalidation events."""
        self.logger.info("Batch invalidation processor started")
        
        batch_events = []
        last_batch_time = time.time()
        batch_timeout = 5.0  # 5 seconds
        
        while self._processing:
            try:
                # Collect batch events
                with self._event_lock:
                    events_to_check = []
                    remaining_events = deque()
                    
                    while self._pending_events:
                        event = self._pending_events.popleft()
                        if event.strategy == InvalidationStrategy.BATCH:
                            events_to_check.append(event)
                        else:
                            remaining_events.append(event)
                    
                    self._pending_events = remaining_events
                
                batch_events.extend(events_to_check)
                
                # Process batch if timeout reached or batch is full
                should_process_batch = (
                    len(batch_events) >= 50 or  # Batch size limit
                    (batch_events and time.time() - last_batch_time > batch_timeout)
                )
                
                if should_process_batch:
                    self._process_batch_invalidation(batch_events)
                    batch_events.clear()
                    last_batch_time = time.time()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in batch processor: {e}")
                time.sleep(1)
        
        # Process remaining batch events
        if batch_events:
            self._process_batch_invalidation(batch_events)
        
        self.logger.info("Batch invalidation processor stopped")
    
    def _process_invalidation_event(self, event: InvalidationEvent):
        """Process single invalidation event."""
        try:
            if event.scope == InvalidationScope.SINGLE_KEY:
                self._invalidate_single_key(event.target)
                
            elif event.scope == InvalidationScope.PATTERN:
                self._invalidate_pattern_keys(event.target)
                
            elif event.scope == InvalidationScope.DEPENDENCY:
                self._invalidate_dependent_keys(event.target)
            
            event.processed = True
            self._event_history.append(event)
            self._invalidation_stats['total_invalidations'] += 1
            
            self.logger.debug(f"Processed invalidation event: {event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process invalidation event {event.event_id}: {e}")
    
    def _process_batch_invalidation(self, events: List[InvalidationEvent]):
        """Process batch of invalidation events."""
        if not events:
            return
        
        try:
            # Group events by pattern for optimization
            pattern_groups = defaultdict(list)
            single_keys = []
            dependency_keys = []
            
            for event in events:
                if event.scope == InvalidationScope.SINGLE_KEY:
                    single_keys.append(event.target)
                elif event.scope == InvalidationScope.PATTERN:
                    pattern_groups[event.target].append(event)
                elif event.scope == InvalidationScope.DEPENDENCY:
                    dependency_keys.append(event.target)
            
            # Process single keys in batch
            if single_keys:
                self._batch_invalidate_keys(single_keys)
            
            # Process patterns
            for pattern, pattern_events in pattern_groups.items():
                self._invalidate_pattern_keys(pattern)
            
            # Process dependencies
            for dep_key in dependency_keys:
                self._invalidate_dependent_keys(dep_key)
            
            # Update statistics
            self._invalidation_stats['batch_invalidations'] += 1
            self._invalidation_stats['total_invalidations'] += len(events)
            
            # Mark events as processed
            for event in events:
                event.processed = True
                self._event_history.append(event)
            
            self.logger.info(f"Processed batch invalidation: {len(events)} events")
            
        except Exception as e:
            self.logger.error(f"Failed to process batch invalidation: {e}")
    
    def _invalidate_single_key(self, key: str):
        """Invalidate single cache key."""
        success = self.cache_manager.delete(key)
        if success:
            self.logger.debug(f"Invalidated cache key: {key}")
        return success
    
    def _invalidate_pattern_keys(self, pattern: str):
        """Invalidate keys matching pattern."""
        self.cache_manager.invalidate_pattern(pattern)
        self._invalidation_stats['pattern_invalidations'] += 1
        self.logger.debug(f"Invalidated pattern: {pattern}")
    
    def _invalidate_dependent_keys(self, changed_key: str):
        """Invalidate keys dependent on changed key."""
        with self._dependency_lock:
            dependent_keys = self._reverse_dependencies.get(changed_key, set())
            
            invalidated_count = 0
            for dep_key in dependent_keys:
                if self._invalidate_single_key(dep_key):
                    invalidated_count += 1
                
                # Recursively invalidate dependencies of dependent keys
                if dep_key in self._reverse_dependencies:
                    self._invalidate_dependent_keys(dep_key)
            
            if invalidated_count > 0:
                self._invalidation_stats['dependency_invalidations'] += 1
                self.logger.debug(f"Invalidated {invalidated_count} dependent keys for: {changed_key}")
    
    def _batch_invalidate_keys(self, keys: List[str]):
        """Batch invalidate multiple keys."""
        for key in keys:
            self.cache_manager.delete(key)
        
        self.logger.debug(f"Batch invalidated {len(keys)} keys")
    
    def _generate_invalidation_patterns(self, table_name: str, record_ids: Optional[List[Any]] = None) -> List[str]:
        """Generate invalidation patterns for data changes."""
        patterns = []
        
        # Use registered pattern handlers
        if table_name in self._pattern_handlers:
            patterns.extend(self._pattern_handlers[table_name](table_name))
        
        # Add specific record patterns if IDs provided
        if record_ids:
            for record_id in record_ids:
                patterns.extend([
                    f"{table_name}:{record_id}:*",
                    f"*:{table_name}:{record_id}"
                ])
        
        # Add general table patterns
        patterns.extend([
            f"{table_name}:*",
            f"*:{table_name}:*"
        ])
        
        return patterns

# Global invalidation manager
_invalidation_manager = None

def get_invalidation_manager() -> CacheInvalidationManager:
    """Get global cache invalidation manager."""
    global _invalidation_manager
    if _invalidation_manager is None:
        from .cache_manager import get_cache_manager
        _invalidation_manager = CacheInvalidationManager(get_cache_manager())
    return _invalidation_manager

# Decorator for automatic cache invalidation
def invalidate_on_change(patterns: List[str]):
    """
    Decorator for automatic cache invalidation on function execution.
    
    Args:
        patterns: Cache patterns to invalidate
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidate specified patterns
            invalidation_manager = get_invalidation_manager()
            for pattern in patterns:
                invalidation_manager.invalidate_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator