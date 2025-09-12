"""
Mobile Optimizer Module for K-POP Dashboard
===========================================

A comprehensive mobile optimization module that provides:
- Mobile-optimized data processing and response formatting
- Push notification generation and delivery
- Offline caching for mobile applications
- Performance optimization for mobile devices
- Responsive data handling for limited bandwidth scenarios

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
import json
import hashlib
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, deque

# Optional config import for integration scenarios
try:
    from ..config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

# ========================================
# Enums and Type Definitions
# ========================================

class MobileDeviceType(Enum):
    """Mobile device types for optimization targeting."""
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    SMARTWATCH = "smartwatch"
    UNKNOWN = "unknown"

class NotificationType(Enum):
    """Types of push notifications."""
    MILESTONE_ACHIEVEMENT = "milestone"
    PERFORMANCE_ALERT = "performance"
    TRENDING_CONTENT = "trending"
    SCHEDULE_REMINDER = "reminder"
    SYSTEM_UPDATE = "system"

class CacheLevel(Enum):
    """Cache optimization levels."""
    MINIMAL = "minimal"      # Essential data only
    STANDARD = "standard"    # Regular mobile experience
    ENHANCED = "enhanced"    # Rich content for high-end devices

class CompressionType(Enum):
    """Data compression methods."""
    GZIP = "gzip"
    JSON_MINIFY = "json_minify"
    IMAGE_OPTIMIZE = "image_optimize"
    NONE = "none"

# ========================================
# Data Classes
# ========================================

@dataclass
class MobileOptimizationConfig:
    """Configuration settings for mobile optimization."""
    device_type: MobileDeviceType = MobileDeviceType.SMARTPHONE
    cache_level: CacheLevel = CacheLevel.STANDARD
    compression_enabled: bool = True
    max_response_size: int = 1024 * 1024  # 1MB default
    image_quality: int = 70  # JPEG quality percentage
    max_cached_items: int = 500
    cache_duration_hours: int = 24
    push_enabled: bool = True
    offline_sync_enabled: bool = True

@dataclass
class PushNotificationPayload:
    """Structure for push notification data."""
    title: str
    body: str
    notification_type: NotificationType
    target_user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    badge_count: int = 0
    sound: str = "default"
    priority: str = "high"  # high, normal, low
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

@dataclass  
class CacheEntry:
    """Structure for cached mobile data."""
    key: str
    data: Any
    compressed_size: int
    cache_level: CacheLevel
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class MobileOptimizationResult:
    """Result of mobile optimization process."""
    original_size: int
    optimized_size: int
    compression_ratio: float
    processing_time_ms: float
    optimization_applied: List[str]
    cache_status: str
    device_compatibility: List[MobileDeviceType]

# ========================================
# Main MobileOptimizer Class
# ========================================

class MobileOptimizer:
    """
    Mobile optimization engine for K-POP Dashboard.
    
    Provides comprehensive mobile experience optimization including:
    - Data compression and size reduction
    - Device-specific optimizations
    - Offline caching strategies
    - Push notification management
    """
    
    def __init__(self, config: Optional[MobileOptimizationConfig] = None):
        """Initialize mobile optimizer with configuration."""
        self.config = config or MobileOptimizationConfig()
        self.cache_storage: Dict[str, CacheEntry] = {}
        self.notification_queue: deque = deque()
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.compression_stats: Dict[str, int] = defaultdict(int)
        
        # Initialize optimization strategies
        self._init_optimization_strategies()
        
        logger.info(f"MobileOptimizer initialized with device_type: {self.config.device_type}")

    def _init_optimization_strategies(self):
        """Initialize device-specific optimization strategies."""
        self.optimization_strategies = {
            MobileDeviceType.SMARTPHONE: {
                'max_items_per_page': 20,
                'image_size_limit': 200 * 1024,  # 200KB
                'data_compression': True,
                'lazy_loading': True,
            },
            MobileDeviceType.TABLET: {
                'max_items_per_page': 50,
                'image_size_limit': 500 * 1024,  # 500KB
                'data_compression': True,
                'lazy_loading': False,
            },
            MobileDeviceType.SMARTWATCH: {
                'max_items_per_page': 5,
                'image_size_limit': 50 * 1024,   # 50KB
                'data_compression': True,
                'lazy_loading': True,
            },
        }

    def optimize_for_mobile(self, data: Any, device_type: Optional[MobileDeviceType] = None) -> MobileOptimizationResult:
        """
        Optimize data for mobile consumption.
        
        Args:
            data: Raw data to optimize (dict, list, or other serializable data)
            device_type: Target device type for optimization
            
        Returns:
            MobileOptimizationResult with optimization details
            
        Raises:
            ValueError: If data cannot be optimized
            RuntimeError: If optimization process fails
        """
        start_time = datetime.now()
        device_type = device_type or self.config.device_type
        optimizations_applied = []
        
        try:
            # Convert data to JSON for consistent processing
            if not isinstance(data, (dict, list)):
                data = {'value': data}
            
            original_json = json.dumps(data, ensure_ascii=False)
            original_size = len(original_json.encode('utf-8'))
            
            logger.debug(f"Starting mobile optimization for {original_size} bytes of data")
            
            # Apply device-specific optimizations
            optimized_data = self._apply_device_optimizations(data, device_type)
            optimizations_applied.append(f"device_optimization_{device_type.value}")
            
            # Apply data compression
            if self.config.compression_enabled:
                optimized_data = self._compress_data(optimized_data)
                optimizations_applied.append("data_compression")
            
            # Apply content filtering based on cache level
            optimized_data = self._filter_content_by_cache_level(optimized_data)
            optimizations_applied.append(f"content_filtering_{self.config.cache_level.value}")
            
            # Calculate final size
            final_json = json.dumps(optimized_data, ensure_ascii=False)
            optimized_size = len(final_json.encode('utf-8'))
            
            # Calculate compression ratio
            compression_ratio = (original_size - optimized_size) / original_size if original_size > 0 else 0
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self.performance_metrics['optimization_time_ms'].append(processing_time)
            self.compression_stats['total_optimizations'] += 1
            self.compression_stats['total_bytes_saved'] += (original_size - optimized_size)
            
            # Determine device compatibility
            compatible_devices = self._determine_device_compatibility(optimized_size)
            
            result = MobileOptimizationResult(
                original_size=original_size,
                optimized_size=optimized_size,
                compression_ratio=compression_ratio,
                processing_time_ms=processing_time,
                optimization_applied=optimizations_applied,
                cache_status="optimized",
                device_compatibility=compatible_devices
            )
            
            logger.info(f"Mobile optimization completed: {compression_ratio:.2%} size reduction")
            return result
            
        except Exception as e:
            logger.error(f"Mobile optimization failed: {str(e)}")
            raise RuntimeError(f"Optimization process failed: {str(e)}")

    def generate_push_notification(self, 
                                 title: str, 
                                 body: str, 
                                 notification_type: NotificationType,
                                 target_user_id: Optional[str] = None,
                                 additional_data: Optional[Dict[str, Any]] = None) -> PushNotificationPayload:
        """
        Generate push notification payload optimized for mobile delivery.
        
        Args:
            title: Notification title (max 65 characters for optimal display)
            body: Notification body (max 240 characters for optimal display)
            notification_type: Type of notification
            target_user_id: Specific user ID for targeted notifications
            additional_data: Additional data to include in notification payload
            
        Returns:
            PushNotificationPayload ready for delivery
            
        Raises:
            ValueError: If notification parameters are invalid
        """
        try:
            # Validate input parameters
            if not title or not body:
                raise ValueError("Title and body are required for push notifications")
            
            if len(title) > 65:
                logger.warning(f"Title truncated from {len(title)} to 65 characters")
                title = title[:62] + "..."
            
            if len(body) > 240:
                logger.warning(f"Body truncated from {len(body)} to 240 characters")
                body = body[:237] + "..."
            
            # Prepare notification data
            notification_data = additional_data or {}
            notification_data.update({
                'type': notification_type.value,
                'timestamp': datetime.now().isoformat(),
                'app_version': '1.0.0',
                'platform': 'mobile'
            })
            
            # Determine priority and sound based on notification type
            priority_map = {
                NotificationType.MILESTONE_ACHIEVEMENT: "high",
                NotificationType.PERFORMANCE_ALERT: "high",
                NotificationType.TRENDING_CONTENT: "normal",
                NotificationType.SCHEDULE_REMINDER: "normal",
                NotificationType.SYSTEM_UPDATE: "low"
            }
            
            sound_map = {
                NotificationType.MILESTONE_ACHIEVEMENT: "celebration.wav",
                NotificationType.PERFORMANCE_ALERT: "alert.wav",
                NotificationType.TRENDING_CONTENT: "notification.wav",
                NotificationType.SCHEDULE_REMINDER: "reminder.wav",
                NotificationType.SYSTEM_UPDATE: "default"
            }
            
            # Calculate expiration time based on notification type
            expiry_hours = {
                NotificationType.MILESTONE_ACHIEVEMENT: 72,  # 3 days
                NotificationType.PERFORMANCE_ALERT: 24,     # 1 day
                NotificationType.TRENDING_CONTENT: 12,      # 12 hours
                NotificationType.SCHEDULE_REMINDER: 6,      # 6 hours
                NotificationType.SYSTEM_UPDATE: 168        # 1 week
            }
            
            # Create notification payload
            payload = PushNotificationPayload(
                title=title,
                body=body,
                notification_type=notification_type,
                target_user_id=target_user_id,
                data=notification_data,
                badge_count=1,
                sound=sound_map.get(notification_type, "default"),
                priority=priority_map.get(notification_type, "normal"),
                scheduled_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=expiry_hours.get(notification_type, 24))
            )
            
            # Add to notification queue if push is enabled
            if self.config.push_enabled:
                self.notification_queue.append(payload)
                logger.info(f"Push notification queued: {notification_type.value} for user {target_user_id or 'broadcast'}")
            
            return payload
            
        except Exception as e:
            logger.error(f"Push notification generation failed: {str(e)}")
            raise ValueError(f"Invalid notification parameters: {str(e)}")

    def cache_for_offline(self, key: str, data: Any, cache_level: Optional[CacheLevel] = None) -> bool:
        """
        Cache data for offline mobile access with intelligent storage management.
        
        Args:
            key: Unique cache key identifier
            data: Data to cache (will be serialized)
            cache_level: Override default cache level for this entry
            
        Returns:
            bool: True if caching was successful, False otherwise
            
        Raises:
            RuntimeError: If caching system encounters an error
        """
        try:
            cache_level = cache_level or self.config.cache_level
            
            # Check if we're at cache capacity
            if len(self.cache_storage) >= self.config.max_cached_items:
                self._evict_cache_entries()
            
            # Serialize and potentially compress the data
            if isinstance(data, (dict, list)):
                serialized_data = json.dumps(data, ensure_ascii=False)
            else:
                serialized_data = str(data)
            
            # Apply compression if enabled and beneficial
            compressed_data = serialized_data
            if self.config.compression_enabled and len(serialized_data) > 1024:  # Only compress if > 1KB
                compressed_bytes = gzip.compress(serialized_data.encode('utf-8'))
                if len(compressed_bytes) < len(serialized_data.encode('utf-8')) * 0.9:  # Only if 10%+ savings
                    compressed_data = compressed_bytes
                    logger.debug(f"Cache entry compressed: {len(serialized_data)} -> {len(compressed_bytes)} bytes")
            
            # Calculate cache entry size
            cache_size = len(compressed_data) if isinstance(compressed_data, bytes) else len(compressed_data.encode('utf-8'))
            
            # Create cache entry
            now = datetime.now()
            expires_at = now + timedelta(hours=self.config.cache_duration_hours)
            
            cache_entry = CacheEntry(
                key=key,
                data=compressed_data,
                compressed_size=cache_size,
                cache_level=cache_level,
                created_at=now,
                expires_at=expires_at,
                access_count=0
            )
            
            # Store in cache
            self.cache_storage[key] = cache_entry
            
            logger.info(f"Data cached successfully: {key} ({cache_size} bytes, expires: {expires_at.strftime('%Y-%m-%d %H:%M')})")
            return True
            
        except Exception as e:
            logger.error(f"Offline caching failed for key '{key}': {str(e)}")
            raise RuntimeError(f"Caching system error: {str(e)}")

    def _apply_device_optimizations(self, data: Any, device_type: MobileDeviceType) -> Any:
        """Apply device-specific optimizations to data."""
        strategy = self.optimization_strategies.get(device_type, self.optimization_strategies[MobileDeviceType.SMARTPHONE])
        
        if isinstance(data, dict):
            optimized = data.copy()
            items_to_update = {}
            
            # Apply pagination for list-like data
            # Collect items to update first to avoid dictionary size change during iteration
            for key, value in optimized.items():
                if isinstance(value, list) and len(value) > strategy['max_items_per_page']:
                    items_to_update[key] = value[:strategy['max_items_per_page']]
                    items_to_update[f'{key}_truncated'] = True
                    items_to_update[f'{key}_total_count'] = len(value)
            
            # Apply updates
            optimized.update(items_to_update)
            return optimized
        
        elif isinstance(data, list) and len(data) > strategy['max_items_per_page']:
            return data[:strategy['max_items_per_page']]
        
        return data

    def _compress_data(self, data: Any) -> Any:
        """Apply data compression techniques."""
        if isinstance(data, dict):
            # Remove null values and empty strings to reduce size
            compressed = {k: v for k, v in data.items() if v is not None and v != ""}
            
            # Round float values to reduce precision if appropriate
            for key, value in compressed.items():
                if isinstance(value, float):
                    compressed[key] = round(value, 4)  # 4 decimal places max
            
            return compressed
        
        return data

    def _filter_content_by_cache_level(self, data: Any) -> Any:
        """Filter content based on cache level settings."""
        if not isinstance(data, dict):
            return data
        
        if self.config.cache_level == CacheLevel.MINIMAL:
            # Keep only essential fields
            essential_fields = {'id', 'name', 'title', 'value', 'score', 'timestamp'}
            filtered = {k: v for k, v in data.items() if k in essential_fields or k.endswith('_id')}
            return filtered
        
        elif self.config.cache_level == CacheLevel.STANDARD:
            # Remove heavy content like detailed descriptions, large images
            filtered = data.copy()
            heavy_fields = ['description', 'full_bio', 'large_image', 'raw_data']
            for field in heavy_fields:
                filtered.pop(field, None)
            return filtered
        
        # CacheLevel.ENHANCED - return all data
        return data

    def _determine_device_compatibility(self, data_size: int) -> List[MobileDeviceType]:
        """Determine which device types can handle the optimized data size."""
        compatible = []
        
        # Device-specific size limits (bytes)
        size_limits = {
            MobileDeviceType.SMARTWATCH: 50 * 1024,    # 50KB
            MobileDeviceType.SMARTPHONE: 500 * 1024,   # 500KB
            MobileDeviceType.TABLET: 2 * 1024 * 1024,  # 2MB
        }
        
        for device, limit in size_limits.items():
            if data_size <= limit:
                compatible.append(device)
        
        return compatible

    def _evict_cache_entries(self):
        """Evict old cache entries using LRU strategy."""
        # Sort by last accessed time (LRU first)
        sorted_entries = sorted(
            self.cache_storage.items(),
            key=lambda item: item[1].last_accessed or item[1].created_at
        )
        
        # Remove oldest 25% of entries
        entries_to_remove = len(sorted_entries) // 4
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self.cache_storage[key]
        
        logger.info(f"Evicted {entries_to_remove} cache entries using LRU strategy")

    def get_cache_entry(self, key: str) -> Optional[Any]:
        """Retrieve data from offline cache."""
        if key not in self.cache_storage:
            return None
        
        entry = self.cache_storage[key]
        
        # Check if entry has expired
        if datetime.now() > entry.expires_at:
            del self.cache_storage[key]
            return None
        
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        # Decompress if needed
        if isinstance(entry.data, bytes):
            try:
                decompressed = gzip.decompress(entry.data).decode('utf-8')
                return json.loads(decompressed)
            except:
                return entry.data.decode('utf-8')
        
        if isinstance(entry.data, str):
            try:
                return json.loads(entry.data)
            except:
                return entry.data
        
        return entry.data

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        stats = {
            'cache_entries': len(self.cache_storage),
            'notification_queue_size': len(self.notification_queue),
            'compression_stats': dict(self.compression_stats),
            'avg_optimization_time_ms': statistics.mean(self.performance_metrics['optimization_time_ms']) if self.performance_metrics['optimization_time_ms'] else 0,
            'config': {
                'device_type': self.config.device_type.value,
                'cache_level': self.config.cache_level.value,
                'compression_enabled': self.config.compression_enabled
            }
        }
        return stats

    def clear_cache(self):
        """Clear all cached data."""
        self.cache_storage.clear()
        logger.info("All cache data cleared")

    def clear_notification_queue(self):
        """Clear all pending notifications."""
        self.notification_queue.clear()
        logger.info("Notification queue cleared")


# ========================================
# Utility Functions
# ========================================

def create_mobile_optimizer(device_type: str = "smartphone", 
                          cache_level: str = "standard") -> MobileOptimizer:
    """
    Factory function to create MobileOptimizer with common configurations.
    
    Args:
        device_type: Device type ('smartphone', 'tablet', 'smartwatch')
        cache_level: Cache level ('minimal', 'standard', 'enhanced')
    
    Returns:
        Configured MobileOptimizer instance
    """
    try:
        device_enum = MobileDeviceType(device_type.lower())
        cache_enum = CacheLevel(cache_level.lower())
    except ValueError as e:
        logger.warning(f"Invalid configuration value: {e}, using defaults")
        device_enum = MobileDeviceType.SMARTPHONE
        cache_enum = CacheLevel.STANDARD
    
    config = MobileOptimizationConfig(
        device_type=device_enum,
        cache_level=cache_enum
    )
    
    return MobileOptimizer(config)


def generate_cache_key(*args) -> str:
    """Generate a consistent cache key from arguments."""
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()