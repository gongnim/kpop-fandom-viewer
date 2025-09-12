"""
Performance Monitoring and Metrics Collection System
===================================================

Comprehensive monitoring system for K-POP dashboard performance:
- Real-time metrics collection and aggregation
- Performance alerting and notifications
- System health monitoring
- Custom metrics tracking
- Dashboard performance analytics
- Resource utilization monitoring

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
import weakref
from contextlib import contextmanager

from .cache_manager import CacheManager
from .query_optimizer import QueryOptimizer
from .data_pipeline import DataPipeline
from ..database_postgresql import get_db_connection

# Configure module logger
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels
        }

@dataclass
class Alert:
    """Performance alert."""
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'severity': self.severity.value,
            'message': self.message,
            'metric_name': self.metric_name,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved
        }

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    comparison: str = "greater"  # greater, less, equal
    window_seconds: int = 60
    consecutive_violations: int = 3

class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._metrics_lock = threading.RLock()
        
        # Aggregated metrics
        self._aggregated_metrics: Dict[str, Dict[str, float]] = {}
        self._aggregation_lock = threading.RLock()
        
        # Collection control
        self._collecting = False
        self._collection_thread = None
        self._collection_interval = 5.0  # seconds
    
    def start_collection(self):
        """Start metrics collection."""
        if self._collecting:
            return
        
        self._collecting = True
        self._collection_thread = threading.Thread(
            target=self._collection_worker,
            daemon=True
        )
        self._collection_thread.start()
        
        self.logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self._collecting = False
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5)
        
        self.logger.info("Metrics collection stopped")
    
    def record_metric(self, metric: Metric):
        """Record a single metric."""
        with self._metrics_lock:
            self._metrics[metric.name].append(metric)
    
    def record_counter(self, name: str, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(name, duration, labels)
    
    def get_metrics_summary(self, time_window: Optional[int] = 300) -> Dict[str, Any]:
        """
        Get summary of collected metrics.
        
        Args:
            time_window: Time window in seconds (None for all data)
            
        Returns:
            Dictionary with metric summaries
        """
        cutoff_time = datetime.now() - timedelta(seconds=time_window) if time_window else None
        summary = {}
        
        with self._metrics_lock:
            for name, metric_deque in self._metrics.items():
                # Filter by time window
                if cutoff_time:
                    filtered_metrics = [m for m in metric_deque if m.timestamp >= cutoff_time]
                else:
                    filtered_metrics = list(metric_deque)
                
                if not filtered_metrics:
                    continue
                
                values = [m.value for m in filtered_metrics]
                metric_type = filtered_metrics[0].metric_type
                
                if metric_type == MetricType.COUNTER:
                    summary[name] = {
                        'type': 'counter',
                        'total': sum(values),
                        'count': len(values),
                        'rate_per_second': sum(values) / time_window if time_window else 0
                    }
                elif metric_type == MetricType.GAUGE:
                    summary[name] = {
                        'type': 'gauge',
                        'current': values[-1],
                        'min': min(values),
                        'max': max(values),
                        'avg': statistics.mean(values)
                    }
                elif metric_type == MetricType.TIMER:
                    summary[name] = {
                        'type': 'timer',
                        'avg_duration': statistics.mean(values),
                        'min_duration': min(values),
                        'max_duration': max(values),
                        'p95': self._percentile(values, 95),
                        'p99': self._percentile(values, 99),
                        'total_calls': len(values)
                    }
        
        return summary
    
    def _collection_worker(self):
        """Background worker for automatic metrics collection."""
        self.logger.info("Metrics collection worker started")
        
        while self._collecting:
            try:
                self._collect_system_metrics()
                self._aggregate_metrics()
                time.sleep(self._collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(self._collection_interval * 2)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge('system.cpu.usage_percent', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge('system.memory.usage_percent', memory.percent)
            self.record_gauge('system.memory.available_mb', memory.available / 1024 / 1024)
            self.record_gauge('system.memory.used_mb', memory.used / 1024 / 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_gauge('system.disk.usage_percent', disk.percent)
            self.record_gauge('system.disk.free_gb', disk.free / 1024 / 1024 / 1024)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.record_counter('system.network.bytes_sent', network.bytes_sent)
            self.record_counter('system.network.bytes_recv', network.bytes_recv)
            
            # Process-specific metrics
            process = psutil.Process()
            self.record_gauge('process.memory.rss_mb', process.memory_info().rss / 1024 / 1024)
            self.record_gauge('process.cpu.usage_percent', process.cpu_percent())
            self.record_gauge('process.threads.count', process.num_threads())
            
            # Python-specific metrics
            self.record_gauge('python.gc.objects', len(gc.get_objects()))
            self.record_counter('python.gc.collections', sum(gc.get_stats()[i]['collections'] for i in range(3)))
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _aggregate_metrics(self):
        """Aggregate metrics for faster querying."""
        with self._aggregation_lock:
            self._aggregated_metrics = self.get_metrics_summary(300)  # 5-minute window
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class AlertManager:
    """
    Manages performance alerts and notifications.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Alert configuration
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        
        # Monitoring control
        self._monitoring = False
        self._monitoring_thread = None
        self._check_interval = 10.0  # seconds
        
        # Alert handlers
        self._alert_handlers: List[Callable[[Alert], None]] = []
        
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(
                metric_name='system.cpu.usage_percent',
                warning_threshold=70.0,
                critical_threshold=90.0
            ),
            PerformanceThreshold(
                metric_name='system.memory.usage_percent',
                warning_threshold=80.0,
                critical_threshold=95.0
            ),
            PerformanceThreshold(
                metric_name='system.disk.usage_percent',
                warning_threshold=85.0,
                critical_threshold=95.0
            ),
            PerformanceThreshold(
                metric_name='database.query.avg_duration',
                warning_threshold=1.0,
                critical_threshold=5.0
            )
        ]
        
        for threshold in default_thresholds:
            self._thresholds[threshold.metric_name] = threshold
    
    def start_monitoring(self):
        """Start alert monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self._monitoring = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("Alert monitoring stopped")
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add performance threshold."""
        self._thresholds[threshold.metric_name] = threshold
        self.logger.info(f"Added threshold for {threshold.metric_name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self._alert_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        alerts = list(self._alert_history)
        if limit:
            alerts = alerts[-limit:]
        return alerts
    
    def _monitoring_worker(self):
        """Background worker for alert monitoring."""
        self.logger.info("Alert monitoring worker started")
        
        while self._monitoring:
            try:
                self._check_thresholds()
                time.sleep(self._check_interval)
            except Exception as e:
                self.logger.error(f"Error in alert monitoring: {e}")
                time.sleep(self._check_interval * 2)
    
    def _check_thresholds(self):
        """Check all configured thresholds."""
        metrics_summary = self.metrics_collector.get_metrics_summary(300)  # 5-minute window
        
        for metric_name, threshold in self._thresholds.items():
            if metric_name not in metrics_summary:
                continue
            
            metric_data = metrics_summary[metric_name]
            current_value = self._extract_current_value(metric_data)
            
            if current_value is None:
                continue
            
            # Check thresholds
            if threshold.critical_threshold is not None:
                if self._threshold_violated(current_value, threshold.critical_threshold, threshold.comparison):
                    self._trigger_alert(metric_name, AlertSeverity.CRITICAL, current_value, threshold.critical_threshold, threshold)
                    continue
            
            if threshold.warning_threshold is not None:
                if self._threshold_violated(current_value, threshold.warning_threshold, threshold.comparison):
                    self._trigger_alert(metric_name, AlertSeverity.WARNING, current_value, threshold.warning_threshold, threshold)
                    continue
            
            # Check if we should resolve existing alert
            alert_key = f"{metric_name}_{AlertSeverity.WARNING.value}"
            if alert_key in self._active_alerts:
                self._resolve_alert(alert_key)
            
            alert_key = f"{metric_name}_{AlertSeverity.CRITICAL.value}"
            if alert_key in self._active_alerts:
                self._resolve_alert(alert_key)
    
    def _extract_current_value(self, metric_data: Dict[str, Any]) -> Optional[float]:
        """Extract current value from metric data."""
        if metric_data.get('type') == 'gauge':
            return metric_data.get('current')
        elif metric_data.get('type') == 'timer':
            return metric_data.get('avg_duration')
        elif metric_data.get('type') == 'counter':
            return metric_data.get('rate_per_second')
        
        return None
    
    def _threshold_violated(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if threshold is violated."""
        if comparison == "greater":
            return value > threshold
        elif comparison == "less":
            return value < threshold
        elif comparison == "equal":
            return abs(value - threshold) < 0.001
        
        return False
    
    def _trigger_alert(
        self,
        metric_name: str,
        severity: AlertSeverity,
        current_value: float,
        threshold: float,
        threshold_config: PerformanceThreshold
    ):
        """Trigger performance alert."""
        alert_key = f"{metric_name}_{severity.value}"
        
        # Check if alert is already active
        if alert_key in self._active_alerts:
            return
        
        alert = Alert(
            name=alert_key,
            severity=severity,
            message=f"{metric_name} {threshold_config.comparison} {threshold} (current: {current_value:.2f})",
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value
        )
        
        self._active_alerts[alert_key] = alert
        self._alert_history.append(alert)
        
        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        self.logger.warning(f"Alert triggered: {alert.message}")
    
    def _resolve_alert(self, alert_key: str):
        """Resolve active alert."""
        if alert_key in self._active_alerts:
            alert = self._active_alerts[alert_key]
            alert.resolved = True
            del self._active_alerts[alert_key]
            
            self.logger.info(f"Alert resolved: {alert.name}")

class PerformanceMonitor:
    """
    Main performance monitoring system.
    
    Coordinates metrics collection, alerting, and performance analysis.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        query_optimizer: Optional[QueryOptimizer] = None,
        data_pipeline: Optional[DataPipeline] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            cache_manager: Cache manager to monitor
            query_optimizer: Query optimizer to monitor
            data_pipeline: Data pipeline to monitor
        """
        self.cache_manager = cache_manager
        self.query_optimizer = query_optimizer
        self.data_pipeline = data_pipeline
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Custom metrics collection
        self._custom_collectors: List[Callable[[], None]] = []
        
        # Setup alert handlers
        self.alert_manager.add_alert_handler(self._default_alert_handler)
    
    def start(self):
        """Start performance monitoring."""
        self.metrics_collector.start_collection()
        self.alert_manager.start_monitoring()
        
        # Add custom metric collectors
        self._setup_custom_collectors()
        
        self.logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring."""
        self.metrics_collector.stop_collection()
        self.alert_manager.stop_monitoring()
        
        self.logger.info("Performance monitoring stopped")
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard performance metrics."""
        metrics = {
            'system': self.metrics_collector.get_metrics_summary(300),
            'alerts': {
                'active': len(self.alert_manager.get_active_alerts()),
                'recent': len([a for a in self.alert_manager.get_alert_history(100) 
                              if (datetime.now() - a.timestamp).total_seconds() < 3600])
            },
            'components': {}
        }
        
        # Add component-specific metrics
        if self.cache_manager:
            metrics['components']['cache'] = self.cache_manager.get_stats()
        
        if self.query_optimizer:
            metrics['components']['query_optimizer'] = self.query_optimizer.get_query_statistics()
        
        if self.data_pipeline:
            metrics['components']['data_pipeline'] = self.data_pipeline.get_metrics()
        
        return metrics
    
    def add_custom_threshold(self, threshold: PerformanceThreshold):
        """Add custom performance threshold."""
        self.alert_manager.add_threshold(threshold)
    
    def record_custom_metric(self, name: str, value: Union[int, float], metric_type: MetricType = MetricType.GAUGE):
        """Record custom metric."""
        metric = Metric(name=name, value=value, metric_type=metric_type)
        self.metrics_collector.record_metric(metric)
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager for measuring operation performance."""
        with self.metrics_collector.timer(f"operation.{operation_name}.duration"):
            yield
        
        self.metrics_collector.record_counter(f"operation.{operation_name}.count")
    
    def _setup_custom_collectors(self):
        """Setup custom metric collectors."""
        if self.cache_manager:
            self._custom_collectors.append(self._collect_cache_metrics)
        
        if self.query_optimizer:
            self._custom_collectors.append(self._collect_query_metrics)
        
        if self.data_pipeline:
            self._custom_collectors.append(self._collect_pipeline_metrics)
    
    def _collect_cache_metrics(self):
        """Collect cache-specific metrics."""
        try:
            cache_stats = self.cache_manager.get_stats()
            
            # Memory cache metrics
            if 'memory' in cache_stats:
                memory_stats = cache_stats['memory']
                self.metrics_collector.record_gauge('cache.memory.hit_rate', memory_stats.get('hit_rate', 0))
                self.metrics_collector.record_gauge('cache.memory.size', memory_stats.get('size', 0))
                self.metrics_collector.record_counter('cache.memory.hits', memory_stats.get('hits', 0))
                self.metrics_collector.record_counter('cache.memory.misses', memory_stats.get('misses', 0))
            
            # Redis cache metrics
            if 'redis' in cache_stats and isinstance(cache_stats['redis'], dict):
                redis_stats = cache_stats['redis']
                if 'hits' in redis_stats:
                    self.metrics_collector.record_counter('cache.redis.hits', redis_stats['hits'])
                if 'misses' in redis_stats:
                    self.metrics_collector.record_counter('cache.redis.misses', redis_stats['misses'])
                if 'used_memory' in redis_stats:
                    self.metrics_collector.record_gauge('cache.redis.memory_mb', redis_stats['used_memory'] / 1024 / 1024)
            
        except Exception as e:
            self.logger.error(f"Error collecting cache metrics: {e}")
    
    def _collect_query_metrics(self):
        """Collect query optimizer metrics."""
        try:
            query_stats = self.query_optimizer.get_query_statistics()
            
            self.metrics_collector.record_gauge('database.queries.total', query_stats.get('total_queries', 0))
            self.metrics_collector.record_gauge('database.queries.slow', query_stats.get('slow_queries', 0))
            
            if 'performance_summary' in query_stats and query_stats['performance_summary']:
                perf_summary = query_stats['performance_summary']
                self.metrics_collector.record_gauge('database.query.avg_duration', 
                                                  perf_summary.get('avg_query_time', 0))
                self.metrics_collector.record_gauge('database.query.cache_hit_rate', 
                                                  perf_summary.get('cache_hit_rate', 0))
            
        except Exception as e:
            self.logger.error(f"Error collecting query metrics: {e}")
    
    def _collect_pipeline_metrics(self):
        """Collect data pipeline metrics."""
        try:
            pipeline_metrics = self.data_pipeline.get_metrics()
            
            if 'pipeline' in pipeline_metrics:
                pipeline_stats = pipeline_metrics['pipeline']
                self.metrics_collector.record_gauge('pipeline.throughput', 
                                                  pipeline_stats.get('throughput', 0))
                self.metrics_collector.record_counter('pipeline.tasks_processed', 
                                                    pipeline_stats.get('tasks_processed', 0))
                self.metrics_collector.record_counter('pipeline.tasks_failed', 
                                                    pipeline_stats.get('tasks_failed', 0))
            
        except Exception as e:
            self.logger.error(f"Error collecting pipeline metrics: {e}")
    
    def _default_alert_handler(self, alert: Alert):
        """Default alert handler."""
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(f"WARNING ALERT: {alert.message}")
        else:
            self.logger.info(f"INFO ALERT: {alert.message}")

# Global monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        from .cache_manager import get_cache_manager
        from .query_optimizer import get_query_optimizer
        from .data_pipeline import get_data_pipeline
        
        _performance_monitor = PerformanceMonitor(
            cache_manager=get_cache_manager(),
            query_optimizer=get_query_optimizer(),
            data_pipeline=get_data_pipeline()
        )
    return _performance_monitor

# Decorator for monitoring function performance
def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.measure_operation(operation_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator