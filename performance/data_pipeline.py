"""
Real-time Data Processing Pipeline Optimization
=============================================

High-performance data processing pipeline for K-POP analytics:
- Asynchronous data processing
- Batch optimization and aggregation
- Stream processing capabilities
- Resource-aware processing
- Pipeline monitoring and metrics
- Error handling and recovery

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
from collections import defaultdict, deque
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import psutil
import weakref

from .cache_manager import CacheManager
from .query_optimizer import QueryOptimizer
from ..database_postgresql import get_db_connection

# Configure module logger
logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline processing stages."""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    STORAGE = "storage"
    INDEXING = "indexing"

class ProcessingMode(Enum):
    """Data processing modes."""
    BATCH = "batch"
    STREAM = "stream"
    HYBRID = "hybrid"

class Priority(Enum):
    """Processing priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ProcessingTask:
    """Individual processing task."""
    task_id: str
    data: Any
    stage: PipelineStage
    priority: Priority = Priority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    retries: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchConfig:
    """Batch processing configuration."""
    batch_size: int = 1000
    timeout_seconds: float = 60.0
    max_memory_mb: int = 500
    parallel_workers: int = 4
    enable_compression: bool = True

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    tasks_processed: int = 0
    tasks_failed: int = 0
    avg_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_size: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

class DataProcessor:
    """Base class for data processing components."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.metrics = PipelineMetrics()
        self._processing_times = deque(maxlen=1000)
    
    async def process(self, task: ProcessingTask) -> ProcessingTask:
        """Process a single task."""
        start_time = time.time()
        
        try:
            result = await self._process_implementation(task)
            processing_time = time.time() - start_time
            
            self._update_metrics(processing_time, success=True)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=False)
            self.logger.error(f"Processing failed for task {task.task_id}: {e}")
            raise
    
    async def _process_implementation(self, task: ProcessingTask) -> ProcessingTask:
        """Override in subclasses."""
        raise NotImplementedError
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update processing metrics."""
        self._processing_times.append(processing_time)
        
        if success:
            self.metrics.tasks_processed += 1
        else:
            self.metrics.tasks_failed += 1
        
        if self._processing_times:
            self.metrics.avg_processing_time = sum(self._processing_times) / len(self._processing_times)
        
        self.metrics.last_updated = datetime.now()

class IngestionProcessor(DataProcessor):
    """Data ingestion processor."""
    
    def __init__(self):
        super().__init__("IngestionProcessor")
        self.cache_manager = None
    
    async def _process_implementation(self, task: ProcessingTask) -> ProcessingTask:
        """Process data ingestion."""
        # Simulate data ingestion processing
        await asyncio.sleep(0.01)  # Simulate I/O
        
        # Add timestamp and source metadata
        task.metadata.update({
            'ingested_at': datetime.now().isoformat(),
            'source': task.metadata.get('source', 'unknown'),
            'raw_size': len(str(task.data))
        })
        
        task.stage = PipelineStage.VALIDATION
        return task

class ValidationProcessor(DataProcessor):
    """Data validation processor."""
    
    def __init__(self):
        super().__init__("ValidationProcessor")
    
    async def _process_implementation(self, task: ProcessingTask) -> ProcessingTask:
        """Process data validation."""
        # Simulate validation logic
        data = task.data
        
        if not isinstance(data, dict):
            raise ValueError("Invalid data format - expected dictionary")
        
        required_fields = ['artist_id', 'platform', 'metric_name', 'metric_value']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate data types
        if not isinstance(data.get('metric_value'), (int, float)):
            raise ValueError("metric_value must be numeric")
        
        task.metadata['validation_passed'] = True
        task.stage = PipelineStage.TRANSFORMATION
        return task

class TransformationProcessor(DataProcessor):
    """Data transformation processor."""
    
    def __init__(self):
        super().__init__("TransformationProcessor")
    
    async def _process_implementation(self, task: ProcessingTask) -> ProcessingTask:
        """Process data transformation."""
        data = task.data.copy()
        
        # Normalize platform names
        platform_mapping = {
            'yt': 'youtube',
            'ig': 'instagram',
            'tw': 'twitter',
            'sp': 'spotify'
        }
        
        if data.get('platform') in platform_mapping:
            data['platform'] = platform_mapping[data['platform']]
        
        # Add derived metrics
        if data.get('metric_name') == 'subscribers' and data.get('metric_value'):
            data['subscriber_tier'] = self._calculate_subscriber_tier(data['metric_value'])
        
        # Add collection timestamp
        data['collected_at'] = task.metadata.get('ingested_at', datetime.now().isoformat())
        
        task.data = data
        task.metadata['transformation_applied'] = True
        task.stage = PipelineStage.AGGREGATION
        return task
    
    def _calculate_subscriber_tier(self, subscribers: Union[int, float]) -> str:
        """Calculate subscriber tier for analytics."""
        if subscribers >= 10000000:  # 10M+
            return 'mega'
        elif subscribers >= 1000000:  # 1M+
            return 'major'
        elif subscribers >= 100000:   # 100K+
            return 'rising'
        else:
            return 'emerging'

class AggregationProcessor(DataProcessor):
    """Data aggregation processor."""
    
    def __init__(self):
        super().__init__("AggregationProcessor")
        self._aggregation_buffer = defaultdict(list)
        self._buffer_lock = threading.Lock()
    
    async def _process_implementation(self, task: ProcessingTask) -> ProcessingTask:
        """Process data aggregation."""
        data = task.data
        
        # Group by artist and platform for aggregation
        agg_key = f"{data.get('artist_id')}:{data.get('platform')}"
        
        with self._buffer_lock:
            self._aggregation_buffer[agg_key].append(data)
        
        # Calculate running aggregates
        task.metadata['aggregation_key'] = agg_key
        task.metadata['aggregation_ready'] = len(self._aggregation_buffer[agg_key]) >= 5
        task.stage = PipelineStage.STORAGE
        
        return task

class StorageProcessor(DataProcessor):
    """Data storage processor."""
    
    def __init__(self, query_optimizer: Optional[QueryOptimizer] = None):
        super().__init__("StorageProcessor")
        self.query_optimizer = query_optimizer
        self._batch_buffer = []
        self._buffer_lock = threading.Lock()
    
    async def _process_implementation(self, task: ProcessingTask) -> ProcessingTask:
        """Process data storage."""
        with self._buffer_lock:
            self._batch_buffer.append(task.data)
            
            # Batch insert when buffer is full
            if len(self._batch_buffer) >= 100:
                await self._flush_batch()
        
        task.metadata['stored_at'] = datetime.now().isoformat()
        task.stage = PipelineStage.INDEXING
        return task
    
    async def _flush_batch(self):
        """Flush batch buffer to database."""
        if not self._batch_buffer:
            return
        
        try:
            # Prepare batch insert
            insert_query = """
                INSERT INTO platform_metrics 
                (artist_id, platform, metric_name, metric_value, collected_at)
                VALUES %s
                ON CONFLICT (artist_id, platform, metric_name, collected_at) 
                DO UPDATE SET metric_value = EXCLUDED.metric_value
            """
            
            values = [
                (
                    data['artist_id'],
                    data['platform'],
                    data['metric_name'],
                    data['metric_value'],
                    data['collected_at']
                )
                for data in self._batch_buffer
            ]
            
            # Execute batch insert
            with get_db_connection() as conn:
                cursor = conn.cursor()
                psycopg2.extras.execute_values(cursor, insert_query, values)
                conn.commit()
            
            self.logger.info(f"Batch inserted {len(self._batch_buffer)} records")
            self._batch_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            # Keep data in buffer for retry
            raise

class DataPipeline:
    """
    High-performance data processing pipeline.
    
    Orchestrates multiple processing stages with:
    - Asynchronous processing
    - Batch optimization
    - Resource monitoring
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        batch_config: Optional[BatchConfig] = None,
        processing_mode: ProcessingMode = ProcessingMode.HYBRID,
        max_concurrent_tasks: int = 100
    ):
        """
        Initialize data pipeline.
        
        Args:
            batch_config: Batch processing configuration
            processing_mode: Processing mode (batch, stream, hybrid)
            max_concurrent_tasks: Maximum concurrent processing tasks
        """
        self.batch_config = batch_config or BatchConfig()
        self.processing_mode = processing_mode
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize processors
        self.processors = {
            PipelineStage.INGESTION: IngestionProcessor(),
            PipelineStage.VALIDATION: ValidationProcessor(),
            PipelineStage.TRANSFORMATION: TransformationProcessor(),
            PipelineStage.AGGREGATION: AggregationProcessor(),
            PipelineStage.STORAGE: StorageProcessor(),
        }
        
        # Processing queues
        self._task_queue = asyncio.Queue(maxsize=10000)
        self._priority_queue = asyncio.PriorityQueue(maxsize=1000)
        self._batch_queue = queue.Queue(maxsize=1000)
        
        # Processing control
        self._processing = False
        self._workers = []
        self._batch_worker = None
        
        # Metrics and monitoring
        self.pipeline_metrics = PipelineMetrics()
        self._start_time = datetime.now()
        
        # Resource monitoring
        self._monitor_resources = True
        self._resource_monitor_thread = None
    
    async def start(self):
        """Start the data processing pipeline."""
        if self._processing:
            return
        
        self._processing = True
        self._start_time = datetime.now()
        
        # Start async workers
        for i in range(self.batch_config.parallel_workers):
            worker = asyncio.create_task(self._async_worker(f"worker-{i}"))
            self._workers.append(worker)
        
        # Start batch processor
        self._batch_worker = threading.Thread(
            target=self._batch_worker_thread,
            daemon=True
        )
        self._batch_worker.start()
        
        # Start resource monitoring
        if self._monitor_resources:
            self._resource_monitor_thread = threading.Thread(
                target=self._resource_monitor_thread_func,
                daemon=True
            )
            self._resource_monitor_thread.start()
        
        self.logger.info("Data pipeline started successfully")
    
    async def stop(self):
        """Stop the data processing pipeline."""
        self._processing = False
        
        # Stop async workers
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        # Process remaining tasks
        await self._process_remaining_tasks()
        
        self.logger.info("Data pipeline stopped")
    
    async def submit_task(self, task: ProcessingTask):
        """Submit task for processing."""
        if not self._processing:
            raise RuntimeError("Pipeline is not running")
        
        if task.priority in [Priority.HIGH, Priority.CRITICAL]:
            await self._priority_queue.put((task.priority.value, task))
        else:
            await self._task_queue.put(task)
    
    async def submit_batch(self, tasks: List[ProcessingTask]):
        """Submit batch of tasks for processing."""
        if self.processing_mode in [ProcessingMode.BATCH, ProcessingMode.HYBRID]:
            # Add to batch queue
            for task in tasks:
                self._batch_queue.put(task, block=False)
        else:
            # Process individually
            for task in tasks:
                await self.submit_task(task)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        runtime = (datetime.now() - self._start_time).total_seconds()
        
        metrics = {
            'pipeline': {
                'runtime_seconds': runtime,
                'tasks_processed': self.pipeline_metrics.tasks_processed,
                'tasks_failed': self.pipeline_metrics.tasks_failed,
                'throughput': self.pipeline_metrics.tasks_processed / runtime if runtime > 0 else 0,
                'queue_sizes': {
                    'task_queue': self._task_queue.qsize(),
                    'priority_queue': self._priority_queue.qsize(),
                    'batch_queue': self._batch_queue.qsize()
                }
            },
            'processors': {},
            'resources': {
                'memory_usage_mb': self.pipeline_metrics.memory_usage_mb,
                'cpu_usage_percent': self.pipeline_metrics.cpu_usage_percent
            }
        }
        
        # Add processor-specific metrics
        for stage, processor in self.processors.items():
            metrics['processors'][stage.value] = {
                'tasks_processed': processor.metrics.tasks_processed,
                'tasks_failed': processor.metrics.tasks_failed,
                'avg_processing_time': processor.metrics.avg_processing_time
            }
        
        return metrics
    
    # Private methods
    
    async def _async_worker(self, worker_name: str):
        """Asynchronous worker for processing tasks."""
        self.logger.info(f"Started async worker: {worker_name}")
        
        while self._processing:
            try:
                # Check priority queue first
                try:
                    priority, task = await asyncio.wait_for(
                        self._priority_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Check regular queue
                    try:
                        task = await asyncio.wait_for(
                            self._task_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                
                # Process task through pipeline stages
                await self._process_task(task)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        self.logger.info(f"Stopped async worker: {worker_name}")
    
    async def _process_task(self, task: ProcessingTask):
        """Process task through all pipeline stages."""
        start_time = time.time()
        
        try:
            current_task = task
            stage_order = [
                PipelineStage.INGESTION,
                PipelineStage.VALIDATION,
                PipelineStage.TRANSFORMATION,
                PipelineStage.AGGREGATION,
                PipelineStage.STORAGE
            ]
            
            for stage in stage_order:
                if stage in self.processors:
                    processor = self.processors[stage]
                    current_task = await processor.process(current_task)
            
            # Update pipeline metrics
            processing_time = time.time() - start_time
            self.pipeline_metrics.tasks_processed += 1
            self._update_pipeline_metrics(processing_time)
            
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            self.pipeline_metrics.tasks_failed += 1
            
            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                await asyncio.sleep(2 ** task.retries)  # Exponential backoff
                await self.submit_task(task)
    
    def _batch_worker_thread(self):
        """Batch processing worker thread."""
        self.logger.info("Started batch processing worker")
        
        batch = []
        last_flush = time.time()
        
        while self._processing:
            try:
                # Collect batch
                try:
                    task = self._batch_queue.get(timeout=1.0)
                    batch.append(task)
                except queue.Empty:
                    pass
                
                # Flush batch if conditions met
                should_flush = (
                    len(batch) >= self.batch_config.batch_size or
                    (batch and time.time() - last_flush > self.batch_config.timeout_seconds) or
                    self._get_memory_usage() > self.batch_config.max_memory_mb
                )
                
                if should_flush and batch:
                    asyncio.run(self._process_batch(batch))
                    batch.clear()
                    last_flush = time.time()
                
            except Exception as e:
                self.logger.error(f"Batch worker error: {e}")
                time.sleep(1)
        
        # Process remaining batch
        if batch:
            asyncio.run(self._process_batch(batch))
        
        self.logger.info("Stopped batch processing worker")
    
    async def _process_batch(self, batch: List[ProcessingTask]):
        """Process batch of tasks efficiently."""
        self.logger.info(f"Processing batch of {len(batch)} tasks")
        
        # Group tasks by stage for efficient processing
        stage_groups = defaultdict(list)
        for task in batch:
            stage_groups[task.stage].append(task)
        
        # Process each stage group
        for stage, tasks in stage_groups.items():
            if stage in self.processors:
                processor = self.processors[stage]
                
                # Process tasks concurrently within stage
                semaphore = asyncio.Semaphore(self.batch_config.parallel_workers)
                
                async def process_with_semaphore(task):
                    async with semaphore:
                        return await processor.process(task)
                
                await asyncio.gather(*[
                    process_with_semaphore(task) for task in tasks
                ], return_exceptions=True)
    
    def _resource_monitor_thread_func(self):
        """Monitor system resources."""
        while self._processing:
            try:
                # Get current resource usage
                process = psutil.Process()
                memory_info = process.memory_info()
                
                self.pipeline_metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
                self.pipeline_metrics.cpu_usage_percent = process.cpu_percent()
                
                # Check resource limits
                if self.pipeline_metrics.memory_usage_mb > 1000:  # 1GB
                    self.logger.warning(f"High memory usage: {self.pipeline_metrics.memory_usage_mb:.1f}MB")
                
                if self.pipeline_metrics.cpu_usage_percent > 80:
                    self.logger.warning(f"High CPU usage: {self.pipeline_metrics.cpu_usage_percent:.1f}%")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(60)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _update_pipeline_metrics(self, processing_time: float):
        """Update pipeline-level metrics."""
        # Update throughput calculation
        runtime = (datetime.now() - self._start_time).total_seconds()
        if runtime > 0:
            self.pipeline_metrics.throughput_per_second = self.pipeline_metrics.tasks_processed / runtime
        
        # Update queue sizes
        self.pipeline_metrics.queue_size = (
            self._task_queue.qsize() + 
            self._priority_queue.qsize() + 
            self._batch_queue.qsize()
        )
        
        self.pipeline_metrics.last_updated = datetime.now()
    
    async def _process_remaining_tasks(self):
        """Process remaining tasks during shutdown."""
        remaining_tasks = 0
        
        # Process priority queue
        while not self._priority_queue.empty():
            try:
                _, task = await asyncio.wait_for(
                    self._priority_queue.get(),
                    timeout=1.0
                )
                await self._process_task(task)
                remaining_tasks += 1
            except asyncio.TimeoutError:
                break
        
        # Process regular queue
        while not self._task_queue.empty():
            try:
                task = await asyncio.wait_for(
                    self._task_queue.get(),
                    timeout=1.0
                )
                await self._process_task(task)
                remaining_tasks += 1
            except asyncio.TimeoutError:
                break
        
        if remaining_tasks > 0:
            self.logger.info(f"Processed {remaining_tasks} remaining tasks during shutdown")

# Global pipeline instance
_data_pipeline = None

def get_data_pipeline() -> DataPipeline:
    """Get global data pipeline instance."""
    global _data_pipeline
    if _data_pipeline is None:
        _data_pipeline = DataPipeline()
    return _data_pipeline