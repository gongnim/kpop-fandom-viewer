"""
Database Query Optimization Module
==================================

Advanced query optimization techniques for PostgreSQL:
- Intelligent query planning and execution
- Index usage optimization
- Connection pooling management
- Query result caching
- Performance monitoring and analysis
- Batch processing optimization

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
import psycopg2
import psycopg2.extras
import psycopg2.pool
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import json
from collections import defaultdict
import statistics
from contextlib import contextmanager
import hashlib

from ..database_postgresql import get_db_connection
from .cache_manager import CacheManager, cached

# Configure module logger
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type classification."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    AGGREGATION = "AGGREGATION"
    JOIN = "JOIN"

class OptimizationLevel(Enum):
    """Optimization level settings."""
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"

@dataclass
class QueryPlan:
    """Query execution plan analysis."""
    query_hash: str
    plan_nodes: List[Dict[str, Any]]
    total_cost: float
    startup_cost: float
    rows: int
    width: int
    actual_time: Optional[float] = None
    planning_time: Optional[float] = None
    execution_time: Optional[float] = None
    
@dataclass
class QueryStats:
    """Query execution statistics."""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_executed: Optional[datetime] = None
    optimization_applied: bool = False
    cache_hits: int = 0
    
    def update_stats(self, execution_time: float):
        """Update execution statistics."""
        self.execution_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.execution_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_executed = datetime.now()

@dataclass
class IndexRecommendation:
    """Index creation recommendation."""
    table_name: str
    columns: List[str]
    index_type: str  # btree, gin, gist, etc.
    estimated_benefit: float
    query_patterns: List[str]
    size_estimate: int
    creation_sql: str

class QueryOptimizer:
    """
    Advanced database query optimizer.
    
    Provides intelligent query optimization including:
    - Automatic query plan analysis
    - Index usage recommendations
    - Query result caching
    - Connection pool management
    - Performance monitoring
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, optimization_level: OptimizationLevel = OptimizationLevel.BASIC):
        """
        Initialize query optimizer.
        
        Args:
            cache_manager: Cache manager for query results
            optimization_level: Level of optimization to apply
        """
        self.cache_manager = cache_manager
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Query statistics tracking
        self._query_stats: Dict[str, QueryStats] = {}
        self._query_plans: Dict[str, QueryPlan] = {}
        self._stats_lock = threading.RLock()
        
        # Index recommendations
        self._index_recommendations: List[IndexRecommendation] = []
        
        # Performance thresholds
        self.slow_query_threshold = 1.0  # 1 second
        self.cache_threshold = 0.5  # Cache queries taking > 500ms
        
    def execute_optimized_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Execute query with optimization and caching.
        
        Args:
            query: SQL query string
            params: Query parameters
            use_cache: Whether to use result caching
            cache_ttl: Cache time to live in seconds
            
        Returns:
            Query results
        """
        start_time = time.time()
        query_hash = self._get_query_hash(query, params)
        
        try:
            # Try cache first if enabled
            if use_cache and self.cache_manager:
                cached_result = self.cache_manager.get(f"query:{query_hash}")
                if cached_result is not None:
                    self._update_cache_stats(query_hash)
                    return cached_result
            
            # Optimize query before execution
            optimized_query = self._optimize_query(query)
            
            # Execute query
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Analyze query plan if slow query
                if self._should_analyze_query(query_hash):
                    self._analyze_query_plan(cursor, optimized_query, params)
                
                # Execute optimized query
                cursor.execute(optimized_query, params)
                results = [dict(row) for row in cursor.fetchall()]
            
            execution_time = time.time() - start_time
            
            # Update statistics
            self._update_query_stats(query, query_hash, execution_time)
            
            # Cache results if beneficial
            if (use_cache and self.cache_manager and 
                execution_time > self.cache_threshold):
                self.cache_manager.set(f"query:{query_hash}", results, cache_ttl)
            
            # Check for optimization opportunities
            if execution_time > self.slow_query_threshold:
                self._analyze_slow_query(query, query_hash, execution_time)
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Query execution failed: {e}")
            self._update_query_stats(query, query_hash, execution_time, error=True)
            raise
    
    def execute_batch_optimized(
        self,
        queries: List[Tuple[str, Optional[Tuple]]],
        use_transaction: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute multiple queries with batch optimization.
        
        Args:
            queries: List of (query, params) tuples
            use_transaction: Whether to wrap in transaction
            
        Returns:
            List of query results
        """
        start_time = time.time()
        results = []
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                if use_transaction:
                    cursor.execute("BEGIN")
                
                try:
                    for query, params in queries:
                        optimized_query = self._optimize_query(query)
                        cursor.execute(optimized_query, params)
                        
                        if query.strip().upper().startswith('SELECT'):
                            query_results = [dict(row) for row in cursor.fetchall()]
                            results.append(query_results)
                        else:
                            results.append([{'affected_rows': cursor.rowcount}])
                    
                    if use_transaction:
                        cursor.execute("COMMIT")
                        
                except Exception as e:
                    if use_transaction:
                        cursor.execute("ROLLBACK")
                    raise
            
            batch_time = time.time() - start_time
            self.logger.info(f"Batch execution completed: {len(queries)} queries in {batch_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch execution failed: {e}")
            raise
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get comprehensive query performance statistics."""
        with self._stats_lock:
            stats = {
                'total_queries': len(self._query_stats),
                'slow_queries': len([s for s in self._query_stats.values() if s.avg_time > self.slow_query_threshold]),
                'cached_queries': len([s for s in self._query_stats.values() if s.cache_hits > 0]),
                'top_slow_queries': self._get_top_slow_queries(),
                'top_frequent_queries': self._get_top_frequent_queries(),
                'optimization_opportunities': len(self._index_recommendations),
                'performance_summary': self._get_performance_summary()
            }
            
            return stats
    
    def get_index_recommendations(self) -> List[IndexRecommendation]:
        """Get index creation recommendations."""
        return self._index_recommendations.copy()
    
    def apply_recommended_indexes(self, recommendations: Optional[List[IndexRecommendation]] = None):
        """
        Apply recommended indexes to database.
        
        Args:
            recommendations: Specific recommendations to apply (default: all)
        """
        if recommendations is None:
            recommendations = self._index_recommendations
        
        applied_count = 0
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                for rec in recommendations:
                    try:
                        # Check if index already exists
                        if not self._index_exists(cursor, rec.table_name, rec.columns):
                            cursor.execute(rec.creation_sql)
                            applied_count += 1
                            self.logger.info(f"Created index on {rec.table_name}({', '.join(rec.columns)})")
                        else:
                            self.logger.info(f"Index already exists on {rec.table_name}({', '.join(rec.columns)})")
                    
                    except Exception as e:
                        self.logger.error(f"Failed to create index: {rec.creation_sql} - {e}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error applying index recommendations: {e}")
        
        self.logger.info(f"Applied {applied_count} index recommendations")
    
    def clear_query_cache(self, pattern: Optional[str] = None):
        """Clear query result cache."""
        if self.cache_manager:
            if pattern:
                self.cache_manager.invalidate_pattern(f"query:{pattern}")
            else:
                self.cache_manager.invalidate_pattern("query:*")
    
    # Private optimization methods
    
    def _optimize_query(self, query: str) -> str:
        """Apply query optimizations based on level setting."""
        optimized = query.strip()
        
        if self.optimization_level == OptimizationLevel.BASIC:
            optimized = self._apply_basic_optimizations(optimized)
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            optimized = self._apply_aggressive_optimizations(optimized)
        elif self.optimization_level == OptimizationLevel.CONSERVATIVE:
            optimized = self._apply_conservative_optimizations(optimized)
        
        return optimized
    
    def _apply_basic_optimizations(self, query: str) -> str:
        """Apply basic query optimizations."""
        # Add LIMIT if missing in potentially large result sets
        if (query.upper().startswith('SELECT') and 
            'LIMIT' not in query.upper() and 
            'COUNT(' not in query.upper()):
            # Be conservative with automatic limits
            if 'platform_metrics' in query.lower():
                query += ' LIMIT 10000'
        
        # Add appropriate indexes hints for common patterns
        if 'ORDER BY' in query.upper():
            # Query already has ordering, likely needs index
            pass
        
        return query
    
    def _apply_aggressive_optimizations(self, query: str) -> str:
        """Apply aggressive query optimizations."""
        optimized = self._apply_basic_optimizations(query)
        
        # Force index usage where beneficial
        if 'platform_metrics' in optimized.lower() and 'artist_id' in optimized.lower():
            # These queries benefit from artist_id index
            optimized = optimized.replace(
                'FROM platform_metrics',
                'FROM platform_metrics /*+ USE INDEX (platform_metrics, idx_platform_metrics_artist_id) */'
            )
        
        # Add query hints for join optimization
        if 'JOIN' in optimized.upper():
            # Suggest nested loop joins for small result sets
            if 'artists' in optimized.lower() and 'companies' in optimized.lower():
                optimized = '/*+ SET enable_nestloop = on */ ' + optimized
        
        return optimized
    
    def _apply_conservative_optimizations(self, query: str) -> str:
        """Apply conservative query optimizations."""
        # Only apply safe, proven optimizations
        optimized = query.strip()
        
        # Ensure consistent formatting
        optimized = ' '.join(optimized.split())
        
        return optimized
    
    def _get_query_hash(self, query: str, params: Optional[Tuple] = None) -> str:
        """Generate hash for query identification."""
        query_normalized = ' '.join(query.split())  # Normalize whitespace
        hash_input = f"{query_normalized}:{params or ''}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _should_analyze_query(self, query_hash: str) -> bool:
        """Determine if query should be analyzed."""
        with self._stats_lock:
            stats = self._query_stats.get(query_hash)
            if not stats:
                return True  # First execution
            
            # Analyze if slow or frequently executed
            return (stats.avg_time > self.slow_query_threshold or 
                    stats.execution_count > 100)
    
    def _analyze_query_plan(self, cursor, query: str, params: Optional[Tuple] = None):
        """Analyze query execution plan."""
        try:
            # Get execution plan
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            cursor.execute(explain_query, params)
            plan_result = cursor.fetchone()[0]
            
            if plan_result:
                plan = plan_result[0]['Plan']
                query_hash = self._get_query_hash(query, params)
                
                query_plan = QueryPlan(
                    query_hash=query_hash,
                    plan_nodes=self._extract_plan_nodes(plan),
                    total_cost=plan.get('Total Cost', 0),
                    startup_cost=plan.get('Startup Cost', 0),
                    rows=plan.get('Plan Rows', 0),
                    width=plan.get('Plan Width', 0),
                    actual_time=plan.get('Actual Total Time'),
                    planning_time=plan_result[0].get('Planning Time'),
                    execution_time=plan_result[0].get('Execution Time')
                )
                
                self._query_plans[query_hash] = query_plan
                
                # Analyze for optimization opportunities
                self._analyze_plan_for_optimizations(query_plan, query)
                
        except Exception as e:
            self.logger.error(f"Error analyzing query plan: {e}")
    
    def _extract_plan_nodes(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relevant plan nodes for analysis."""
        nodes = []
        
        def extract_node(node):
            nodes.append({
                'node_type': node.get('Node Type'),
                'relation_name': node.get('Relation Name'),
                'index_name': node.get('Index Name'),
                'total_cost': node.get('Total Cost'),
                'rows': node.get('Plan Rows'),
                'actual_time': node.get('Actual Total Time')
            })
            
            # Recursively extract child plans
            if 'Plans' in node:
                for child in node['Plans']:
                    extract_node(child)
        
        extract_node(plan)
        return nodes
    
    def _analyze_plan_for_optimizations(self, plan: QueryPlan, query: str):
        """Analyze execution plan for optimization opportunities."""
        try:
            # Look for sequential scans that could benefit from indexes
            for node in plan.plan_nodes:
                if node['node_type'] == 'Seq Scan' and node.get('relation_name'):
                    self._suggest_index_for_sequential_scan(node, query)
                
                # Look for expensive joins
                if (node['node_type'] in ['Hash Join', 'Nested Loop'] and 
                    node.get('total_cost', 0) > 1000):
                    self._analyze_expensive_join(node, query)
                
                # Look for sorts that could benefit from indexes
                if node['node_type'] == 'Sort' and node.get('total_cost', 0) > 100:
                    self._suggest_index_for_sort(node, query)
                    
        except Exception as e:
            self.logger.error(f"Error analyzing plan for optimizations: {e}")
    
    def _suggest_index_for_sequential_scan(self, node: Dict[str, Any], query: str):
        """Suggest index for sequential scan operations."""
        table_name = node.get('relation_name')
        if not table_name:
            return
        
        # Extract WHERE clause columns
        where_columns = self._extract_where_columns(query, table_name)
        if where_columns:
            recommendation = IndexRecommendation(
                table_name=table_name,
                columns=where_columns,
                index_type='btree',
                estimated_benefit=node.get('total_cost', 0) * 0.8,
                query_patterns=[query[:100] + '...' if len(query) > 100 else query],
                size_estimate=len(where_columns) * 1000000,  # Rough estimate
                creation_sql=f"CREATE INDEX idx_{table_name}_{'_'.join(where_columns)} ON {table_name} ({', '.join(where_columns)})"
            )
            
            # Check if similar recommendation already exists
            if not any(r.table_name == table_name and r.columns == where_columns 
                      for r in self._index_recommendations):
                self._index_recommendations.append(recommendation)
    
    def _extract_where_columns(self, query: str, table_name: str) -> List[str]:
        """Extract columns used in WHERE clause for specific table."""
        # Simplified column extraction - in production would use SQL parser
        columns = []
        query_lower = query.lower()
        
        # Look for common patterns
        common_columns = ['artist_id', 'company_id', 'group_id', 'platform', 'collected_at']
        for col in common_columns:
            if f"{col} =" in query_lower or f"{col} IN" in query_lower:
                columns.append(col)
        
        return columns[:3]  # Limit to 3 columns for compound index
    
    def _update_query_stats(self, query: str, query_hash: str, execution_time: float, error: bool = False):
        """Update query execution statistics."""
        with self._stats_lock:
            if query_hash not in self._query_stats:
                self._query_stats[query_hash] = QueryStats(
                    query_hash=query_hash,
                    query_text=query[:200] + '...' if len(query) > 200 else query
                )
            
            if not error:
                self._query_stats[query_hash].update_stats(execution_time)
    
    def _update_cache_stats(self, query_hash: str):
        """Update cache hit statistics."""
        with self._stats_lock:
            if query_hash in self._query_stats:
                self._query_stats[query_hash].cache_hits += 1
    
    def _analyze_slow_query(self, query: str, query_hash: str, execution_time: float):
        """Analyze slow query for optimization opportunities."""
        self.logger.warning(f"Slow query detected: {execution_time:.3f}s - {query[:100]}")
        
        # Add to recommendations if not already analyzed
        with self._stats_lock:
            stats = self._query_stats.get(query_hash)
            if stats and not stats.optimization_applied:
                # Mark as analyzed to avoid repeated analysis
                stats.optimization_applied = True
                
                # Trigger index analysis
                self._suggest_optimizations_for_slow_query(query, execution_time)
    
    def _suggest_optimizations_for_slow_query(self, query: str, execution_time: float):
        """Suggest optimizations for slow query."""
        # This would contain more sophisticated analysis
        # For now, basic suggestions based on query patterns
        
        if 'ORDER BY' in query.upper() and 'LIMIT' not in query.upper():
            self.logger.info(f"Suggestion: Add LIMIT clause to ORDER BY query")
        
        if 'platform_metrics' in query.lower() and 'collected_at' in query.lower():
            self.logger.info(f"Suggestion: Consider partitioning platform_metrics by collected_at")
        
        if execution_time > 5.0:
            self.logger.warning(f"Critical: Query taking {execution_time:.3f}s may need immediate optimization")
    
    def _get_top_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top slow queries by average execution time."""
        with self._stats_lock:
            sorted_queries = sorted(
                self._query_stats.values(),
                key=lambda x: x.avg_time,
                reverse=True
            )
            
            return [
                {
                    'query_hash': q.query_hash,
                    'query_text': q.query_text,
                    'avg_time': q.avg_time,
                    'execution_count': q.execution_count,
                    'total_time': q.total_time
                }
                for q in sorted_queries[:limit]
            ]
    
    def _get_top_frequent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently executed queries."""
        with self._stats_lock:
            sorted_queries = sorted(
                self._query_stats.values(),
                key=lambda x: x.execution_count,
                reverse=True
            )
            
            return [
                {
                    'query_hash': q.query_hash,
                    'query_text': q.query_text,
                    'execution_count': q.execution_count,
                    'avg_time': q.avg_time,
                    'cache_hits': q.cache_hits
                }
                for q in sorted_queries[:limit]
            ]
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        with self._stats_lock:
            if not self._query_stats:
                return {}
            
            all_times = [s.avg_time for s in self._query_stats.values()]
            execution_counts = [s.execution_count for s in self._query_stats.values()]
            cache_hits = sum(s.cache_hits for s in self._query_stats.values())
            total_executions = sum(execution_counts)
            
            return {
                'avg_query_time': statistics.mean(all_times),
                'median_query_time': statistics.median(all_times),
                'total_executions': total_executions,
                'cache_hit_rate': (cache_hits / total_executions * 100) if total_executions > 0 else 0,
                'queries_optimized': sum(1 for s in self._query_stats.values() if s.optimization_applied)
            }
    
    def _index_exists(self, cursor, table_name: str, columns: List[str]) -> bool:
        """Check if index exists on specified columns."""
        try:
            query = """
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = %s 
                AND indexdef LIKE %s
            """
            
            # Create pattern to match column list
            column_pattern = f"%({', '.join(columns)})%"
            
            cursor.execute(query, (table_name, column_pattern))
            return cursor.fetchone() is not None
            
        except Exception as e:
            self.logger.error(f"Error checking index existence: {e}")
            return False

# Global optimizer instance
_query_optimizer = None

def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer
    if _query_optimizer is None:
        from .cache_manager import get_cache_manager
        _query_optimizer = QueryOptimizer(
            cache_manager=get_cache_manager(),
            optimization_level=OptimizationLevel.BASIC
        )
    return _query_optimizer

# Decorator for automatic query optimization
def optimized_query(cache_ttl: int = 300, use_cache: bool = True):
    """
    Decorator for automatic query optimization.
    
    Args:
        cache_ttl: Cache time to live in seconds
        use_cache: Whether to use result caching
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            optimizer = get_query_optimizer()
            
            # Extract query and params from function
            if len(args) >= 2:
                query, params = args[0], args[1]
                return optimizer.execute_optimized_query(
                    query, params, use_cache, cache_ttl
                )
            else:
                # Fallback to original function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator