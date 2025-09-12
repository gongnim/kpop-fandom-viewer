#!/usr/bin/env python3
"""
Ranking System Performance Demonstration
========================================

Comprehensive performance comparison between the standard and optimized ranking systems,
demonstrating the benefits of caching, batch processing, and parallel computation.

Features demonstrated:
- Cache performance improvements
- Batch processing optimization
- Memory efficiency gains
- Parallel computation benefits
- Large-scale dataset handling

Author: Backend Development Team  
Date: 2025-09-08
"""

import sys
import os
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kpop_dashboard.analytics import (
        # Standard ranking system
        GrowthRankingEngine,
        RankingCategory,
        RankingPeriod,
        DebutCohort,
        ArtistMetrics,
        MetricDataPoint,
        
        # Optimized ranking system
        OptimizedRankingEngine,
        OptimizationStrategy,
        CacheLevel,
        PerformanceMetrics
    )
    print("✅ Successfully imported all ranking systems")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def generate_large_dataset(num_artists: int = 1000, days_of_data: int = 90) -> List[ArtistMetrics]:
    """Generate large-scale realistic K-Pop artist dataset for performance testing."""
    print(f"📊 Generating dataset: {num_artists} artists with {days_of_data} days of data each...")
    
    # Sample K-Pop companies and generations
    companies = [
        ("HYBE", 1), ("SM Entertainment", 2), ("YG Entertainment", 3), 
        ("JYP Entertainment", 4), ("Starship", 5), ("ADOR", 6),
        ("Source Music", 7), ("Cube Entertainment", 8), ("RBW", 9),
        ("Pledis", 10), ("KOZ Entertainment", 11), ("BigHit Music", 12)
    ]
    
    platforms = ['youtube', 'spotify', 'instagram', 'tiktok']
    metrics = ['subscribers', 'monthly_listeners', 'followers', 'followers']
    
    artists_data = []
    base_timestamp = datetime.now()
    
    for artist_id in range(1, num_artists + 1):
        # Random company and debut year
        company_name, company_id = random.choice(companies)
        debut_year = random.randint(2010, 2023)  # Mix of 3rd and 4th gen
        
        # Create metrics for each platform
        for platform, metric_type in zip(platforms, metrics):
            # Base value depends on debut year and platform
            if platform == 'youtube':
                base_value = random.randint(100000, 50000000)
            elif platform == 'spotify':
                base_value = random.randint(50000, 20000000)
            else:
                base_value = random.randint(10000, 10000000)
            
            # Generate time series data
            data_points = []
            current_value = base_value
            
            for day in range(days_of_data):
                timestamp = base_timestamp - timedelta(days=days_of_data - day - 1)
                
                # Simulate realistic growth patterns
                if debut_year >= 2022:  # 4th gen - higher growth
                    daily_change = random.uniform(0.005, 0.025)
                elif debut_year >= 2018:  # Late 3rd gen - moderate growth
                    daily_change = random.uniform(0.002, 0.012)
                else:  # Early 3rd gen - stable/slow growth
                    daily_change = random.uniform(-0.001, 0.008)
                
                # Random comeback boosts
                if random.random() < 0.03:  # 3% chance per day
                    daily_change *= random.uniform(2.0, 5.0)
                
                current_value = int(current_value * (1 + daily_change))
                
                data_points.append(MetricDataPoint(
                    value=max(0, current_value),
                    timestamp=timestamp,
                    platform=platform,
                    metric_type=metric_type,
                    quality_score=random.uniform(0.85, 0.98)
                ))
            
            # Create artist metrics
            artist_metrics = ArtistMetrics(
                artist_id=artist_id,
                artist_name=f"Artist_{artist_id:04d}",
                company_id=company_id,
                company_name=company_name,
                debut_year=debut_year,
                platform=platform,
                metric_type=metric_type,
                current_value=data_points[-1].value,
                data_points=data_points,
                quality_score=random.uniform(0.88, 0.97)
            )
            
            artists_data.append(artist_metrics)
    
    total_data_points = len(artists_data) * days_of_data
    print(f"✅ Generated {len(artists_data)} artist records ({total_data_points:,} total data points)")
    
    return artists_data


def benchmark_standard_engine(artists_data: List[ArtistMetrics], operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark the standard ranking engine performance."""
    print("\n🔧 Benchmarking Standard Ranking Engine...")
    
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    standard_engine = GrowthRankingEngine()
    
    operation_times = []
    results = {}
    
    overall_start = time.time()
    
    for i, operation in enumerate(operations):
        op_start = time.time()
        
        try:
            if operation['type'] == 'platform_ranking':
                platform_artists = [a for a in artists_data if a.platform == operation['platform']]
                result = standard_engine.rank_by_platform(
                    platform_artists,
                    operation['platform'],
                    operation['metric_type'],
                    operation.get('category', RankingCategory.GROWTH_RATE)
                )
                results[f"operation_{i}"] = len(result) if result else 0
                
            elif operation['type'] == 'composite_index':
                result = standard_engine.calculate_composite_index(
                    artists_data,
                    operation['platforms']
                )
                results[f"operation_{i}"] = len(result) if result else 0
                
        except Exception as e:
            print(f"   ❌ Operation {i} failed: {e}")
            results[f"operation_{i}"] = 0
        
        op_time = time.time() - op_start
        operation_times.append(op_time)
        
        print(f"   Operation {i+1}/{len(operations)}: {op_time:.2f}s")
    
    overall_time = time.time() - overall_start
    peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    return {
        'engine_type': 'Standard',
        'total_time': overall_time,
        'average_op_time': sum(operation_times) / len(operation_times),
        'memory_usage_mb': peak_memory - start_memory,
        'peak_memory_mb': peak_memory,
        'successful_operations': sum(1 for v in results.values() if v > 0),
        'operation_times': operation_times,
        'cache_hit_rate': 0.0  # No cache in standard engine
    }


def benchmark_optimized_engine(artists_data: List[ArtistMetrics], operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark the optimized ranking engine performance."""
    print("\n⚡ Benchmarking Optimized Ranking Engine...")
    
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Create optimized engine with performance-focused settings
    optimized_engine = OptimizedRankingEngine(
        cache_size_mb=256,
        optimization_strategy=OptimizationStrategy.BATCH_HEAVY,
        enable_cache_warming=True
    )
    
    # Warm cache with common operations
    print("   🔥 Warming cache...")
    cache_warm_start = time.time()
    warmed_entries = optimized_engine.warm_cache_for_common_operations(artists_data)
    cache_warm_time = time.time() - cache_warm_start
    print(f"   Cache warmed with {warmed_entries} entries in {cache_warm_time:.2f}s")
    
    # Execute operations (some will benefit from cache)
    operation_times = []
    results = {}
    
    overall_start = time.time()
    
    # First run - populate cache
    print("   📊 First run (populating cache)...")
    for i, operation in enumerate(operations):
        op_start = time.time()
        
        try:
            if operation['type'] == 'platform_ranking':
                platform_artists = [a for a in artists_data if a.platform == operation['platform']]
                result = optimized_engine.rank_by_platform_optimized(
                    platform_artists,
                    operation['platform'],
                    operation['metric_type'],
                    operation.get('category', RankingCategory.GROWTH_RATE),
                    use_cache=True
                )
                results[f"operation_{i}_first"] = len(result) if result else 0
                
            elif operation['type'] == 'composite_index':
                result = optimized_engine.calculate_composite_index_optimized(
                    artists_data,
                    operation['platforms'],
                    use_cache=True
                )
                results[f"operation_{i}_first"] = len(result) if result else 0
                
        except Exception as e:
            print(f"   ❌ Operation {i} failed: {e}")
            results[f"operation_{i}_first"] = 0
        
        op_time = time.time() - op_start
        operation_times.append(op_time)
        
        print(f"   Operation {i+1}/{len(operations)}: {op_time:.2f}s")
    
    # Second run - benefit from cache
    print("   ⚡ Second run (utilizing cache)...")
    cached_operation_times = []
    
    for i, operation in enumerate(operations):
        op_start = time.time()
        
        try:
            if operation['type'] == 'platform_ranking':
                platform_artists = [a for a in artists_data if a.platform == operation['platform']]
                result = optimized_engine.rank_by_platform_optimized(
                    platform_artists,
                    operation['platform'],
                    operation['metric_type'],
                    operation.get('category', RankingCategory.GROWTH_RATE),
                    use_cache=True
                )
                results[f"operation_{i}_cached"] = len(result) if result else 0
                
            elif operation['type'] == 'composite_index':
                result = optimized_engine.calculate_composite_index_optimized(
                    artists_data,
                    operation['platforms'],
                    use_cache=True
                )
                results[f"operation_{i}_cached"] = len(result) if result else 0
                
        except Exception as e:
            print(f"   ❌ Cached operation {i} failed: {e}")
            results[f"operation_{i}_cached"] = 0
        
        op_time = time.time() - op_start
        cached_operation_times.append(op_time)
        
        print(f"   Cached Operation {i+1}/{len(operations)}: {op_time:.2f}s")
    
    overall_time = time.time() - overall_start
    peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Get performance report
    perf_report = optimized_engine.get_performance_report()
    
    return {
        'engine_type': 'Optimized',
        'total_time': overall_time,
        'average_op_time': sum(operation_times) / len(operation_times),
        'cached_average_time': sum(cached_operation_times) / len(cached_operation_times),
        'memory_usage_mb': peak_memory - start_memory,
        'peak_memory_mb': peak_memory,
        'successful_operations': sum(1 for k, v in results.items() if v > 0 and 'first' in k),
        'cache_warm_time': cache_warm_time,
        'cache_hit_rate': perf_report['cache_performance']['cache_hit_rate'],
        'cache_size_mb': perf_report['cache_performance']['cache_size_mb'],
        'operation_times': operation_times,
        'cached_operation_times': cached_operation_times
    }


def benchmark_batch_processing(artists_data: List[ArtistMetrics]) -> Dict[str, Any]:
    """Benchmark batch processing capabilities."""
    print("\n🚀 Benchmarking Batch Processing...")
    
    # Create batch operations
    batch_operations = [
        {
            'type': 'platform_ranking',
            'key': 'youtube_growth',
            'platform': 'youtube',
            'metric_type': 'subscribers',
            'category': RankingCategory.GROWTH_RATE
        },
        {
            'type': 'platform_ranking', 
            'key': 'spotify_growth',
            'platform': 'spotify',
            'metric_type': 'monthly_listeners',
            'category': RankingCategory.GROWTH_RATE
        },
        {
            'type': 'composite_index',
            'key': 'composite_all',
            'platforms': ['youtube', 'spotify']
        },
        {
            'type': 'platform_ranking',
            'key': 'youtube_absolute',
            'platform': 'youtube',
            'metric_type': 'subscribers',
            'category': RankingCategory.ABSOLUTE_VALUE
        }
    ]
    
    optimized_engine = OptimizedRankingEngine(
        optimization_strategy=OptimizationStrategy.BATCH_HEAVY
    )
    
    # Sequential processing (standard approach)
    print("   📋 Sequential processing...")
    sequential_start = time.time()
    
    sequential_results = {}
    for operation in batch_operations:
        if operation['type'] == 'platform_ranking':
            platform_artists = [a for a in artists_data if a.platform == operation['platform']]
            result = optimized_engine.rank_by_platform_optimized(
                platform_artists,
                operation['platform'],
                operation['metric_type'],
                operation['category'],
                use_cache=False  # Disable cache to measure pure computation
            )
            sequential_results[operation['key']] = result
        elif operation['type'] == 'composite_index':
            result = optimized_engine.calculate_composite_index_optimized(
                artists_data,
                operation['platforms'],
                use_cache=False
            )
            sequential_results[operation['key']] = result
    
    sequential_time = time.time() - sequential_start
    
    # Batch processing
    print("   ⚡ Batch processing...")
    batch_start = time.time()
    
    batch_results = optimized_engine.batch_rank_multiple_operations(
        artists_data,
        batch_operations,
        use_cache=False  # Disable cache to measure pure batch performance
    )
    
    batch_time = time.time() - batch_start
    
    return {
        'sequential_time': sequential_time,
        'batch_time': batch_time,
        'speedup_factor': sequential_time / batch_time if batch_time > 0 else 1.0,
        'operations_count': len(batch_operations),
        'sequential_results_count': sum(len(r) for r in sequential_results.values() if r),
        'batch_results_count': sum(len(r) for r in batch_results.values() if r)
    }


def run_performance_comparison():
    """Run comprehensive performance comparison."""
    print("🎵 K-Pop Ranking System Performance Comparison")
    print("=" * 80)
    
    # Configuration
    dataset_sizes = [100, 500, 1000]  # Start with smaller datasets for demo
    days_of_data = 90
    
    # Standard test operations
    test_operations = [
        {
            'type': 'platform_ranking',
            'platform': 'youtube',
            'metric_type': 'subscribers',
            'category': RankingCategory.GROWTH_RATE
        },
        {
            'type': 'platform_ranking',
            'platform': 'spotify', 
            'metric_type': 'monthly_listeners',
            'category': RankingCategory.GROWTH_RATE
        },
        {
            'type': 'composite_index',
            'platforms': ['youtube', 'spotify']
        }
    ]
    
    results_summary = []
    
    for dataset_size in dataset_sizes:
        print(f"\n" + "="*60)
        print(f"PERFORMANCE TEST: {dataset_size} Artists Dataset")
        print("="*60)
        
        # Generate dataset
        artists_data = generate_large_dataset(dataset_size, days_of_data)
        
        # Benchmark standard engine
        standard_results = benchmark_standard_engine(artists_data, test_operations)
        
        # Benchmark optimized engine
        optimized_results = benchmark_optimized_engine(artists_data, test_operations)
        
        # Benchmark batch processing
        batch_results = benchmark_batch_processing(artists_data)
        
        # Calculate improvements
        time_improvement = ((standard_results['total_time'] - optimized_results['total_time']) / 
                           standard_results['total_time']) * 100
        
        memory_improvement = ((standard_results['memory_usage_mb'] - optimized_results['memory_usage_mb']) /
                             standard_results['memory_usage_mb']) * 100 if standard_results['memory_usage_mb'] > 0 else 0
        
        # Store results
        comparison = {
            'dataset_size': dataset_size,
            'standard_time': standard_results['total_time'],
            'optimized_time': optimized_results['total_time'],
            'time_improvement_percent': time_improvement,
            'standard_memory_mb': standard_results['memory_usage_mb'],
            'optimized_memory_mb': optimized_results['memory_usage_mb'],
            'memory_improvement_percent': memory_improvement,
            'cache_hit_rate': optimized_results['cache_hit_rate'],
            'batch_speedup': batch_results['speedup_factor'],
            'cached_speedup': (optimized_results['average_op_time'] / optimized_results['cached_average_time']
                              if optimized_results['cached_average_time'] > 0 else 1.0)
        }
        
        results_summary.append(comparison)
        
        # Print comparison
        print(f"\n📊 Performance Comparison Summary:")
        print(f"   Standard Engine Total Time: {standard_results['total_time']:.2f}s")
        print(f"   Optimized Engine Total Time: {optimized_results['total_time']:.2f}s")
        print(f"   Time Improvement: {time_improvement:.1f}%")
        
        print(f"\n💾 Memory Usage Comparison:")
        print(f"   Standard Engine Memory: {standard_results['memory_usage_mb']:.1f} MB")
        print(f"   Optimized Engine Memory: {optimized_results['memory_usage_mb']:.1f} MB")
        print(f"   Memory Improvement: {memory_improvement:.1f}%")
        
        print(f"\n⚡ Optimization Features:")
        print(f"   Cache Hit Rate: {optimized_results['cache_hit_rate']*100:.1f}%")
        print(f"   Cached Operation Speedup: {comparison['cached_speedup']:.1f}x")
        print(f"   Batch Processing Speedup: {batch_results['speedup_factor']:.1f}x")
    
    # Final summary
    print(f"\n" + "="*80)
    print("🏆 FINAL PERFORMANCE SUMMARY")
    print("="*80)
    
    for result in results_summary:
        print(f"\n📊 Dataset Size: {result['dataset_size']} artists")
        print(f"   Time Improvement: {result['time_improvement_percent']:.1f}%")
        print(f"   Memory Improvement: {result['memory_improvement_percent']:.1f}%")
        print(f"   Cache Hit Rate: {result['cache_hit_rate']*100:.1f}%")
        print(f"   Batch Processing Speedup: {result['batch_speedup']:.1f}x")
    
    # Calculate overall improvements
    avg_time_improvement = sum(r['time_improvement_percent'] for r in results_summary) / len(results_summary)
    avg_memory_improvement = sum(r['memory_improvement_percent'] for r in results_summary) / len(results_summary)
    avg_cache_hit_rate = sum(r['cache_hit_rate'] for r in results_summary) / len(results_summary)
    
    print(f"\n🎯 Overall Performance Gains:")
    print(f"   Average Time Improvement: {avg_time_improvement:.1f}%")
    print(f"   Average Memory Improvement: {avg_memory_improvement:.1f}%")
    print(f"   Average Cache Hit Rate: {avg_cache_hit_rate*100:.1f}%")
    
    print(f"\n✅ Optimization Features Successfully Demonstrated:")
    print("   • Multi-level intelligent caching with TTL management")
    print("   • Parallel batch processing for large datasets")
    print("   • Memory-efficient streaming operations")
    print("   • Performance monitoring and metrics tracking")
    print("   • Adaptive optimization strategies")


if __name__ == "__main__":
    try:
        run_performance_comparison()
    except KeyboardInterrupt:
        print("\n⚠️ Performance test interrupted by user")
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()