#!/usr/bin/env python3
"""
MobileOptimizer Usage Examples
=============================

This file demonstrates how to use the MobileOptimizer class for K-POP Dashboard
mobile optimization scenarios. It includes practical examples for real-world usage.

Author: Backend Development Team
Date: 2025-09-09
"""

import sys
import os
from datetime import datetime

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kpi.mobile_optimizer import (
        MobileOptimizer,
        MobileOptimizationConfig,
        MobileDeviceType,
        NotificationType,
        CacheLevel,
        create_mobile_optimizer,
        generate_cache_key
    )
    print("✅ Successfully imported MobileOptimizer from project")
except ImportError as e:
    print(f"⚠️  Could not import from project: {e}")
    print("💡 Run this example from the kpop_dashboard directory")
    sys.exit(1)


def example_1_basic_optimization():
    """Example 1: Basic data optimization for mobile."""
    print("\n🔥 Example 1: Basic Mobile Data Optimization")
    print("-" * 50)
    
    # Create optimizer with default settings
    optimizer = MobileOptimizer()
    
    # Sample K-POP artist data
    artist_data = {
        'id': 1,
        'name': 'NewJeans',
        'group_type': 'girl_group',
        'debut_date': '2022-07-22',
        'company': 'ADOR',
        'members': [
            {'name': 'Minji', 'position': 'Leader, Vocalist'},
            {'name': 'Hanni', 'position': 'Vocalist'},
            {'name': 'Danielle', 'position': 'Vocalist'},
            {'name': 'Haerin', 'position': 'Vocalist'},
            {'name': 'Hyein', 'position': 'Vocalist, Rapper'}
        ],
        'discography': [
            {'title': 'New Jeans', 'release_date': '2022-08-01', 'type': 'EP'},
            {'title': 'OMG', 'release_date': '2023-01-02', 'type': 'Single'},
            {'title': 'Get Up', 'release_date': '2023-07-21', 'type': 'EP'}
        ],
        'metrics': {
            'youtube_subscribers': 8500000,
            'spotify_monthly_listeners': 25000000,
            'instagram_followers': 12000000,
            'twitter_followers': 3200000,
            'tiktok_followers': 15000000
        },
        'description': 'NewJeans is a South Korean girl group formed by ADOR. ' * 20,  # Long description
        'biography': 'Detailed biography content...' * 50,  # Very long bio
        'raw_analytics_data': {'complex': 'data'} * 100  # Heavy data
    }
    
    # Optimize for smartphone
    result = optimizer.optimize_for_mobile(artist_data, MobileDeviceType.SMARTPHONE)
    
    print(f"📊 Optimization Results:")
    print(f"   Original Size: {result.original_size:,} bytes")
    print(f"   Optimized Size: {result.optimized_size:,} bytes")
    print(f"   Compression Ratio: {result.compression_ratio:.1%}")
    print(f"   Processing Time: {result.processing_time_ms:.2f}ms")
    print(f"   Device Compatibility: {[d.value for d in result.device_compatibility]}")
    print(f"   Optimizations Applied: {result.optimization_applied}")


def example_2_device_specific_optimization():
    """Example 2: Device-specific optimization strategies."""
    print("\n📱 Example 2: Device-Specific Optimization")
    print("-" * 50)
    
    # Large dataset with multiple artists
    large_dataset = {
        'artists': [
            {
                'id': i,
                'name': f'K-POP Artist {i}',
                'followers': i * 10000,
                'recent_songs': [f'Song {j}' for j in range(20)],  # 20 songs each
                'detailed_bio': f'Very detailed biography for artist {i}...' * 30
            } 
            for i in range(1, 101)  # 100 artists
        ],
        'metadata': {
            'total_count': 100,
            'last_updated': datetime.now().isoformat(),
            'data_source': 'K-POP Analytics API'
        }
    }
    
    devices = [
        (MobileDeviceType.SMARTWATCH, "⌚ Smartwatch"),
        (MobileDeviceType.SMARTPHONE, "📱 Smartphone"), 
        (MobileDeviceType.TABLET, "📟 Tablet")
    ]
    
    optimizer = MobileOptimizer()
    
    for device_type, device_name in devices:
        result = optimizer.optimize_for_mobile(large_dataset, device_type)
        
        print(f"{device_name}:")
        print(f"   Optimized Size: {result.optimized_size:,} bytes")
        print(f"   Compression: {result.compression_ratio:.1%}")
        print(f"   Compatible Devices: {len(result.device_compatibility)}")


def example_3_push_notifications():
    """Example 3: Push notification generation for different scenarios."""
    print("\n🔔 Example 3: Push Notification Generation")
    print("-" * 50)
    
    optimizer = MobileOptimizer()
    
    # Scenario 1: Milestone Achievement
    milestone_notification = optimizer.generate_push_notification(
        title="🎉 BLACKPINK Milestone!",
        body="BLACKPINK just reached 100 million YouTube subscribers!",
        notification_type=NotificationType.MILESTONE_ACHIEVEMENT,
        target_user_id="fan_12345",
        additional_data={
            'artist_id': 'blackpink',
            'milestone_type': 'youtube_subscribers',
            'milestone_value': 100000000,
            'achievement_date': datetime.now().isoformat()
        }
    )
    
    print(f"🏆 Milestone Notification:")
    print(f"   Title: {milestone_notification.title}")
    print(f"   Body: {milestone_notification.body}")
    print(f"   Priority: {milestone_notification.priority}")
    print(f"   Expires: {milestone_notification.expires_at.strftime('%Y-%m-%d %H:%M')}")
    
    # Scenario 2: Trending Content
    trending_notification = optimizer.generate_push_notification(
        title="📈 Trending Now",
        body="IVE's new music video is #1 trending on YouTube Korea!",
        notification_type=NotificationType.TRENDING_CONTENT,
        additional_data={
            'artist_id': 'ive',
            'content_type': 'music_video',
            'trending_position': 1,
            'platform': 'youtube_korea'
        }
    )
    
    print(f"📈 Trending Notification:")
    print(f"   Title: {trending_notification.title}")
    print(f"   Priority: {trending_notification.priority}")
    print(f"   Sound: {trending_notification.sound}")
    
    # Scenario 3: Performance Alert
    alert_notification = optimizer.generate_push_notification(
        title="⚡ Performance Alert",
        body="ITZY's popularity score increased by 25% this week!",
        notification_type=NotificationType.PERFORMANCE_ALERT,
        target_user_id="analyst_789",
        additional_data={
            'artist_id': 'itzy',
            'metric_type': 'popularity_score',
            'change_percentage': 25.4,
            'time_period': 'week'
        }
    )
    
    print(f"⚡ Performance Alert:")
    print(f"   Title: {alert_notification.title}")
    print(f"   Priority: {alert_notification.priority}")
    print(f"   Target User: {alert_notification.target_user_id}")
    
    print(f"\n📨 Total notifications in queue: {len(optimizer.notification_queue)}")


def example_4_offline_caching():
    """Example 4: Offline caching with different cache levels."""
    print("\n💾 Example 4: Offline Caching Strategies")
    print("-" * 50)
    
    # Artist profile data
    artist_profile = {
        'id': 'aespa_001',
        'name': 'aespa',
        'essential_info': {
            'debut_date': '2020-11-17',
            'company': 'SM Entertainment',
            'member_count': 4
        },
        'detailed_metrics': {
            'youtube_subscribers': 7800000,
            'spotify_monthly_listeners': 22000000,
            'instagram_followers': 9500000,
            'weekly_chart_position': 3,
            'monthly_growth_rate': 12.5
        },
        'full_biography': 'aespa is a South Korean girl group...' * 100,  # Long bio
        'detailed_discography': [
            {'title': f'Song {i}', 'stats': {'plays': i * 1000000}} 
            for i in range(50)  # 50 songs with detailed stats
        ],
        'raw_data': {'analytics': 'complex data'} * 200  # Heavy raw data
    }
    
    cache_levels = [
        (CacheLevel.MINIMAL, "⚡ Minimal", "Essential data only"),
        (CacheLevel.STANDARD, "📄 Standard", "Regular mobile experience"),
        (CacheLevel.ENHANCED, "🔥 Enhanced", "Rich content for high-end devices")
    ]
    
    for cache_level, level_name, description in cache_levels:
        # Create optimizer with specific cache level
        config = MobileOptimizationConfig(cache_level=cache_level)
        optimizer = MobileOptimizer(config)
        
        # Cache the artist profile
        cache_key = generate_cache_key("artist", "aespa", cache_level.value)
        success = optimizer.cache_for_offline(cache_key, artist_profile, cache_level)
        
        if success:
            # Retrieve cached data to see what was stored
            cached_data = optimizer.get_cache_entry(cache_key)
            cache_entry = optimizer.cache_storage[cache_key]
            
            print(f"{level_name} ({description}):")
            print(f"   Cache Size: {cache_entry.compressed_size:,} bytes")
            print(f"   Fields Cached: {len(cached_data) if cached_data else 0}")
            print(f"   Expires: {cache_entry.expires_at.strftime('%Y-%m-%d %H:%M')}")


def example_5_performance_monitoring():
    """Example 5: Performance monitoring and system statistics."""
    print("\n📊 Example 5: Performance Monitoring")
    print("-" * 50)
    
    optimizer = MobileOptimizer()
    
    # Simulate mobile app operations
    print("🔄 Simulating mobile operations...")
    
    # Multiple optimization operations
    for i in range(10):
        test_data = {
            'artist_id': f'artist_{i}',
            'metrics': {'followers': i * 100000, 'views': i * 1000000},
            'content': [f'Item {j}' for j in range(i * 10)]  # Variable size content
        }
        
        # Optimize for different devices
        devices = [MobileDeviceType.SMARTPHONE, MobileDeviceType.TABLET, MobileDeviceType.SMARTWATCH]
        device = devices[i % 3]
        
        optimizer.optimize_for_mobile(test_data, device)
        
        # Cache some data
        if i % 2 == 0:
            cache_key = generate_cache_key("test_data", i)
            optimizer.cache_for_offline(cache_key, test_data)
        
        # Generate some notifications
        if i % 3 == 0:
            optimizer.generate_push_notification(
                f"Update {i}",
                f"Test notification number {i}",
                NotificationType.SYSTEM_UPDATE
            )
    
    # Get performance statistics
    stats = optimizer.get_performance_stats()
    
    print("📈 Performance Statistics:")
    print(f"   Total Optimizations: {stats['compression_stats']['total_optimizations']}")
    print(f"   Average Processing Time: {stats['avg_optimization_time_ms']:.2f}ms")
    print(f"   Total Bytes Saved: {stats['compression_stats']['total_bytes_saved']:,}")
    print(f"   Cache Entries: {stats['cache_entries']}")
    print(f"   Notification Queue Size: {stats['notification_queue_size']}")
    print(f"   Current Configuration:")
    print(f"     Device Type: {stats['config']['device_type']}")
    print(f"     Cache Level: {stats['config']['cache_level']}")
    print(f"     Compression Enabled: {stats['config']['compression_enabled']}")


def example_6_factory_functions():
    """Example 6: Using factory functions for common configurations."""
    print("\n🏭 Example 6: Factory Functions & Utilities")
    print("-" * 50)
    
    # Create optimizers for different scenarios
    scenarios = [
        ("smartphone", "standard", "📱 Regular smartphone user"),
        ("tablet", "enhanced", "📟 Tablet with high-speed connection"),
        ("smartwatch", "minimal", "⌚ Smartwatch with limited resources"),
        ("smartphone", "minimal", "📱 Smartphone with limited data plan")
    ]
    
    test_data = {
        'artists': [{'id': i, 'name': f'Artist {i}'} for i in range(20)],
        'metadata': {'total': 20, 'page': 1}
    }
    
    for device_type, cache_level, description in scenarios:
        # Use factory function
        optimizer = create_mobile_optimizer(device_type, cache_level)
        
        # Test optimization
        result = optimizer.optimize_for_mobile(test_data)
        
        print(f"{description}:")
        print(f"   Optimized Size: {result.optimized_size:,} bytes")
        print(f"   Compression: {result.compression_ratio:.1%}")
        print(f"   Device Compatibility: {len(result.device_compatibility)} devices")
    
    # Demonstrate cache key generation
    print("\n🔑 Cache Key Generation:")
    keys = [
        generate_cache_key("artist", "blackpink", "profile"),
        generate_cache_key("group", "bts", "metrics", "2025-09"),
        generate_cache_key("trending", "korea", datetime.now().strftime("%Y%m%d"))
    ]
    
    for i, key in enumerate(keys, 1):
        print(f"   Key {i}: {key}")


def main():
    """Run all examples."""
    print("🎵 K-POP Dashboard MobileOptimizer Examples")
    print("=" * 60)
    print("📝 Demonstrating mobile optimization capabilities for K-POP data")
    
    examples = [
        example_1_basic_optimization,
        example_2_device_specific_optimization,
        example_3_push_notifications,
        example_4_offline_caching,
        example_5_performance_monitoring,
        example_6_factory_functions
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Error in {example_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("✨ MobileOptimizer examples completed!")
    print("💡 These examples show how to integrate MobileOptimizer into your K-POP Dashboard")


if __name__ == "__main__":
    main()