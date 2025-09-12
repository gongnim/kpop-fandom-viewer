#!/usr/bin/env python3
"""
K-Pop Ranking System Examples
=============================

Comprehensive examples demonstrating the GrowthRankingEngine capabilities
for artist performance analysis, comparative rankings, and multi-dimensional scoring.

Run this script to see the ranking system in action with realistic K-Pop data scenarios.

Author: Backend Development Team
Date: 2025-09-08
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kpop_dashboard.analytics import (
        GrowthRankingEngine,
        RankingCategory,
        RankingPeriod,
        DebutCohort,
        ArtistMetrics,
        MetricDataPoint,
        RankingResult,
        CompositeIndex
    )
    print("✅ Successfully imported ranking system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def generate_sample_artist_data() -> List[ArtistMetrics]:
    """Generate realistic sample data for multiple artists."""
    artists_data = []
    base_timestamp = datetime.now()
    
    # Sample K-Pop artists with realistic data patterns
    sample_artists = [
        # 4th Generation (Recent debuts, high growth)
        {"id": 1, "name": "NEWJEANS", "company_id": 1, "company": "ADOR", "debut": 2022, "base_subs": 8000000, "growth_pattern": "explosive"},
        {"id": 2, "name": "IVE", "company_id": 2, "company": "Starship", "debut": 2021, "base_subs": 6500000, "growth_pattern": "rapid"},
        {"id": 3, "name": "LESSERAFIM", "company_id": 3, "company": "Source Music", "debut": 2022, "base_subs": 5200000, "growth_pattern": "rapid"},
        
        # 3rd Generation (Established, steady growth)
        {"id": 4, "name": "BLACKPINK", "company_id": 4, "company": "YG", "debut": 2016, "base_subs": 89000000, "growth_pattern": "steady"},
        {"id": 5, "name": "TWICE", "company_id": 5, "company": "JYP", "debut": 2015, "base_subs": 15000000, "growth_pattern": "steady"},
        {"id": 6, "name": "RED VELVET", "company_id": 6, "company": "SM", "debut": 2014, "base_subs": 7800000, "growth_pattern": "stable"},
        
        # 2nd Generation (Mature artists, slower growth)
        {"id": 7, "name": "SNSD", "company_id": 6, "company": "SM", "debut": 2007, "base_subs": 4500000, "growth_pattern": "mature"},
        {"id": 8, "name": "BIGBANG", "company_id": 4, "company": "YG", "debut": 2006, "base_subs": 3200000, "growth_pattern": "mature"},
    ]
    
    for artist_info in sample_artists:
        # Generate YouTube subscriber data
        youtube_data_points = generate_metric_timeline(
            base_value=artist_info["base_subs"],
            days=90,  # 3 months of data
            pattern=artist_info["growth_pattern"],
            base_timestamp=base_timestamp
        )
        
        youtube_metrics = ArtistMetrics(
            artist_id=artist_info["id"],
            artist_name=artist_info["name"],
            company_id=artist_info["company_id"],
            company_name=artist_info["company"],
            debut_year=artist_info["debut"],
            platform="youtube",
            metric_type="subscribers",
            current_value=youtube_data_points[0].value,
            data_points=youtube_data_points,
            quality_score=random.uniform(0.85, 0.98)
        )
        artists_data.append(youtube_metrics)
        
        # Generate Spotify monthly listeners data (scaled appropriately)
        spotify_base = int(artist_info["base_subs"] * random.uniform(0.3, 0.8))  # Generally lower than YouTube
        spotify_data_points = generate_metric_timeline(
            base_value=spotify_base,
            days=90,
            pattern=artist_info["growth_pattern"],
            base_timestamp=base_timestamp
        )
        
        spotify_metrics = ArtistMetrics(
            artist_id=artist_info["id"],
            artist_name=artist_info["name"],
            company_id=artist_info["company_id"],
            company_name=artist_info["company"],
            debut_year=artist_info["debut"],
            platform="spotify",
            metric_type="monthly_listeners",
            current_value=spotify_data_points[0].value,
            data_points=spotify_data_points,
            quality_score=random.uniform(0.80, 0.95)
        )
        artists_data.append(spotify_metrics)
    
    return artists_data


def generate_metric_timeline(base_value: int, days: int, pattern: str, base_timestamp: datetime) -> List[MetricDataPoint]:
    """Generate a realistic timeline of metric data points."""
    data_points = []
    current_value = base_value
    
    for i in range(days):
        timestamp = base_timestamp - timedelta(days=days-i-1)
        
        # Apply growth pattern
        if pattern == "explosive":
            daily_change = random.uniform(0.015, 0.035)  # 1.5-3.5% daily growth
        elif pattern == "rapid":
            daily_change = random.uniform(0.008, 0.020)  # 0.8-2.0% daily growth  
        elif pattern == "steady":
            daily_change = random.uniform(0.003, 0.012)  # 0.3-1.2% daily growth
        elif pattern == "stable":
            daily_change = random.uniform(0.001, 0.006)  # 0.1-0.6% daily growth
        elif pattern == "mature":
            daily_change = random.uniform(-0.002, 0.004)  # -0.2% to 0.4% daily change
        else:
            daily_change = random.uniform(0.001, 0.008)   # Default pattern
        
        # Add comeback boost simulation (random chance)
        if random.random() < 0.05:  # 5% chance of comeback boost
            daily_change *= random.uniform(2.0, 4.0)
        
        # Apply change
        current_value = int(current_value * (1 + daily_change))
        
        # Add some noise
        noise_factor = random.uniform(0.995, 1.005)
        current_value = int(current_value * noise_factor)
        
        quality_score = random.uniform(0.88, 0.98)
        
        data_points.append(MetricDataPoint(
            value=max(0, current_value),
            timestamp=timestamp,
            platform="youtube",  # Will be overridden by caller
            metric_type="subscribers",  # Will be overridden by caller
            quality_score=quality_score
        ))
    
    return data_points


def example_platform_ranking():
    """Demonstrate platform-specific artist ranking."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Platform-Specific Artist Ranking")
    print("="*60)
    
    # Generate sample data
    artists_data = generate_sample_artist_data()
    youtube_artists = [a for a in artists_data if a.platform == "youtube"]
    
    print(f"📊 Analyzing {len(youtube_artists)} artists on YouTube (subscribers)")
    
    # Create ranking engine
    engine = GrowthRankingEngine()
    
    # Rank by growth rate
    print("\n🚀 Ranking by Growth Rate (Monthly):")
    growth_rankings = engine.rank_by_platform(
        youtube_artists,
        platform="youtube",
        metric_type="subscribers",
        category=RankingCategory.GROWTH_RATE,
        period=RankingPeriod.MONTHLY
    )
    
    for i, ranking in enumerate(growth_rankings[:5]):  # Top 5
        print(f"   {ranking.rank}. {ranking.artist_name} ({ranking.company_name})")
        print(f"      Growth: {ranking.growth_rate:.2f}% | Score: {ranking.score:.1f} | Percentile: {ranking.percentile:.1f}%")
    
    # Rank by absolute value
    print("\n📈 Ranking by Absolute Subscribers:")
    absolute_rankings = engine.rank_by_platform(
        youtube_artists,
        platform="youtube", 
        metric_type="subscribers",
        category=RankingCategory.ABSOLUTE_VALUE
    )
    
    for ranking in absolute_rankings[:5]:  # Top 5
        print(f"   {ranking.rank}. {ranking.artist_name}: {ranking.current_value:,} subscribers")
        print(f"      Score: {ranking.score:.1f} | Percentile: {ranking.percentile:.1f}%")


def example_composite_index():
    """Demonstrate multi-dimensional composite index calculation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Dimensional Composite Index")
    print("="*60)
    
    artists_data = generate_sample_artist_data()
    
    print(f"📊 Calculating composite index for artists across YouTube & Spotify")
    
    # Create ranking engine with custom weights
    custom_weights = {
        'growth_rate': 0.40,      # Emphasize growth
        'absolute_value': 0.20,   # Current performance
        'consistency': 0.20,      # Stability
        'momentum': 0.15,         # Recent trends
        'engagement_rate': 0.05   # Quality metrics
    }
    
    engine = GrowthRankingEngine(default_weights=custom_weights)
    
    # Calculate composite index
    composite_rankings = engine.calculate_composite_index(
        artists_data,
        platforms=["youtube", "spotify"],
        weights=custom_weights,
        include_analysis=True
    )
    
    print(f"\n🏆 Overall Artist Rankings (Composite Index):")
    for ranking in composite_rankings:
        print(f"   {ranking.rank}. {ranking.artist_name}")
        print(f"      Score: {ranking.overall_score:.1f} | Percentile: {ranking.percentile:.1f}%")
        print(f"      Strengths: {', '.join(ranking.strengths) if ranking.strengths else 'None identified'}")
        print(f"      Areas for growth: {', '.join(ranking.weaknesses) if ranking.weaknesses else 'Well-balanced'}")
        print()


def example_company_ranking():
    """Demonstrate intra-company artist ranking."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Intra-Company Artist Ranking")
    print("="*60)
    
    artists_data = generate_sample_artist_data()
    
    # Focus on a specific company (SM Entertainment, company_id=6)
    print("🏢 Analyzing SM Entertainment Artists:")
    
    engine = GrowthRankingEngine()
    
    company_analysis = engine.rank_within_company(
        artists_data,
        company_id=6,
        platform="youtube",
        metric_type="subscribers",
        category=RankingCategory.COMPOSITE,
        include_company_stats=True
    )
    
    rankings = company_analysis['rankings']
    company_stats = company_analysis['company_stats']
    
    print(f"\n📊 {company_stats['company_name']} Company Statistics:")
    print(f"   Total Artists: {company_stats['total_artists']}")
    print(f"   Top Performer: {company_stats['top_performer']}")
    print(f"   Average Score: {company_stats['average_score']:.1f}")
    print(f"   Performance Spread: {company_stats['performance_spread']:.1f}")
    
    print(f"\n🏆 Internal Artist Rankings:")
    for ranking in rankings:
        print(f"   {ranking.rank}. {ranking.artist_name}")
        print(f"      Score: {ranking.score:.1f} | Growth: {ranking.growth_rate:.2f}%")
        if 'strengths' in ranking.calculation_details:
            print(f"      Strengths: {', '.join(ranking.calculation_details['strengths'])}")


def example_debut_cohort_ranking():
    """Demonstrate generational/cohort-based ranking."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Debut Cohort Analysis")
    print("="*60)
    
    artists_data = generate_sample_artist_data()
    
    # Analyze 4th generation artists
    print("🎭 Analyzing 4th Generation K-Pop Artists (2020-2029):")
    
    engine = GrowthRankingEngine()
    
    cohort_analysis = engine.rank_debut_cohort(
        artists_data,
        cohort=DebutCohort.FOURTH_GEN,
        platform="youtube",
        metric_type="subscribers",
        category=RankingCategory.GROWTH_RATE,
        include_cohort_analysis=True
    )
    
    rankings = cohort_analysis['rankings']
    cohort_info = cohort_analysis['cohort_analysis']
    
    print(f"\n📈 {cohort_info['cohort_name']} Cohort Overview:")
    print(f"   Total Artists: {cohort_info['total_artists']}")
    print(f"   Debut Years: {cohort_info['debut_year_range']}")
    print(f"   Average Years Active: {cohort_info['average_years_active']:.1f}")
    print(f"   Cohort Leader: {cohort_info['cohort_leader']}")
    print(f"   Maturity Trend: {cohort_info['maturity_trend']}")
    
    print(f"\n🏆 Cohort Rankings by Growth Rate:")
    for ranking in rankings:
        years_active = ranking.calculation_details['years_active']
        print(f"   {ranking.rank}. {ranking.artist_name} (Debut: {ranking.calculation_details['debut_year']})")
        print(f"      Growth: {ranking.growth_rate:.2f}% | {years_active} years active")
    
    # Also analyze 3rd generation for comparison
    print("\n" + "-"*40)
    print("🎭 3rd Generation Comparison (2010-2019):")
    
    third_gen_analysis = engine.rank_debut_cohort(
        artists_data,
        cohort=DebutCohort.THIRD_GEN,
        platform="youtube",
        metric_type="subscribers",
        category=RankingCategory.ABSOLUTE_VALUE,  # Use absolute value for mature artists
        include_cohort_analysis=True
    )
    
    third_gen_rankings = third_gen_analysis['rankings']
    third_gen_info = third_gen_analysis['cohort_analysis']
    
    print(f"   Cohort Leader: {third_gen_info['cohort_leader']}")
    print(f"   Average Years Active: {third_gen_info['average_years_active']:.1f}")
    print(f"   Maturity Trend: {third_gen_info['maturity_trend']}")


def example_comprehensive_analysis():
    """Demonstrate comprehensive multi-category analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Comprehensive Multi-Category Analysis")
    print("="*60)
    
    artists_data = generate_sample_artist_data()
    youtube_artists = [a for a in artists_data if a.platform == "youtube"]
    
    print(f"📊 Multi-Category Analysis for {len(youtube_artists)} YouTube Artists")
    
    engine = GrowthRankingEngine()
    
    categories = [
        (RankingCategory.GROWTH_RATE, "Growth Rate"),
        (RankingCategory.ABSOLUTE_VALUE, "Total Subscribers"),
        (RankingCategory.CONSISTENCY, "Performance Consistency"),
        (RankingCategory.MOMENTUM, "Recent Momentum")
    ]
    
    all_rankings = {}
    
    for category, category_name in categories:
        print(f"\n📈 Top 3 by {category_name}:")
        
        rankings = engine.rank_by_platform(
            youtube_artists,
            platform="youtube",
            metric_type="subscribers", 
            category=category
        )
        
        all_rankings[category_name] = rankings
        
        for ranking in rankings[:3]:
            print(f"   {ranking.rank}. {ranking.artist_name}")
            if category == RankingCategory.GROWTH_RATE:
                print(f"      Growth: {ranking.growth_rate:.2f}%")
            elif category == RankingCategory.ABSOLUTE_VALUE:
                print(f"      Subscribers: {ranking.current_value:,}")
            else:
                print(f"      Score: {ranking.score:.1f}")
    
    # Find artists that appear in multiple top rankings
    print(f"\n🌟 Multi-Category Top Performers:")
    top_performers = {}
    
    for category_name, rankings in all_rankings.items():
        for ranking in rankings[:3]:  # Top 3 in each category
            artist_name = ranking.artist_name
            if artist_name not in top_performers:
                top_performers[artist_name] = []
            top_performers[artist_name].append(category_name)
    
    multi_category_artists = {k: v for k, v in top_performers.items() if len(v) > 1}
    
    for artist, categories in multi_category_artists.items():
        print(f"   🏆 {artist}: Top performer in {', '.join(categories)}")


def main():
    """Run all ranking system examples."""
    print("🎵 K-Pop Growth Ranking System Examples")
    print("=" * 80)
    print("Backend Development Team - 2025-09-08")
    
    try:
        # Run all examples
        example_platform_ranking()
        example_composite_index()
        example_company_ranking()
        example_debut_cohort_ranking()
        example_comprehensive_analysis()
        
        print("\n" + "="*80)
        print("✅ All ranking examples completed successfully!")
        
        print("\n📚 Key Features Demonstrated:")
        print("   1. Platform-specific artist ranking by multiple criteria")
        print("   2. Multi-dimensional composite index with weighted scoring")
        print("   3. Intra-company competitive analysis")
        print("   4. Generational/cohort-based comparative rankings")
        print("   5. Multi-category performance analysis")
        
        print("\n🚀 Next Steps:")
        print("   1. Integrate with database for real-time artist data")
        print("   2. Add machine learning models for predictive rankings")
        print("   3. Implement real-time ranking updates")
        print("   4. Create ranking visualization components")
        print("   5. Add ranking history tracking and trend analysis")
        
    except Exception as e:
        print(f"\n❌ Error during examples execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()