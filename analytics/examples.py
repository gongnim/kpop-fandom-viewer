#!/usr/bin/env python3
"""
K-Pop Dashboard Analytics Package Examples
==========================================

Comprehensive usage examples demonstrating the analytics package capabilities
for growth rate calculation, statistical analysis, and data quality assessment.

Run this script to see the analytics package in action.

Author: Backend Development Team
Date: 2025-09-08
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kpop_dashboard.analytics import (
        GrowthRateCalculator,
        MetricDataPoint,
        CalculationMethod,
        GrowthPeriod,
        calculate_daily_growth_rate,
        calculate_compound_growth_rate,
        classify_growth_rate,
        get_severity_level,
        validate_platform,
        validate_metric_type,
        get_package_info
    )
    print("✅ Successfully imported analytics package")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

def generate_sample_data(base_value: int = 1000000, 
                        days: int = 30, 
                        platform: str = 'youtube',
                        metric_type: str = 'subscribers',
                        trend: str = 'growing') -> List[MetricDataPoint]:
    """Generate sample data for demonstration purposes."""
    data_points = []
    current_value = base_value
    
    for i in range(days):
        timestamp = datetime.now() - timedelta(days=days-i-1)
        
        # Add trend-based variation
        if trend == 'growing':
            growth_factor = random.uniform(1.001, 1.015)  # 0.1% to 1.5% daily growth
        elif trend == 'declining':
            growth_factor = random.uniform(0.985, 0.999)  # -1.5% to -0.1% daily decline
        elif trend == 'volatile':
            growth_factor = random.uniform(0.95, 1.05)    # High volatility
        else:  # stable
            growth_factor = random.uniform(0.998, 1.002)  # Very stable
        
        current_value = int(current_value * growth_factor)
        
        # Add some noise and occasional outliers
        if random.random() < 0.05:  # 5% chance of outlier
            noise_factor = random.uniform(0.8, 1.3)
            current_value = int(current_value * noise_factor)
        
        quality_score = random.uniform(0.85, 1.0)  # Realistic quality scores
        
        data_points.append(MetricDataPoint(
            value=max(0, current_value),  # Ensure non-negative
            timestamp=timestamp,
            platform=platform,
            metric_type=metric_type,
            quality_score=quality_score
        ))
    
    return data_points

def example_basic_usage():
    """Demonstrate basic usage of the analytics package."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create calculator with default settings
    calculator = GrowthRateCalculator()
    
    # Generate sample data
    print("📊 Generating sample data (30 days, growing trend)...")
    data_points = generate_sample_data(
        base_value=1500000,
        days=30,
        platform='youtube',
        metric_type='subscribers',
        trend='growing'
    )
    
    print(f"   Generated {len(data_points)} data points")
    print(f"   Value range: {min(p.value for p in data_points):,} to {max(p.value for p in data_points):,}")
    
    # Calculate daily growth rate
    print("\n🔢 Calculating daily growth rate...")
    result = calculator.calculate_growth_rate(
        data_points=data_points,
        method=CalculationMethod.ROLLING_AVERAGE,
        period=GrowthPeriod.DAILY
    )
    
    if result:
        print(f"✅ Growth Rate: {result.growth_rate:.2f}%")
        print(f"   Data Quality: {result.data_quality_score:.3f}")
        print(f"   Statistically Significant: {result.is_significant}")
        print(f"   Sample Size: {result.sample_size}")
        print(f"   Outliers Detected: {result.outlier_detected}")
        
        if result.confidence_interval:
            lower, upper = result.confidence_interval
            print(f"   95% Confidence Interval: [{lower:.2f}%, {upper:.2f}%]")
        
        # Classify the growth rate
        category = classify_growth_rate(result.growth_rate)
        print(f"   Growth Category: {category}")
        
    else:
        print("❌ Failed to calculate growth rate")

def example_multiple_methods():
    """Demonstrate different calculation methods."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Calculation Methods")
    print("="*60)
    
    calculator = GrowthRateCalculator()
    
    # Generate volatile data
    print("📊 Generating volatile sample data...")
    data_points = generate_sample_data(
        base_value=800000,
        days=45,
        platform='spotify',
        metric_type='monthly_listeners',
        trend='volatile'
    )
    
    methods = [
        CalculationMethod.SIMPLE,
        CalculationMethod.ROLLING_AVERAGE,
        CalculationMethod.WEIGHTED,
        CalculationMethod.EXPONENTIAL_SMOOTHING
    ]
    
    print("\n🔍 Comparing calculation methods for weekly growth:")
    for method in methods:
        result = calculator.calculate_growth_rate(
            data_points=data_points,
            method=method,
            period=GrowthPeriod.WEEKLY
        )
        
        if result:
            print(f"   {method.value:20}: {result.growth_rate:8.2f}% (quality: {result.data_quality_score:.3f})")
        else:
            print(f"   {method.value:20}: Failed")

def example_multiple_periods():
    """Demonstrate multiple time period analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Time Periods Analysis")
    print("="*60)
    
    calculator = GrowthRateCalculator()
    
    # Generate long-term data
    print("📊 Generating long-term sample data (6 months)...")
    data_points = generate_sample_data(
        base_value=2000000,
        days=180,
        platform='instagram',
        metric_type='followers',
        trend='growing'
    )
    
    print("\n📈 Growth rates across different periods:")
    
    # Calculate for all periods
    results = calculator.calculate_multiple_periods(
        data_points=data_points,
        method=CalculationMethod.ROLLING_AVERAGE
    )
    
    for period_name, result in results.items():
        if result:
            category = classify_growth_rate(result.growth_rate)
            significance = "✓" if result.is_significant else "✗"
            print(f"   {period_name:15}: {result.growth_rate:8.2f}% ({category}, sig: {significance})")
        else:
            print(f"   {period_name:15}: No data")

def example_data_quality_analysis():
    """Demonstrate data quality analysis features."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Data Quality Analysis")
    print("="*60)
    
    # Test with different quality scenarios
    scenarios = [
        ('High Quality', 'stable', 0.95, 30),
        ('Medium Quality', 'volatile', 0.75, 20), 
        ('Low Quality', 'volatile', 0.45, 10),
    ]
    
    for scenario_name, trend, base_quality, days in scenarios:
        print(f"\n🔍 Testing {scenario_name} Data:")
        
        # Generate data with specific quality characteristics
        data_points = generate_sample_data(
            base_value=1200000,
            days=days,
            platform='tiktok',
            metric_type='followers',
            trend=trend
        )
        
        # Adjust quality scores
        for point in data_points:
            point.quality_score = base_quality + random.uniform(-0.1, 0.1)
            point.quality_score = max(0.1, min(1.0, point.quality_score))
        
        # Calculate with high precision settings
        precise_calculator = GrowthRateCalculator(
            min_data_points=5,
            outlier_threshold=2.0,
            confidence_level=0.95
        )
        
        result = precise_calculator.calculate_growth_rate(
            data_points=data_points,
            method=CalculationMethod.ROLLING_AVERAGE,
            period=GrowthPeriod.DAILY
        )
        
        if result:
            print(f"   Growth Rate: {result.growth_rate:.2f}%")
            print(f"   Data Quality Score: {result.data_quality_score:.3f}")
            print(f"   Sample Size: {result.sample_size}")
            print(f"   Outliers Detected: {result.outlier_detected}")
            
            # Quality interpretation
            if result.data_quality_score >= 0.9:
                quality_desc = "Excellent"
            elif result.data_quality_score >= 0.7:
                quality_desc = "Good"
            elif result.data_quality_score >= 0.5:
                quality_desc = "Fair (use with caution)"
            else:
                quality_desc = "Poor (unreliable)"
                
            print(f"   Quality Assessment: {quality_desc}")

def example_alert_generation():
    """Demonstrate alert generation based on growth rates."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Alert Generation")
    print("="*60)
    
    calculator = GrowthRateCalculator()
    
    # Test different growth scenarios for alert generation
    scenarios = [
        ('Explosive Growth', 'youtube', 'subscribers', 150.0),
        ('Rapid Growth', 'spotify', 'monthly_listeners', 75.0),
        ('Significant Decline', 'twitter', 'followers', -35.0),
        ('Normal Growth', 'instagram', 'followers', 5.0),
    ]
    
    for scenario_name, platform, metric_type, target_growth in scenarios:
        print(f"\n🚨 {scenario_name} Scenario:")
        print(f"   Platform: {platform}, Metric: {metric_type}")
        
        # Generate appropriate data to achieve target growth
        base_value = 1000000
        current_value = int(base_value * (1 + target_growth/100))
        
        data_points = [
            MetricDataPoint(
                value=current_value,
                timestamp=datetime.now(),
                platform=platform,
                metric_type=metric_type,
                quality_score=0.95
            ),
            MetricDataPoint(
                value=base_value,
                timestamp=datetime.now() - timedelta(days=1),
                platform=platform,
                metric_type=metric_type,
                quality_score=0.93
            )
        ]
        
        result = calculator.calculate_growth_rate(
            data_points=data_points,
            method=CalculationMethod.SIMPLE,
            period=GrowthPeriod.DAILY
        )
        
        if result:
            growth_rate = result.growth_rate
            is_decline = growth_rate < 0
            
            # Generate alert information
            category = classify_growth_rate(growth_rate)
            severity = get_severity_level(growth_rate, is_decline)
            
            print(f"   Growth Rate: {growth_rate:.2f}%")
            print(f"   Category: {category}")
            print(f"   Severity Level: {severity}")
            print(f"   Alert Required: {'Yes' if abs(growth_rate) > 20 else 'No'}")
            
            # Sample alert message
            if abs(growth_rate) > 50:
                alert_type = "Critical Alert"
                emoji = "🔥" if growth_rate > 0 else "🚨"
            elif abs(growth_rate) > 20:
                alert_type = "Warning Alert"
                emoji = "⚡" if growth_rate > 0 else "⚠️"
            else:
                alert_type = "Info Alert"
                emoji = "📈" if growth_rate > 0 else "📉"
                
            print(f"   Alert Type: {alert_type}")
            print(f"   Message: {emoji} {platform.title()} {metric_type} "
                  f"{'증가' if growth_rate > 0 else '감소'}: {abs(growth_rate):.1f}%")

def example_standalone_functions():
    """Demonstrate standalone utility functions."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Standalone Functions")
    print("="*60)
    
    # Simple calculations
    print("🔢 Simple Growth Calculations:")
    daily_growth = calculate_daily_growth_rate(1500000, 1450000)
    print(f"   Daily Growth Rate: {daily_growth:.2f}%")
    
    # Compound growth rate (CAGR)
    cagr = calculate_compound_growth_rate(
        initial_value=1000000,
        final_value=2500000, 
        periods=365  # 1 year
    )
    print(f"   Compound Annual Growth Rate: {cagr:.2f}%")
    
    # Validation functions
    print("\n✅ Validation Functions:")
    platforms = ['youtube', 'spotify', 'invalid_platform']
    metrics = ['subscribers', 'followers', 'invalid_metric']
    
    for platform in platforms:
        valid = validate_platform(platform)
        print(f"   Platform '{platform}': {'Valid' if valid else 'Invalid'}")
    
    for metric in metrics:
        valid = validate_metric_type(metric)
        print(f"   Metric '{metric}': {'Valid' if valid else 'Invalid'}")

def example_package_info():
    """Display package information and capabilities."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Package Information")
    print("="*60)
    
    info = get_package_info()
    
    print("📦 Package Information:")
    print(f"   Name: {info['name']}")
    print(f"   Version: {info['version']}")
    print(f"   Author: {info['author']}")
    print(f"   Description: {info['description']}")
    
    print("\n🔧 Capabilities:")
    print(f"   Growth Calculator Available: {info['growth_calculator_available']}")
    print(f"   Supported Platforms: {len(info['supported_platforms'])}")
    print(f"   Supported Metrics: {len(info['supported_metrics'])}")
    print(f"   Calculation Methods: {len(info['calculation_methods'])}")
    
    print("\n🌐 Supported Platforms:")
    for platform in info['supported_platforms']:
        print(f"   - {platform}")
    
    print("\n📊 Supported Metrics:")
    for metric in info['supported_metrics']:
        print(f"   - {metric}")
    
    print("\n⚙️ Calculation Methods:")
    for method in info['calculation_methods']:
        print(f"   - {method}")

def main():
    """Run all examples."""
    print("🎵 K-Pop Dashboard Analytics Package Examples")
    print("=" * 80)
    print("Backend Development Team - 2025-09-08")
    
    try:
        # Run all examples
        example_basic_usage()
        example_multiple_methods()
        example_multiple_periods() 
        example_data_quality_analysis()
        example_alert_generation()
        example_standalone_functions()
        example_package_info()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        
        print("\n📚 Next Steps:")
        print("   1. Review the analytics/README.md for detailed documentation")
        print("   2. Integrate with your database using database_postgresql.py")
        print("   3. Use the analytics package in your Streamlit dashboard")
        print("   4. Set up automated growth rate calculations")
        print("   5. Implement alert generation based on growth thresholds")
        
    except Exception as e:
        print(f"\n❌ Error during examples execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()