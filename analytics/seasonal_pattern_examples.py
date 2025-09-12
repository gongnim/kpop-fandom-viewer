"""
Seasonal Pattern Analyzer Examples and Demonstrations
===================================================

Comprehensive examples and demonstrations of the SeasonalPatternAnalyzer
functionality for K-Pop artist metrics analysis.

This module provides:
- Basic usage examples
- Advanced analysis demonstrations
- Integration examples with existing analytics
- Visualization examples for seasonal patterns
- Performance benchmarking utilities

Author: Analytics Team
Version: 1.0.0
Date: 2025-09-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .seasonal_pattern_analyzer import (
        SeasonalPatternAnalyzer,
        SeasonalTrend,
        EventImpact,
        HolidayEffect,
        GlobalTrendCorrelation,
        SeasonalAnalysisResult,
        EventType,
        HolidayType,
        TrendCorrelationType
    )
except ImportError:
    # For standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from seasonal_pattern_analyzer import (
        SeasonalPatternAnalyzer,
        SeasonalTrend,
        EventImpact,
        HolidayEffect,
        GlobalTrendCorrelation,
        SeasonalAnalysisResult,
        EventType,
        HolidayType,
        TrendCorrelationType
    )


class SeasonalPatternDemo:
    """
    Demonstration class for SeasonalPatternAnalyzer functionality.
    
    This class provides comprehensive examples and demonstrations of:
    - Seasonal trend analysis
    - Event impact detection
    - Holiday effects analysis
    - Global trend correlation
    - Integrated seasonal analysis
    """
    
    def __init__(self):
        """Initialize the demonstration environment."""
        self.analyzer = SeasonalPatternAnalyzer()
        self.logger = logger
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info("SeasonalPatternDemo initialized")
    
    def generate_sample_data(
        self,
        artist_id: str = "artist_001",
        artist_name: str = "Sample Artist",
        days: int = 730,
        platforms: List[str] = None
    ) -> tuple:
        """
        Generate realistic sample data for demonstration.
        
        Args:
            artist_id: Artist identifier
            artist_name: Artist name
            days: Number of days of data to generate
            platforms: List of platforms to include
            
        Returns:
            Tuple of (metrics_data, events_data, external_trends)
        """
        if platforms is None:
            platforms = ['youtube', 'spotify', 'twitter']
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate base metrics with seasonal patterns
        metrics_data = pd.DataFrame(index=date_range)
        
        for platform in platforms:
            # Base growth with seasonal patterns
            base_trend = np.linspace(1000, 5000, len(date_range))
            
            # Add seasonal patterns
            seasonal_monthly = 500 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25 * 12)
            seasonal_yearly = 300 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25)
            
            # Add holiday effects
            holiday_effects = self._add_holiday_effects(date_range)
            
            # Add random noise
            noise = np.random.normal(0, 100, len(date_range))
            
            # Combine all components
            if platform == 'youtube':
                col_name = 'youtube_subscribers'
                multiplier = 1.0
            elif platform == 'spotify':
                col_name = 'spotify_followers'
                multiplier = 0.8
            else:
                col_name = f'{platform}_followers'
                multiplier = 0.6
            
            metrics_data[col_name] = (
                base_trend + seasonal_monthly + seasonal_yearly + 
                holiday_effects + noise
            ) * multiplier
            
            # Ensure no negative values
            metrics_data[col_name] = np.maximum(metrics_data[col_name], 100)
        
        # Generate sample events
        events_data = self._generate_sample_events(artist_id, date_range)
        
        # Generate external trends
        external_trends = self._generate_external_trends(date_range)
        
        self.logger.info(f"Generated sample data: {len(date_range)} days, {len(platforms)} platforms")
        return metrics_data, events_data, external_trends
    
    def demo_seasonal_trends_analysis(self) -> None:
        """Demonstrate seasonal trends analysis functionality."""
        print("\\n" + "="*60)
        print("SEASONAL TRENDS ANALYSIS DEMONSTRATION")
        print("="*60)
        
        # Generate sample data
        metrics_data, _, _ = self.generate_sample_data(days=730)
        
        # Analyze seasonal trends
        seasonal_trends = self.analyzer.analyze_seasonal_trends(
            artist_id="demo_artist",
            metrics_data=metrics_data,
            period_analysis="monthly"
        )
        
        print(f"\\nFound {len(seasonal_trends)} seasonal trends:")
        
        for i, trend in enumerate(seasonal_trends, 1):
            print(f"\\nTrend {i}:")
            print(f"  Component: {trend.component.value}")
            print(f"  Period: {trend.period}")
            print(f"  Strength: {trend.strength:.3f}")
            print(f"  Confidence: {trend.confidence:.3f}")
            print(f"  P-value: {trend.p_value:.4f}")
            print(f"  Trend Direction: {trend.trend_direction}")
            print(f"  Statistical Significance: {trend.statistical_significance}")
            print(f"  Peak Periods: {', '.join(trend.peak_periods[:3])}")
            print(f"  Trough Periods: {', '.join(trend.trough_periods[:3])}")
        
        # Visualize seasonal patterns
        self._plot_seasonal_trends(metrics_data, seasonal_trends)
        
        return seasonal_trends
    
    def demo_event_impact_analysis(self) -> None:
        """Demonstrate event impact detection functionality."""
        print("\\n" + "="*60)
        print("EVENT IMPACT ANALYSIS DEMONSTRATION")
        print("="*60)
        
        # Generate sample data
        metrics_data, events_data, _ = self.generate_sample_data(days=365)
        
        # Analyze event impacts
        event_impacts = self.analyzer.detect_event_impact(
            artist_id="demo_artist",
            metrics_data=metrics_data,
            events_data=events_data,
            impact_window_days=30,
            baseline_days=14
        )
        
        print(f"\\nDetected {len(event_impacts)} significant event impacts:")
        
        for i, impact in enumerate(event_impacts[:5], 1):  # Show top 5
            print(f"\\nEvent Impact {i}:")
            print(f"  Event Type: {impact.event_type.value}")
            print(f"  Event Date: {impact.event_date.strftime('%Y-%m-%d')}")
            print(f"  Pre-Event Baseline: {impact.pre_event_baseline:.1f}")
            print(f"  Post-Event Peak: {impact.post_event_peak:.1f}")
            print(f"  Impact Magnitude: {impact.impact_magnitude:.3f} ({impact.impact_magnitude*100:.1f}%)")
            print(f"  Impact Duration: {impact.impact_duration_days} days")
            print(f"  Recovery Time: {impact.recovery_days} days")
            print(f"  Statistical Significance (p-value): {impact.statistical_significance:.4f}")
            print(f"  Confidence Interval: ({impact.confidence_interval[0]:.3f}, {impact.confidence_interval[1]:.3f})")
            print(f"  Relative Impact Score: {impact.relative_impact_score:.3f}")
        
        # Visualize event impacts
        self._plot_event_impacts(metrics_data, events_data, event_impacts[:3])
        
        return event_impacts
    
    def demo_holiday_effects_analysis(self) -> None:
        """Demonstrate holiday effects analysis functionality."""
        print("\\n" + "="*60)
        print("HOLIDAY EFFECTS ANALYSIS DEMONSTRATION")
        print("="*60)
        
        # Generate sample data
        metrics_data, _, _ = self.generate_sample_data(days=1095)  # 3 years
        
        # Analyze holiday effects
        holiday_effects = self.analyzer.analyze_holiday_effects(
            artist_id="demo_artist",
            metrics_data=metrics_data,
            analysis_years=3,
            cultural_context="korean"
        )
        
        print(f"\\nAnalyzed {len(holiday_effects)} significant holiday effects:")
        
        for i, effect in enumerate(holiday_effects, 1):
            print(f"\\nHoliday Effect {i}:")
            print(f"  Holiday Type: {effect.holiday_type.value}")
            print(f"  Average Impact Magnitude: {effect.avg_impact_magnitude:.3f} ({effect.avg_impact_magnitude*100:.1f}%)")
            print(f"  Consistency Score: {effect.consistency_score:.3f}")
            print(f"  Lead Time: {effect.lead_time_days} days")
            print(f"  Recovery Time: {effect.recovery_time_days} days")
            print(f"  Cultural Significance: {effect.cultural_significance_score:.3f}")
            print(f"  Year-over-Year Consistency: {effect.year_over_year_consistency:.3f}")
            print(f"  Holiday Occurrences: {len(effect.holiday_dates)}")
            
            print("  Platform-Specific Effects:")
            for platform, platform_effect in effect.platform_specific_effects.items():
                print(f"    {platform}: {platform_effect:.3f} ({platform_effect*100:.1f}%)")
        
        # Visualize holiday effects
        self._plot_holiday_effects(metrics_data, holiday_effects[:3])
        
        return holiday_effects
    
    def demo_global_trend_correlation(self) -> None:
        """Demonstrate global trend correlation functionality."""
        print("\\n" + "="*60)
        print("GLOBAL TREND CORRELATION DEMONSTRATION")
        print("="*60)
        
        # Generate sample data
        metrics_data, _, external_trends = self.generate_sample_data(days=365)
        
        # Analyze global correlations
        correlations = self.analyzer.correlate_global_trends(
            artist_id="demo_artist",
            metrics_data=metrics_data,
            external_trends=external_trends,
            correlation_window=365,
            min_correlation=0.3
        )
        
        print(f"\\nFound {len(correlations)} significant correlations:")
        
        for i, corr in enumerate(correlations, 1):
            print(f"\\nCorrelation {i}:")
            print(f"  Trend Name: {corr.trend_name}")
            print(f"  Correlation Type: {corr.correlation_type.value}")
            print(f"  Correlation Coefficient: {corr.correlation_coefficient:.3f}")
            print(f"  P-value: {corr.p_value:.4f}")
            print(f"  Time Lag: {corr.time_lag_days} days")
            print(f"  Strength Category: {corr.strength_category}")
            print(f"  Confidence Level: {corr.confidence_level:.3f}")
            print(f"  Data Source: {corr.external_data_source}")
            print(f"  Analysis Window: {corr.correlation_window[0].strftime('%Y-%m-%d')} to {corr.correlation_window[1].strftime('%Y-%m-%d')}")
        
        # Visualize correlations
        self._plot_global_correlations(metrics_data, external_trends, correlations[:3])
        
        return correlations
    
    def demo_comprehensive_analysis(self) -> SeasonalAnalysisResult:
        """Demonstrate comprehensive seasonal pattern analysis."""
        print("\\n" + "="*80)
        print("COMPREHENSIVE SEASONAL PATTERN ANALYSIS DEMONSTRATION")
        print("="*80)
        
        # Generate comprehensive sample data
        metrics_data, events_data, external_trends = self.generate_sample_data(days=1095)  # 3 years
        
        # Perform comprehensive analysis
        analysis_result = self.analyzer.generate_comprehensive_analysis(
            artist_id="demo_artist",
            artist_name="Demo K-Pop Artist",
            metrics_data=metrics_data,
            events_data=events_data,
            external_trends=external_trends,
            analysis_config={
                'period_analysis': 'monthly',
                'impact_window_days': 30,
                'analysis_years': 3,
                'correlation_window': 365
            }
        )
        
        print(f"\\n📊 COMPREHENSIVE ANALYSIS RESULTS")
        print(f"   Artist: {analysis_result.artist_name} (ID: {analysis_result.artist_id})")
        print(f"   Analysis Period: {analysis_result.analysis_period[0].strftime('%Y-%m-%d')} to {analysis_result.analysis_period[1].strftime('%Y-%m-%d')}")
        print(f"   Platforms Analyzed: {', '.join(analysis_result.platform_metrics)}")
        print(f"   Analysis Timestamp: {analysis_result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Data Quality Score: {analysis_result.data_quality_score:.3f}")
        
        print(f"\\n🔍 KEY INSIGHTS:")
        print(f"   Overall Seasonality Score: {analysis_result.overall_seasonality_score:.3f}")
        print(f"   Predictability Index: {analysis_result.predictability_index:.3f}")
        print(f"   Seasonal Trends Detected: {len(analysis_result.seasonal_trends)}")
        print(f"   Significant Event Impacts: {len(analysis_result.event_impacts)}")
        print(f"   Holiday Effects Identified: {len(analysis_result.holiday_effects)}")
        print(f"   Global Trend Correlations: {len(analysis_result.global_correlations)}")
        
        # Summary statistics
        if analysis_result.seasonal_trends:
            avg_strength = np.mean([st.strength for st in analysis_result.seasonal_trends])
            print(f"   Average Seasonal Strength: {avg_strength:.3f}")
        
        if analysis_result.event_impacts:
            avg_impact = np.mean([ei.impact_magnitude for ei in analysis_result.event_impacts])
            print(f"   Average Event Impact: {avg_impact:.3f} ({avg_impact*100:.1f}%)")
        
        if analysis_result.holiday_effects:
            avg_holiday_impact = np.mean([he.avg_impact_magnitude for he in analysis_result.holiday_effects])
            print(f"   Average Holiday Impact: {avg_holiday_impact:.3f} ({avg_holiday_impact*100:.1f}%)")
        
        # Create comprehensive visualization
        self._create_comprehensive_dashboard(analysis_result, metrics_data)
        
        return analysis_result
    
    def benchmark_performance(self, data_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark analyzer performance with different data sizes."""
        if data_sizes is None:
            data_sizes = [365, 730, 1095, 1460]  # 1, 2, 3, 4 years
        
        print("\\n" + "="*60)
        print("PERFORMANCE BENCHMARKING")
        print("="*60)
        
        benchmark_results = {}
        
        for days in data_sizes:
            print(f"\\nBenchmarking with {days} days of data...")
            
            # Generate data
            start_time = datetime.now()
            metrics_data, events_data, external_trends = self.generate_sample_data(days=days)
            data_gen_time = (datetime.now() - start_time).total_seconds()
            
            # Benchmark comprehensive analysis
            start_time = datetime.now()
            result = self.analyzer.generate_comprehensive_analysis(
                artist_id=f"benchmark_{days}",
                artist_name="Benchmark Artist",
                metrics_data=metrics_data,
                events_data=events_data,
                external_trends=external_trends
            )
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            benchmark_results[days] = {
                'data_generation_time': data_gen_time,
                'analysis_time': analysis_time,
                'total_time': data_gen_time + analysis_time,
                'data_points': len(metrics_data),
                'seasonal_trends': len(result.seasonal_trends),
                'event_impacts': len(result.event_impacts),
                'holiday_effects': len(result.holiday_effects),
                'global_correlations': len(result.global_correlations),
                'data_quality_score': result.data_quality_score,
                'predictability_index': result.predictability_index
            }
            
            print(f"  Data Generation: {data_gen_time:.2f}s")
            print(f"  Analysis Time: {analysis_time:.2f}s")
            print(f"  Total Time: {data_gen_time + analysis_time:.2f}s")
            print(f"  Data Quality: {result.data_quality_score:.3f}")
            print(f"  Predictability: {result.predictability_index:.3f}")
        
        # Create performance visualization
        self._plot_performance_benchmark(benchmark_results)
        
        return benchmark_results
    
    def _add_holiday_effects(self, date_range: pd.DatetimeIndex) -> np.ndarray:
        """Add holiday effects to the data."""
        effects = np.zeros(len(date_range))
        
        for i, date in enumerate(date_range):
            # New Year effect
            if date.month == 1 and date.day <= 7:
                effects[i] += 200
            
            # Valentine's Day effect
            if date.month == 2 and 10 <= date.day <= 20:
                effects[i] += 150
            
            # Christmas effect
            if date.month == 12 and date.day >= 20:
                effects[i] += 300
            
            # Summer vacation effect (July-August)
            if date.month in [7, 8]:
                effects[i] += 100
            
            # Exam periods (April, November) - negative effect
            if date.month in [4, 11]:
                effects[i] -= 50
        
        return effects
    
    def _generate_sample_events(self, artist_id: str, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate sample events data."""
        events = []
        event_types = list(EventType)
        
        # Generate random events throughout the period
        num_events = max(5, len(date_range) // 60)  # Roughly one event per 2 months
        
        for i in range(num_events):
            event_date = np.random.choice(date_range)
            event_type = np.random.choice(event_types)
            
            events.append({
                'event_id': f'event_{i+1:03d}',
                'artist_id': artist_id,
                'event_type': event_type.value,
                'event_date': event_date,
                'event_name': f'{event_type.value.replace("_", " ").title()} Event {i+1}',
                'description': f'Sample {event_type.value} event for demonstration'
            })
        
        return pd.DataFrame(events)
    
    def _generate_external_trends(self, date_range: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Generate sample external trends data."""
        trends = {}
        
        # Economic trend
        economic_base = 100 + np.cumsum(np.random.normal(0, 2, len(date_range)))
        trends['economic_index'] = pd.Series(economic_base, index=date_range)
        
        # Social media trend
        social_trend = 50 + 30 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365) + np.random.normal(0, 5, len(date_range))
        trends['social_media_engagement'] = pd.Series(social_trend, index=date_range)
        
        # Music industry trend
        music_trend = 75 + 25 * np.cos(2 * np.pi * np.arange(len(date_range)) / 180) + np.random.normal(0, 3, len(date_range))
        trends['music_industry_index'] = pd.Series(music_trend, index=date_range)
        
        return trends
    
    def _plot_seasonal_trends(self, metrics_data: pd.DataFrame, trends: List[SeasonalTrend]) -> None:
        """Plot seasonal trends visualization."""
        if not trends:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Seasonal Trends Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series with trend
        ax = axes[0, 0]
        for col in metrics_data.columns[:2]:  # Show first 2 metrics
            ax.plot(metrics_data.index, metrics_data[col], label=col, alpha=0.7)
        ax.set_title('Original Time Series Data')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Seasonal strength comparison
        ax = axes[0, 1]
        trend_names = [f"Trend {i+1}" for i in range(len(trends))]
        strengths = [t.strength for t in trends]
        bars = ax.bar(trend_names, strengths, alpha=0.7)
        ax.set_title('Seasonal Strength by Trend')
        ax.set_ylabel('Seasonal Strength')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, strength in zip(bars, strengths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{strength:.3f}', ha='center', va='bottom')
        
        # Plot 3: Statistical significance
        ax = axes[1, 0]
        significance = [1 if t.statistical_significance else 0 for t in trends]
        p_values = [t.p_value for t in trends]
        colors = ['green' if sig else 'red' for sig in significance]
        
        bars = ax.bar(trend_names, p_values, color=colors, alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance Threshold')
        ax.set_title('Statistical Significance (p-values)')
        ax.set_ylabel('P-value')
        ax.set_yscale('log')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Trend directions
        ax = axes[1, 1]
        directions = [t.trend_direction for t in trends]
        direction_counts = pd.Series(directions).value_counts()
        colors = ['lightblue', 'lightcoral']
        ax.pie(direction_counts.values, labels=direction_counts.index, 
               autopct='%1.1f%%', colors=colors[:len(direction_counts)])
        ax.set_title('Trend Directions Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_event_impacts(self, metrics_data: pd.DataFrame, events_data: pd.DataFrame, impacts: List[EventImpact]) -> None:
        """Plot event impacts visualization."""
        if not impacts:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Event Impact Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Timeline with events
        ax = axes[0, 0]
        metric_col = metrics_data.columns[0]  # Use first metric
        ax.plot(metrics_data.index, metrics_data[metric_col], alpha=0.7, label='Metric Value')
        
        # Mark events
        for impact in impacts:
            ax.axvline(x=impact.event_date, color='red', alpha=0.6, linestyle='--')
            ax.annotate(f'{impact.event_type.value}\\n{impact.impact_magnitude*100:.1f}%',
                       xy=(impact.event_date, metrics_data[metric_col].max()*0.9),
                       ha='center', fontsize=8, rotation=45)
        
        ax.set_title('Timeline with Event Impacts')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Impact magnitudes
        ax = axes[0, 1]
        event_labels = [f"{i.event_type.value}\\n{i.event_date.strftime('%m-%d')}" for i in impacts]
        magnitudes = [i.impact_magnitude * 100 for i in impacts]  # Convert to percentage
        colors = ['green' if m > 0 else 'red' for m in magnitudes]
        
        bars = ax.bar(range(len(magnitudes)), magnitudes, color=colors, alpha=0.7)
        ax.set_xticks(range(len(event_labels)))
        ax.set_xticklabels(event_labels, rotation=45, ha='right')
        ax.set_title('Event Impact Magnitudes')
        ax.set_ylabel('Impact (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, magnitude in zip(bars, magnitudes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (1 if height > 0 else -3),
                   f'{magnitude:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 3: Recovery times
        ax = axes[1, 0]
        recovery_times = [i.recovery_days for i in impacts]
        ax.bar(range(len(recovery_times)), recovery_times, alpha=0.7, color='orange')
        ax.set_xticks(range(len(event_labels)))
        ax.set_xticklabels([el.split('\\n')[0] for el in event_labels], rotation=45, ha='right')
        ax.set_title('Recovery Times')
        ax.set_ylabel('Days to Recovery')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Statistical significance
        ax = axes[1, 1]
        p_values = [i.statistical_significance for i in impacts]
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        
        bars = ax.bar(range(len(p_values)), p_values, color=colors, alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance (0.05)')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Marginal (0.10)')
        ax.set_xticks(range(len(event_labels)))
        ax.set_xticklabels([el.split('\\n')[0] for el in event_labels], rotation=45, ha='right')
        ax.set_title('Statistical Significance (p-values)')
        ax.set_ylabel('P-value')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_holiday_effects(self, metrics_data: pd.DataFrame, effects: List[HolidayEffect]) -> None:
        """Plot holiday effects visualization."""
        if not effects:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Holiday Effects Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Holiday impact magnitudes
        ax = axes[0, 0]
        holiday_names = [e.holiday_type.value.replace('_', ' ').title() for e in effects]
        impacts = [e.avg_impact_magnitude * 100 for e in effects]  # Convert to percentage
        
        bars = ax.bar(holiday_names, impacts, alpha=0.7, color='purple')
        ax.set_title('Average Holiday Impact Magnitudes')
        ax.set_ylabel('Average Impact (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, impact in zip(bars, impacts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{impact:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Cultural significance vs Impact
        ax = axes[0, 1]
        cultural_scores = [e.cultural_significance_score for e in effects]
        ax.scatter(cultural_scores, impacts, alpha=0.7, s=100)
        
        # Add labels for each point
        for i, (x, y, name) in enumerate(zip(cultural_scores, impacts, holiday_names)):
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Cultural Significance Score')
        ax.set_ylabel('Average Impact (%)')
        ax.set_title('Cultural Significance vs Holiday Impact')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Consistency scores
        ax = axes[1, 0]
        consistency_scores = [e.consistency_score for e in effects]
        colors = plt.cm.RdYlGn(np.array(consistency_scores))
        
        bars = ax.bar(holiday_names, consistency_scores, color=colors, alpha=0.8)
        ax.set_title('Holiday Effect Consistency Scores')
        ax.set_ylabel('Consistency Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, consistency_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom')
        
        # Plot 4: Platform-specific effects for first holiday
        ax = axes[1, 1]
        if effects:
            first_holiday = effects[0]
            platforms = list(first_holiday.platform_specific_effects.keys())
            platform_effects = [first_holiday.platform_specific_effects[p] * 100 for p in platforms]
            
            bars = ax.bar(platforms, platform_effects, alpha=0.7, color='teal')
            ax.set_title(f'Platform Effects: {holiday_names[0]}')
            ax.set_ylabel('Platform-Specific Impact (%)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, effect in zip(bars, platform_effects):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{effect:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_global_correlations(self, metrics_data: pd.DataFrame, external_trends: Dict[str, pd.Series], correlations: List[GlobalTrendCorrelation]) -> None:
        """Plot global correlations visualization."""
        if not correlations:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Global Trend Correlations', fontsize=16, fontweight='bold')
        
        # Plot 1: Correlation coefficients
        ax = axes[0, 0]
        trend_names = [c.trend_name for c in correlations]
        coefficients = [c.correlation_coefficient for c in correlations]
        colors = ['green' if c > 0 else 'red' for c in coefficients]
        
        bars = ax.bar(trend_names, coefficients, color=colors, alpha=0.7)
        ax.set_title('Correlation Coefficients')
        ax.set_ylabel('Correlation Coefficient')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, coeff in zip(bars, coefficients):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 if height > 0 else -0.03),
                   f'{coeff:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 2: Strength categories
        ax = axes[0, 1]
        strength_categories = [c.strength_category for c in correlations]
        strength_counts = pd.Series(strength_categories).value_counts()
        colors = ['green', 'yellow', 'red']
        ax.pie(strength_counts.values, labels=strength_counts.index, autopct='%1.1f%%', colors=colors[:len(strength_counts)])
        ax.set_title('Correlation Strength Distribution')
        
        # Plot 3: Time series comparison (first correlation)
        ax = axes[1, 0]
        if correlations and external_trends:
            first_corr = correlations[0]
            metric_col = metrics_data.columns[0]
            trend_name = first_corr.trend_name
            
            # Normalize data for comparison
            metric_normalized = (metrics_data[metric_col] - metrics_data[metric_col].mean()) / metrics_data[metric_col].std()
            trend_normalized = (external_trends[trend_name] - external_trends[trend_name].mean()) / external_trends[trend_name].std()
            
            ax.plot(metrics_data.index, metric_normalized, label=f'{metric_col} (normalized)', alpha=0.7)
            ax.plot(external_trends[trend_name].index, trend_normalized, label=f'{trend_name} (normalized)', alpha=0.7)
            ax.set_title(f'Time Series Comparison: {first_corr.correlation_coefficient:.3f} correlation')
            ax.set_ylabel('Normalized Values')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Statistical significance
        ax = axes[1, 1]
        p_values = [c.p_value for c in correlations]
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        
        bars = ax.bar(trend_names, p_values, color=colors, alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance (0.05)')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Marginal (0.10)')
        ax.set_title('Statistical Significance (p-values)')
        ax.set_ylabel('P-value')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _create_comprehensive_dashboard(self, result: SeasonalAnalysisResult, metrics_data: pd.DataFrame) -> None:
        """Create comprehensive analysis dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        fig.suptitle(f'Comprehensive Seasonal Analysis: {result.artist_name}', fontsize=20, fontweight='bold')
        
        # Main time series (spans 2 columns)
        ax_main = fig.add_subplot(gs[0, :2])
        for col in metrics_data.columns:
            ax_main.plot(metrics_data.index, metrics_data[col], alpha=0.7, label=col)
        ax_main.set_title('Artist Metrics Time Series')
        ax_main.set_ylabel('Metric Values')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Key metrics summary
        ax_summary = fig.add_subplot(gs[0, 2:])
        summary_data = {
            'Overall Seasonality': result.overall_seasonality_score,
            'Predictability Index': result.predictability_index,
            'Data Quality': result.data_quality_score
        }
        bars = ax_summary.bar(summary_data.keys(), summary_data.values(), alpha=0.8, color=['blue', 'green', 'orange'])
        ax_summary.set_title('Key Performance Indicators')
        ax_summary.set_ylabel('Score (0-1)')
        ax_summary.set_ylim(0, 1)
        
        # Add value labels
        for bar, (key, value) in zip(bars, summary_data.items()):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Seasonal trends strength
        ax_seasonal = fig.add_subplot(gs[1, 0])
        if result.seasonal_trends:
            strengths = [st.strength for st in result.seasonal_trends]
            ax_seasonal.hist(strengths, bins=10, alpha=0.7, color='purple')
            ax_seasonal.axvline(np.mean(strengths), color='red', linestyle='--', label=f'Mean: {np.mean(strengths):.3f}')
        ax_seasonal.set_title('Seasonal Strength Distribution')
        ax_seasonal.set_xlabel('Seasonal Strength')
        ax_seasonal.set_ylabel('Frequency')
        ax_seasonal.legend()
        
        # Event impacts
        ax_events = fig.add_subplot(gs[1, 1])
        if result.event_impacts:
            impacts = [ei.impact_magnitude * 100 for ei in result.event_impacts]
            ax_events.hist(impacts, bins=10, alpha=0.7, color='red')
            ax_events.axvline(np.mean(impacts), color='blue', linestyle='--', label=f'Mean: {np.mean(impacts):.1f}%')
        ax_events.set_title('Event Impact Distribution')
        ax_events.set_xlabel('Impact Magnitude (%)')
        ax_events.set_ylabel('Frequency')
        ax_events.legend()
        
        # Holiday effects
        ax_holidays = fig.add_subplot(gs[1, 2])
        if result.holiday_effects:
            holiday_names = [he.holiday_type.value.replace('_', ' ')[:10] for he in result.holiday_effects]
            holiday_impacts = [he.avg_impact_magnitude * 100 for he in result.holiday_effects]
            ax_holidays.bar(holiday_names, holiday_impacts, alpha=0.7, color='green')
            ax_holidays.set_title('Holiday Effects')
            ax_holidays.set_ylabel('Average Impact (%)')
            ax_holidays.tick_params(axis='x', rotation=45)
        
        # Global correlations
        ax_correlations = fig.add_subplot(gs[1, 3])
        if result.global_correlations:
            corr_names = [gc.trend_name[:10] for gc in result.global_correlations]
            corr_coeffs = [gc.correlation_coefficient for gc in result.global_correlations]
            colors = ['green' if c > 0 else 'red' for c in corr_coeffs]
            ax_correlations.bar(corr_names, corr_coeffs, color=colors, alpha=0.7)
            ax_correlations.set_title('Global Correlations')
            ax_correlations.set_ylabel('Correlation Coefficient')
            ax_correlations.tick_params(axis='x', rotation=45)
            ax_correlations.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Analysis timeline (spans full width)
        ax_timeline = fig.add_subplot(gs[2, :])
        
        # Plot base metric
        metric_col = metrics_data.columns[0]
        ax_timeline.plot(metrics_data.index, metrics_data[metric_col], alpha=0.5, color='gray', label='Base Metric')
        
        # Mark events
        if result.event_impacts:
            for impact in result.event_impacts[:10]:  # Show top 10 events
                ax_timeline.axvline(x=impact.event_date, color='red', alpha=0.6, linestyle='--')
        
        # Mark holidays
        if result.holiday_effects:
            for effect in result.holiday_effects:
                for holiday_date in effect.holiday_dates[-5:]:  # Show recent occurrences
                    ax_timeline.axvline(x=holiday_date, color='green', alpha=0.4, linestyle=':')
        
        ax_timeline.set_title('Analysis Timeline (Red: Events, Green: Holidays)')
        ax_timeline.set_xlabel('Date')
        ax_timeline.set_ylabel('Metric Value')
        ax_timeline.legend()
        ax_timeline.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_performance_benchmark(self, benchmark_results: Dict[int, Any]) -> None:
        """Plot performance benchmarking results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Benchmarking Results', fontsize=16, fontweight='bold')
        
        days = list(benchmark_results.keys())
        
        # Plot 1: Execution times
        ax = axes[0, 0]
        analysis_times = [benchmark_results[d]['analysis_time'] for d in days]
        ax.plot(days, analysis_times, 'o-', linewidth=2, markersize=8)
        ax.set_title('Analysis Execution Time')
        ax.set_xlabel('Data Size (days)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Results count
        ax = axes[0, 1]
        seasonal_counts = [benchmark_results[d]['seasonal_trends'] for d in days]
        event_counts = [benchmark_results[d]['event_impacts'] for d in days]
        holiday_counts = [benchmark_results[d]['holiday_effects'] for d in days]
        correlation_counts = [benchmark_results[d]['global_correlations'] for d in days]
        
        width = 0.2
        x = np.arange(len(days))
        ax.bar(x - 1.5*width, seasonal_counts, width, label='Seasonal Trends', alpha=0.8)
        ax.bar(x - 0.5*width, event_counts, width, label='Event Impacts', alpha=0.8)
        ax.bar(x + 0.5*width, holiday_counts, width, label='Holiday Effects', alpha=0.8)
        ax.bar(x + 1.5*width, correlation_counts, width, label='Global Correlations', alpha=0.8)
        
        ax.set_title('Analysis Results Count')
        ax.set_xlabel('Data Size (days)')
        ax.set_ylabel('Number of Results')
        ax.set_xticks(x)
        ax.set_xticklabels(days)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Quality scores
        ax = axes[1, 0]
        data_quality = [benchmark_results[d]['data_quality_score'] for d in days]
        predictability = [benchmark_results[d]['predictability_index'] for d in days]
        
        ax.plot(days, data_quality, 'o-', label='Data Quality', linewidth=2, markersize=8)
        ax.plot(days, predictability, 's-', label='Predictability Index', linewidth=2, markersize=8)
        ax.set_title('Quality Metrics')
        ax.set_xlabel('Data Size (days)')
        ax.set_ylabel('Score (0-1)')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Performance efficiency
        ax = axes[1, 1]
        data_points = [benchmark_results[d]['data_points'] for d in days]
        efficiency = [benchmark_results[d]['analysis_time'] / benchmark_results[d]['data_points'] * 1000 for d in days]  # ms per data point
        
        ax.plot(data_points, efficiency, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_title('Processing Efficiency')
        ax.set_xlabel('Number of Data Points')
        ax.set_ylabel('Time per Data Point (ms)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def run_all_demos():
    """Run all demonstration functions."""
    print("🎵 K-POP SEASONAL PATTERN ANALYZER DEMONSTRATION 🎵")
    print("="*80)
    
    # Initialize demo
    demo = SeasonalPatternDemo()
    
    # Run individual demos
    demo.demo_seasonal_trends_analysis()
    demo.demo_event_impact_analysis()
    demo.demo_holiday_effects_analysis()
    demo.demo_global_trend_correlation()
    
    # Run comprehensive demo
    comprehensive_result = demo.demo_comprehensive_analysis()
    
    # Run performance benchmark
    benchmark_results = demo.benchmark_performance()
    
    print("\\n" + "="*80)
    print("🎯 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return {
        'comprehensive_result': comprehensive_result,
        'benchmark_results': benchmark_results,
        'demo_instance': demo
    }


if __name__ == "__main__":
    # Run demonstrations
    results = run_all_demos()
    
    print("\\n🏁 All demonstrations completed!")
    print("Check the generated visualizations for detailed analysis results.")