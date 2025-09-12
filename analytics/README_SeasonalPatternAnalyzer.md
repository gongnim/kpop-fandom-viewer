# SeasonalPatternAnalyzer Integration Guide

## Overview

The SeasonalPatternAnalyzer is a comprehensive seasonal analysis engine for K-Pop artist metrics, providing advanced capabilities for detecting seasonal trends, event impacts, holiday effects, and global trend correlations.

## Features

### 🔍 Core Analysis Methods

1. **`analyze_seasonal_trends()`** - Detects seasonal patterns in artist metrics
2. **`detect_event_impact()`** - Quantifies the impact of events on performance
3. **`analyze_holiday_effects()`** - Analyzes holiday effects with cultural context
4. **`correlate_global_trends()`** - Correlates artist metrics with external trends

### 📊 Analysis Capabilities

- **Multi-dimensional seasonal decomposition** with statistical significance testing
- **Event-driven anomaly detection** with impact quantification
- **Holiday impact analysis** with Korean cultural context
- **Cross-correlation analysis** with external data sources
- **Comprehensive reporting** with predictability scoring

## Quick Start

```python
from analytics.seasonal_pattern_analyzer import SeasonalPatternAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = SeasonalPatternAnalyzer()

# Prepare your data
metrics_data = pd.DataFrame({
    'youtube_subscribers': [1000, 1100, 1200, ...],
    'spotify_followers': [800, 850, 900, ...]
}, index=pd.date_range('2023-01-01', periods=365, freq='D'))

# Run comprehensive analysis
result = analyzer.generate_comprehensive_analysis(
    artist_id="artist_001",
    artist_name="Sample Artist",
    metrics_data=metrics_data
)

print(f"Seasonality Score: {result.overall_seasonality_score}")
print(f"Predictability Index: {result.predictability_index}")
```

## Integration Points

### 1. Database Integration

```python
# Connect with your database
from database_postgresql import DatabaseManager

db = DatabaseManager()
analyzer = SeasonalPatternAnalyzer(database_connection=db)

# Fetch artist data directly
artist_data = db.get_artist_metrics(artist_id, start_date, end_date)
events_data = db.get_artist_events(artist_id)

# Run analysis
result = analyzer.generate_comprehensive_analysis(
    artist_id, artist_name, artist_data, events_data
)
```

### 2. Streamlit Dashboard Integration

```python
# In your Streamlit page
import streamlit as st
from analytics import SeasonalPatternAnalyzer

st.title("🌊 Seasonal Pattern Analysis")

analyzer = SeasonalPatternAnalyzer()

# User inputs
artist_selected = st.selectbox("Select Artist", artist_options)
analysis_period = st.select_slider("Analysis Period", ["6 months", "1 year", "2 years", "3 years"])

if st.button("Run Analysis"):
    with st.spinner("Analyzing seasonal patterns..."):
        result = analyzer.generate_comprehensive_analysis(...)
        
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Seasonality Score", f"{result.overall_seasonality_score:.3f}")
    with col2:
        st.metric("Predictability Index", f"{result.predictability_index:.3f}")
    with col3:
        st.metric("Data Quality", f"{result.data_quality_score:.3f}")
```

### 3. Automated Analysis Pipeline

```python
# In scheduler.py
from analytics import SeasonalPatternAnalyzer

def run_weekly_seasonal_analysis():
    """Weekly seasonal analysis for all active artists."""
    analyzer = SeasonalPatternAnalyzer()
    
    active_artists = get_active_artists()
    
    for artist in active_artists:
        try:
            # Fetch latest data
            metrics_data = get_artist_metrics(artist.id, days=365)
            events_data = get_artist_events(artist.id)
            
            # Run analysis
            result = analyzer.generate_comprehensive_analysis(
                artist.id, artist.name, metrics_data, events_data
            )
            
            # Store results
            store_seasonal_analysis(artist.id, result)
            
            # Generate alerts if needed
            if result.predictability_index < 0.3:
                create_alert(artist.id, "Low predictability detected")
                
        except Exception as e:
            logger.error(f"Seasonal analysis failed for {artist.name}: {e}")

# Schedule the analysis
scheduler.add_job(run_weekly_seasonal_analysis, 'cron', day_of_week='mon')
```

## Configuration Options

### Analysis Parameters

```python
analysis_config = {
    'period_analysis': 'monthly',     # 'weekly', 'monthly', 'quarterly'
    'impact_window_days': 30,         # Days to analyze after events
    'baseline_days': 14,              # Days before event for baseline
    'analysis_years': 3,              # Years of data for holiday analysis
    'correlation_window': 365,        # Days for correlation analysis
    'min_correlation': 0.3            # Minimum correlation threshold
}

result = analyzer.generate_comprehensive_analysis(
    artist_id, artist_name, metrics_data,
    analysis_config=analysis_config
)
```

### Cultural Context Settings

```python
# Korean cultural context (default)
analyzer = SeasonalPatternAnalyzer()

# Custom holiday definitions
analyzer.korean_holidays[HolidayType.CUSTOM_HOLIDAY] = {
    "significance": 0.8, 
    "duration": 2
}
```

## Output Data Structures

### SeasonalAnalysisResult

```python
result = analyzer.generate_comprehensive_analysis(...)

# Access analysis components
seasonal_trends = result.seasonal_trends      # List[SeasonalTrend]
event_impacts = result.event_impacts          # List[EventImpact]
holiday_effects = result.holiday_effects      # List[HolidayEffect]
global_correlations = result.global_correlations  # List[GlobalTrendCorrelation]

# Key metrics
overall_seasonality = result.overall_seasonality_score  # float 0-1
predictability = result.predictability_index           # float 0-1
data_quality = result.data_quality_score              # float 0-1
```

### Individual Analysis Results

```python
# Seasonal trends
for trend in result.seasonal_trends:
    print(f"Strength: {trend.strength}")
    print(f"Peak periods: {trend.peak_periods}")
    print(f"Statistical significance: {trend.statistical_significance}")

# Event impacts
for impact in result.event_impacts:
    print(f"Event: {impact.event_type.value}")
    print(f"Impact magnitude: {impact.impact_magnitude}")
    print(f"Recovery days: {impact.recovery_days}")

# Holiday effects
for effect in result.holiday_effects:
    print(f"Holiday: {effect.holiday_type.value}")
    print(f"Average impact: {effect.avg_impact_magnitude}")
    print(f"Consistency: {effect.consistency_score}")
```

## Performance Considerations

### Data Requirements

- **Minimum data**: 60 days for seasonal analysis
- **Recommended data**: 1-3 years for comprehensive analysis
- **Event data**: Optional but improves analysis quality
- **External trends**: Optional for correlation analysis

### Performance Benchmarks

| Data Size | Analysis Time | Memory Usage |
|-----------|---------------|--------------|
| 365 days  | ~2-3 seconds  | ~10-20 MB    |
| 730 days  | ~4-6 seconds  | ~15-30 MB    |
| 1095 days | ~6-10 seconds | ~20-40 MB    |

### Optimization Tips

1. **Use appropriate time windows** - Don't analyze more data than needed
2. **Cache external trends** - Reuse external data across multiple artists
3. **Batch processing** - Analyze multiple artists in parallel when possible
4. **Filter significant results** - Set appropriate thresholds to reduce noise

## Error Handling

```python
try:
    result = analyzer.generate_comprehensive_analysis(...)
except ValueError as e:
    logger.error(f"Invalid data provided: {e}")
except Exception as e:
    logger.error(f"Analysis failed: {e}")
    # Fallback to simpler analysis or default values
```

## Example Integration in Dashboard Page

```python
# pages/6_계절성_패턴_분석.py
import streamlit as st
import pandas as pd
from analytics import SeasonalPatternAnalyzer
from utils.charts import create_seasonal_chart

st.title("🌊 계절성 패턴 분석")

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return SeasonalPatternAnalyzer()

analyzer = get_analyzer()

# Artist selection
artists = get_all_artists()
selected_artist = st.selectbox("아티스트 선택", artists)

if selected_artist:
    # Analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        analysis_years = st.slider("분석 기간 (년)", 1, 5, 3)
    with col2:
        period_type = st.selectbox("분석 주기", ["monthly", "weekly", "quarterly"])
    
    if st.button("계절성 분석 시작", type="primary"):
        with st.spinner("계절성 패턴을 분석중입니다..."):
            # Fetch data
            metrics_data = get_artist_metrics(selected_artist.id, days=365*analysis_years)
            events_data = get_artist_events(selected_artist.id)
            
            # Run analysis
            result = analyzer.generate_comprehensive_analysis(
                selected_artist.id,
                selected_artist.name,
                metrics_data,
                events_data,
                analysis_config={'period_analysis': period_type, 'analysis_years': analysis_years}
            )
            
            # Display results
            st.success("분석 완료!")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "전체 계절성 점수",
                    f"{result.overall_seasonality_score:.3f}",
                    help="0-1 범위, 높을수록 계절성이 강함"
                )
            with col2:
                st.metric(
                    "예측가능성 지수",
                    f"{result.predictability_index:.3f}",
                    help="패턴의 예측가능성 정도"
                )
            with col3:
                st.metric(
                    "데이터 품질",
                    f"{result.data_quality_score:.3f}",
                    help="분석에 사용된 데이터의 품질"
                )
            
            # Seasonal trends
            if result.seasonal_trends:
                st.subheader("🔄 계절성 트렌드")
                for i, trend in enumerate(result.seasonal_trends):
                    with st.expander(f"트렌드 {i+1} - 강도: {trend.strength:.3f}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**분석 주기**: {trend.period}")
                            st.write(f"**트렌드 방향**: {trend.trend_direction}")
                            st.write(f"**통계적 유의성**: {'예' if trend.statistical_significance else '아니오'}")
                        with col2:
                            st.write(f"**피크 시기**: {', '.join(trend.peak_periods[:3])}")
                            st.write(f"**저점 시기**: {', '.join(trend.trough_periods[:3])}")
                            st.write(f"**신뢰도**: {trend.confidence:.3f}")
            
            # Event impacts
            if result.event_impacts:
                st.subheader("🎯 이벤트 영향 분석")
                for impact in result.event_impacts[:5]:
                    impact_pct = impact.impact_magnitude * 100
                    st.write(f"**{impact.event_type.value}** ({impact.event_date.strftime('%Y-%m-%d')}): "
                            f"{impact_pct:+.1f}% 영향")
            
            # Holiday effects
            if result.holiday_effects:
                st.subheader("🎄 휴일 효과 분석")
                holiday_df = pd.DataFrame([
                    {
                        "휴일": effect.holiday_type.value.replace('_', ' ').title(),
                        "평균 영향": f"{effect.avg_impact_magnitude*100:.1f}%",
                        "일관성": f"{effect.consistency_score:.2f}",
                        "문화적 중요도": f"{effect.cultural_significance_score:.2f}"
                    }
                    for effect in result.holiday_effects
                ])
                st.dataframe(holiday_df, use_container_width=True)
            
            # Global correlations
            if result.global_correlations:
                st.subheader("🌐 글로벌 트렌드 상관관계")
                corr_df = pd.DataFrame([
                    {
                        "트렌드": corr.trend_name,
                        "상관계수": f"{corr.correlation_coefficient:.3f}",
                        "강도": corr.strength_category,
                        "유의성": f"{corr.p_value:.4f}"
                    }
                    for corr in result.global_correlations
                ])
                st.dataframe(corr_df, use_container_width=True)
```

## Best Practices

1. **Data Quality**: Ensure clean, consistent time series data
2. **Analysis Frequency**: Run comprehensive analysis weekly/monthly, not daily
3. **Threshold Tuning**: Adjust correlation and significance thresholds based on your data
4. **Cultural Context**: Leverage Korean holiday calendar for accurate holiday analysis
5. **Result Interpretation**: Combine quantitative results with domain expertise
6. **Performance Monitoring**: Track analysis execution times and optimize as needed

## Troubleshooting

### Common Issues

1. **No seasonal trends detected**: Check data quality and time span (need 60+ days)
2. **Low correlation scores**: Verify external trend data alignment and relevance
3. **Memory issues**: Reduce analysis window or batch process multiple artists
4. **Missing holiday effects**: Ensure sufficient historical data (2+ years recommended)

### Debugging Tools

```python
# Enable detailed logging
import logging
logging.getLogger('analytics.seasonal_pattern_analyzer').setLevel(logging.DEBUG)

# Check package status
from analytics import get_package_info
info = get_package_info()
print(f"Seasonal analyzer available: {info['seasonal_pattern_analyzer_available']}")

# Validate data before analysis
def validate_metrics_data(data):
    assert isinstance(data.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert len(data) >= 60, "Need at least 60 days of data"
    assert not data.empty, "Data cannot be empty"
    return True
```