# K-Pop Dashboard Analytics Package

**Version:** 1.0.0  
**Author:** Backend Development Team  
**Date:** 2025-09-08

## 📋 Overview

The Analytics package provides comprehensive backend capabilities for K-Pop artist performance analysis and growth rate calculations. This package is designed with enterprise-grade reliability, statistical accuracy, and scalable performance.

## 🏗️ Architecture

```
kpop_dashboard/analytics/
├── __init__.py                    # Package initialization and exports
├── growth_rate_calculator.py     # Core growth rate calculation engine
└── README.md                      # Documentation (this file)
```

## 📊 Core Features

### Growth Rate Calculator Engine
- **Multiple Algorithms:** Simple, rolling average, weighted, exponential smoothing
- **Time Periods:** Daily, weekly, monthly, quarterly, yearly calculations
- **Statistical Validation:** Outlier detection, confidence intervals, significance testing
- **Data Quality Assessment:** Quality scoring and validation framework
- **Batch Processing:** Efficient handling of multiple accounts/platforms

### Statistical Analysis
- **Outlier Detection:** Z-score based anomaly detection
- **Confidence Intervals:** Statistical uncertainty quantification
- **Significance Testing:** Statistical significance determination
- **Data Quality Scoring:** Multi-factor quality assessment

### Performance Optimization
- **Efficient Algorithms:** Optimized calculation methods for large datasets
- **Memory Management:** Minimal memory footprint for batch operations
- **Error Handling:** Comprehensive error handling and logging
- **Type Safety:** Full type hint coverage for reliability

## 🚀 Quick Start

### Basic Usage

```python
from kpop_dashboard.analytics import (
    GrowthRateCalculator,
    MetricDataPoint,
    CalculationMethod,
    GrowthPeriod
)
from datetime import datetime, timedelta

# Create calculator instance
calculator = GrowthRateCalculator(
    min_data_points=3,
    outlier_threshold=2.5,
    confidence_level=0.95
)

# Prepare data points
data_points = [
    MetricDataPoint(
        value=1500000,
        timestamp=datetime.now(),
        platform='youtube',
        metric_type='subscribers',
        quality_score=0.95
    ),
    MetricDataPoint(
        value=1450000,
        timestamp=datetime.now() - timedelta(days=1),
        platform='youtube', 
        metric_type='subscribers',
        quality_score=0.98
    ),
    # ... more data points
]

# Calculate growth rate
result = calculator.calculate_growth_rate(
    data_points=data_points,
    method=CalculationMethod.ROLLING_AVERAGE,
    period=GrowthPeriod.DAILY
)

if result:
    print(f"Growth Rate: {result.growth_rate:.2f}%")
    print(f"Data Quality: {result.data_quality_score:.3f}")
    print(f"Significant: {result.is_significant}")
    if result.confidence_interval:
        lower, upper = result.confidence_interval
        print(f"Confidence Interval: [{lower:.2f}%, {upper:.2f}%]")
```

### Standalone Functions

```python
from kpop_dashboard.analytics import (
    calculate_daily_growth_rate,
    calculate_compound_growth_rate,
    classify_growth_rate
)

# Simple daily growth calculation
daily_growth = calculate_daily_growth_rate(
    current_value=1500000,
    previous_value=1450000  
)
print(f"Daily Growth: {daily_growth:.2f}%")

# Compound growth rate
cagr = calculate_compound_growth_rate(
    initial_value=1000000,
    final_value=1500000,
    periods=365  # days
)
print(f"CAGR: {cagr:.2f}%")

# Growth classification
category = classify_growth_rate(daily_growth)
print(f"Growth Category: {category}")
```

## 🎯 Advanced Usage

### Multiple Period Analysis

```python
# Calculate growth for all periods
results = calculator.calculate_multiple_periods(
    data_points=data_points,
    method=CalculationMethod.EXPONENTIAL_SMOOTHING
)

for period_name, result in results.items():
    if result:
        print(f"{period_name}: {result.growth_rate:.2f}%")
```

### Custom Configuration

```python
# High-precision calculator for critical analysis
precision_calculator = GrowthRateCalculator(
    min_data_points=10,        # Require more data points
    outlier_threshold=2.0,     # More strict outlier detection
    confidence_level=0.99      # Higher confidence level
)

# Quality-weighted calculations
result = precision_calculator.calculate_growth_rate(
    data_points=data_points,
    method=CalculationMethod.WEIGHTED,  # Use quality scores
    period=GrowthPeriod.MONTHLY
)
```

### Batch Processing

```python
# Process multiple accounts efficiently
accounts_data = {
    'account_1': data_points_1,
    'account_2': data_points_2,
    'account_3': data_points_3,
    # ... more accounts
}

batch_results = {}
for account_id, data in accounts_data.items():
    result = calculator.calculate_growth_rate(
        data_points=data,
        method=CalculationMethod.ROLLING_AVERAGE,
        period=GrowthPeriod.WEEKLY
    )
    batch_results[account_id] = result

# Analysis of batch results
successful_calculations = sum(1 for r in batch_results.values() if r is not None)
print(f"Successfully calculated growth for {successful_calculations} accounts")
```

## 📈 Calculation Methods

### 1. Simple Growth Rate
Basic percentage change between two points:
```
growth_rate = ((current - previous) / previous) * 100
```

### 2. Rolling Average Growth
Uses moving averages to smooth out volatility:
```python
method=CalculationMethod.ROLLING_AVERAGE
```

### 3. Weighted Growth Rate  
Incorporates data quality scores:
```python
method=CalculationMethod.WEIGHTED
```

### 4. Exponential Smoothing
Advanced time-series analysis method:
```python
method=CalculationMethod.EXPONENTIAL_SMOOTHING
```

## 🔍 Data Quality Framework

### Quality Score Components
- **Base Quality:** Individual data point quality scores
- **Outlier Penalty:** Reduction for detected anomalies
- **Sample Size Bonus:** Reward for larger datasets
- **Consistency Bonus:** Reward for low variance

### Quality Score Interpretation
- **0.90-1.00:** Excellent data quality
- **0.70-0.89:** Good data quality
- **0.50-0.69:** Fair data quality (use with caution)
- **Below 0.50:** Poor data quality (results may be unreliable)

## ⚠️ Growth Rate Classification

| Category | Daily Growth Rate | Interpretation |
|----------|------------------|----------------|
| Explosive | ≥100% | Extraordinary growth (requires investigation) |
| Rapid | 50-99% | Very high growth |
| Significant | 25-49% | High growth |
| Moderate | 10-24% | Good growth |
| Stable | 0-9% | Normal/stable |
| Declining | <0% | Negative growth |

## 🚨 Error Handling

### Common Scenarios
```python
try:
    result = calculator.calculate_growth_rate(data_points, method, period)
    if result is None:
        print("Calculation failed: Insufficient or invalid data")
    elif result.outlier_detected:
        print("Warning: Outliers detected in data")
    elif result.data_quality_score < 0.7:
        print("Warning: Low data quality, use results with caution")
except Exception as e:
    print(f"Calculation error: {e}")
```

### Data Validation
- **Minimum Data Points:** At least 2 data points required
- **Non-negative Values:** All metric values must be ≥ 0
- **Valid Timestamps:** All timestamps must be datetime objects
- **Platform/Metric Validation:** Uses package constants for validation

## 📊 Integration Examples

### Database Integration
```python
from kpop_dashboard.database_postgresql import get_all_metrics_for_artist
from kpop_dashboard.analytics import GrowthRateCalculator, MetricDataPoint

def calculate_artist_growth(artist_id: int, platform: str, metric_type: str):
    """Calculate growth rate for artist from database."""
    
    # Get data from database
    metrics_data = get_all_metrics_for_artist(artist_id)
    
    # Filter and convert to MetricDataPoint objects
    data_points = []
    for metric in metrics_data:
        if metric['platform'] == platform and metric['metric_type'] == metric_type:
            data_points.append(MetricDataPoint(
                value=metric['value'],
                timestamp=metric['collected_at'],
                platform=metric['platform'],
                metric_type=metric['metric_type'],
                quality_score=1.0  # Default quality score
            ))
    
    # Calculate growth rate
    calculator = GrowthRateCalculator()
    result = calculator.calculate_growth_rate(
        data_points=data_points,
        method=CalculationMethod.ROLLING_AVERAGE,
        period=GrowthPeriod.MONTHLY
    )
    
    return result
```

### Alert Integration
```python
from kpop_dashboard.analytics import get_severity_level, classify_growth_rate

def generate_growth_alert(growth_result):
    """Generate alert based on growth calculation result."""
    
    if growth_result is None:
        return None
    
    growth_rate = growth_result.growth_rate
    is_decline = growth_rate < 0
    
    alert_data = {
        'growth_rate': growth_rate,
        'growth_category': classify_growth_rate(growth_rate),
        'severity_level': get_severity_level(growth_rate, is_decline),
        'is_significant': growth_result.is_significant,
        'confidence_interval': growth_result.confidence_interval,
        'data_quality': growth_result.data_quality_score
    }
    
    return alert_data
```

## 🔧 Configuration

### Package Constants
```python
from kpop_dashboard.analytics import (
    SUPPORTED_PLATFORMS,      # ['youtube', 'spotify', 'twitter', ...]
    SUPPORTED_METRICS,        # ['subscribers', 'followers', 'views', ...]
    CALCULATION_METHODS,      # ['simple', 'rolling_average', ...]
    GROWTH_RATE_THRESHOLDS   # {'explosive': 100.0, 'rapid': 50.0, ...}
)
```

### Validation Functions
```python
from kpop_dashboard.analytics import validate_platform, validate_metric_type

# Validate input parameters
if not validate_platform('youtube'):
    raise ValueError("Unsupported platform")

if not validate_metric_type('subscribers'):
    raise ValueError("Unsupported metric type")
```

## 📝 Best Practices

### 1. Data Preparation
- Ensure data points are sorted by timestamp
- Use consistent quality scores across data points
- Include sufficient historical data for reliable calculations

### 2. Method Selection
- Use **Simple** for basic comparisons
- Use **Rolling Average** for general analysis (recommended)
- Use **Weighted** when data quality varies significantly
- Use **Exponential Smoothing** for time-series forecasting

### 3. Result Interpretation
- Always check `data_quality_score` before using results
- Consider `confidence_interval` for uncertainty assessment
- Use `is_significant` for statistical validity
- Monitor `outlier_detected` flag for data anomalies

### 4. Performance Optimization
- Batch process multiple calculations when possible
- Reuse calculator instances for consistent configuration
- Monitor memory usage for large datasets

## 🐛 Troubleshooting

### Common Issues

**Issue:** `calculate_growth_rate()` returns `None`
- **Cause:** Insufficient data points (< min_data_points)
- **Solution:** Provide more historical data or reduce min_data_points

**Issue:** Low `data_quality_score`
- **Cause:** Many outliers or inconsistent data
- **Solution:** Review data collection process or adjust outlier_threshold

**Issue:** `is_significant = False`
- **Cause:** Growth rate not statistically significant
- **Solution:** Collect more data or review calculation method

### Debug Mode
```python
import logging
logging.getLogger('kpop_dashboard.analytics').setLevel(logging.DEBUG)
```

## 🔄 Version History

- **v1.0.0** (2025-09-08): Initial release with core calculation engine

## 📚 API Reference

### Classes
- `GrowthRateCalculator`: Main calculation engine
- `MetricDataPoint`: Data structure for metric measurements
- `GrowthRateResult`: Result structure with metadata
- `CalculationMethod`: Enumeration of calculation methods
- `GrowthPeriod`: Enumeration of time periods

### Functions
- `calculate_daily_growth_rate()`: Simple daily growth calculation
- `calculate_weekly_growth_rate()`: Weekly growth from data points
- `calculate_monthly_growth_rate()`: Monthly growth from data points
- `calculate_rolling_average_growth()`: Rolling average calculation
- `calculate_compound_growth_rate()`: CAGR calculation
- `classify_growth_rate()`: Growth rate categorization
- `get_severity_level()`: Alert severity determination
- `validate_platform()`: Platform validation
- `validate_metric_type()`: Metric type validation

For detailed API documentation, see the docstrings in each module.