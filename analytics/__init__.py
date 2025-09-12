"""
K-Pop Dashboard Analytics Package
==================================

Backend analytics package for K-Pop artist performance analysis and growth rate calculations.

This package provides:
- Growth rate calculation engines
- Statistical analysis tools  
- Performance metrics processors
- Predictive modeling utilities
- Alert generation systems

Core Modules:
- growth_rate_calculator: Core growth rate calculation logic
- metrics_processor: Metrics aggregation and processing
- trend_analyzer: Trend detection and analysis
- alert_engine: Intelligent alert generation
- prediction_models: ML-based prediction models

Author: Backend Development Team
Version: 1.0.0
Date: 2025-09-08
"""

from typing import Dict, List, Any, Optional
import logging

# Package metadata
__version__ = "1.0.0"
__author__ = "Backend Development Team"
__email__ = "backend@kpop-dashboard.com"
__description__ = "K-Pop Dashboard Analytics Package for growth rate calculation and performance analysis"

# Configure package-level logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)

# Import core modules when available
try:
    from .growth_rate_calculator import (
        GrowthRateCalculator,
        MetricDataPoint,
        GrowthRateResult,
        CalculationMethod,
        GrowthPeriod,
        calculate_daily_growth_rate,
        calculate_weekly_growth_rate,
        calculate_monthly_growth_rate,
        calculate_rolling_average_growth,
        calculate_compound_growth_rate
    )
    logger.info("Successfully imported growth rate calculator")
    
    # Core calculator classes
    _GROWTH_CALCULATOR_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Growth rate calculator not available: {e}")
    _GROWTH_CALCULATOR_AVAILABLE = False

# Import ranking system when available
try:
    from .ranking_system import (
        GrowthRankingEngine,
        RankingCategory,
        RankingPeriod,
        DebutCohort,
        ArtistMetrics,
        RankingResult,
        CompositeIndex
    )
    logger.info("Successfully imported ranking system")
    
    # Ranking system classes
    _RANKING_SYSTEM_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Ranking system not available: {e}")
    _RANKING_SYSTEM_AVAILABLE = False

# Import ranking optimizer when available
try:
    from .ranking_optimizer import (
        OptimizedRankingEngine,
        IntelligentCache,
        BatchRankingProcessor,
        CacheLevel,
        OptimizationStrategy,
        PerformanceMetrics
    )
    logger.info("Successfully imported ranking optimizer")
    
    # Ranking optimizer classes
    _RANKING_OPTIMIZER_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Ranking optimizer not available: {e}")
    _RANKING_OPTIMIZER_AVAILABLE = False

# Import alert system when available
try:
    from .alert_system import (
        AlertEngine,
        Alert,
        AlertType,
        AlertSeverity,
        AlertStatus,
        AlertThresholds,
        AnomalyDetectionMethod,
        AnomalyResult
    )
    logger.info("Successfully imported alert system")
    
    # Alert system classes
    _ALERT_SYSTEM_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Alert system not available: {e}")
    _ALERT_SYSTEM_AVAILABLE = False

# Import prediction models when available
try:
    from .prediction_models import (
        PredictionEngine,
        PredictiveModelingEngine,
        PredictionResult,
        SeasonalityAnalysis,
        ModelConfig,
        ModelType,
        PredictionHorizon,
        ConfidenceLevel
    )
    logger.info("Successfully imported prediction models")
    _PREDICTION_MODELS_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Prediction models not available: {e}")
    _PREDICTION_MODELS_AVAILABLE = False

# Import model validation system when available
try:
    from .model_validation import (
        ValidationEngine,
        ModelValidator,
        HyperparameterTuner,
        ModelComparator,
        ValidationVisualizer,
        ValidationResult,
        TuningResult,
        ModelComparisonResult,
        LearningCurveResult,
        ValidationStrategy,
        TuningMethod,
        ComparisonMetric,
        get_validation_system_info,
        LIBRARIES_AVAILABLE
    )
    logger.info("Successfully imported model validation system")
    _MODEL_VALIDATION_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Model validation system not available: {e}")
    _MODEL_VALIDATION_AVAILABLE = False

# Import seasonal pattern analyzer when available
try:
    from .seasonal_pattern_analyzer import (
        SeasonalPatternAnalyzer,
        SeasonalTrend,
        EventImpact,
        HolidayEffect,
        GlobalTrendCorrelation,
        SeasonalAnalysisResult,
        SeasonalComponent,
        EventType,
        HolidayType,
        TrendCorrelationType
    )
    logger.info("Successfully imported seasonal pattern analyzer")
    _SEASONAL_PATTERN_ANALYZER_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Seasonal pattern analyzer not available: {e}")
    _SEASONAL_PATTERN_ANALYZER_AVAILABLE = False

# Import K-Pop event calendar system when available
try:
    from .kpop_event_calendar import (
        KPopEventCalendar,
        KPopEvent,
        EventCategory,
        EventImportance,
        ComebackSeason,
        AwardShow,
        EventImpactMetrics
    )
    logger.info("Successfully imported K-Pop event calendar system")
    _KPOP_EVENT_CALENDAR_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"K-Pop event calendar system not available: {e}")
    _KPOP_EVENT_CALENDAR_AVAILABLE = False

# Import award shows data manager when available
try:
    from .award_shows_data import (
        AwardShowDataManager,
        AwardShowInfo,
        AwardResult,
        AwardCategory
    )
    logger.info("Successfully imported award shows data manager")
    _AWARD_SHOWS_DATA_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Award shows data manager not available: {e}")
    _AWARD_SHOWS_DATA_AVAILABLE = False

# Import comeback season analyzer when available
try:
    from .comeback_season_analyzer import (
        ComebackSeasonAnalyzer,
        ComebackAnalysis,
        SeasonalPattern,
        CompetitionLevel,
        PerformanceMetric
    )
    logger.info("Successfully imported comeback season analyzer")
    _COMEBACK_SEASON_ANALYZER_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Comeback season analyzer not available: {e}")
    _COMEBACK_SEASON_ANALYZER_AVAILABLE = False

# Import event impact analyzer when available
try:
    from .event_impact_analyzer import (
        EventImpactAnalyzer,
        EventImpactAnalysis,
        ImpactMeasurement,
        ImpactType,
        ImpactDirection,
        ImpactMagnitude
    )
    logger.info("Successfully imported event impact analyzer")
    _EVENT_IMPACT_ANALYZER_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Event impact analyzer not available: {e}")
    _EVENT_IMPACT_ANALYZER_AVAILABLE = False

# Import additional modules as they become available
try:
    # Future modules will be imported here
    # from .metrics_processor import MetricsProcessor
    # from .trend_analyzer import TrendAnalyzer
    # from .prediction_models import PredictionModels
    pass
except ImportError:
    pass

# Package-level constants
SUPPORTED_PLATFORMS = [
    'youtube', 'spotify', 'twitter', 'instagram', 'tiktok', 'melon'
]

SUPPORTED_METRICS = [
    'subscribers', 'followers', 'total_views', 'monthly_listeners', 
    'popularity', 'likes', 'plays'
]

CALCULATION_METHODS = [
    'simple', 'rolling_average', 'weighted', 'exponential_smoothing'
]

GROWTH_RATE_THRESHOLDS = {
    'explosive': 100.0,      # 100%+ daily growth
    'rapid': 50.0,           # 50%+ daily growth  
    'significant': 25.0,     # 25%+ daily growth
    'moderate': 10.0,        # 10%+ daily growth
    'stable': 0.0,           # 0%+ daily growth
    'declining': -10.0       # -10% or worse
}

ALERT_SEVERITY_LEVELS = {
    'critical': 'red',
    'warning': 'yellow', 
    'info': 'green'
}

# Package-level utility functions
def get_package_info() -> Dict[str, Any]:
    """Get package information and status."""
    return {
        'name': __name__,
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'growth_calculator_available': _GROWTH_CALCULATOR_AVAILABLE,
        'ranking_system_available': _RANKING_SYSTEM_AVAILABLE,
        'ranking_optimizer_available': _RANKING_OPTIMIZER_AVAILABLE,
        'alert_system_available': _ALERT_SYSTEM_AVAILABLE,
        'prediction_models_available': _PREDICTION_MODELS_AVAILABLE,
        'model_validation_available': _MODEL_VALIDATION_AVAILABLE,
        'seasonal_pattern_analyzer_available': _SEASONAL_PATTERN_ANALYZER_AVAILABLE,
        'kpop_event_calendar_available': _KPOP_EVENT_CALENDAR_AVAILABLE,
        'award_shows_data_available': _AWARD_SHOWS_DATA_AVAILABLE,
        'comeback_season_analyzer_available': _COMEBACK_SEASON_ANALYZER_AVAILABLE,
        'event_impact_analyzer_available': _EVENT_IMPACT_ANALYZER_AVAILABLE,
        'supported_platforms': SUPPORTED_PLATFORMS,
        'supported_metrics': SUPPORTED_METRICS,
        'calculation_methods': CALCULATION_METHODS
    }

def validate_platform(platform: str) -> bool:
    """Validate if platform is supported."""
    return platform.lower() in SUPPORTED_PLATFORMS

def validate_metric_type(metric_type: str) -> bool:
    """Validate if metric type is supported."""
    return metric_type.lower() in SUPPORTED_METRICS

def classify_growth_rate(growth_rate: float) -> str:
    """Classify growth rate into categories."""
    if growth_rate >= GROWTH_RATE_THRESHOLDS['explosive']:
        return 'explosive'
    elif growth_rate >= GROWTH_RATE_THRESHOLDS['rapid']:
        return 'rapid'
    elif growth_rate >= GROWTH_RATE_THRESHOLDS['significant']:
        return 'significant'
    elif growth_rate >= GROWTH_RATE_THRESHOLDS['moderate']:
        return 'moderate'
    elif growth_rate >= GROWTH_RATE_THRESHOLDS['stable']:
        return 'stable'
    else:
        return 'declining'

def get_severity_level(growth_rate: float, is_decline: bool = False) -> str:
    """Determine alert severity level based on growth rate."""
    abs_rate = abs(growth_rate)
    
    if is_decline:
        # For declines, higher absolute values are more critical
        if abs_rate >= 50.0:
            return ALERT_SEVERITY_LEVELS['critical']
        elif abs_rate >= 25.0:
            return ALERT_SEVERITY_LEVELS['warning']
        else:
            return ALERT_SEVERITY_LEVELS['info']
    else:
        # For growth, very high rates might need attention too
        if abs_rate >= 100.0:
            return ALERT_SEVERITY_LEVELS['warning']  # Unusual growth
        elif abs_rate >= 50.0:
            return ALERT_SEVERITY_LEVELS['info']     # Good growth
        else:
            return ALERT_SEVERITY_LEVELS['info']     # Normal growth

# Export main components
__all__ = [
    # Constants
    'SUPPORTED_PLATFORMS',
    'SUPPORTED_METRICS', 
    'CALCULATION_METHODS',
    'GROWTH_RATE_THRESHOLDS',
    'ALERT_SEVERITY_LEVELS',
    
    # Utility functions
    'get_package_info',
    'validate_platform',
    'validate_metric_type', 
    'classify_growth_rate',
    'get_severity_level',
    
    # Core classes (when available)
    'GrowthRateCalculator' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'MetricDataPoint' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'GrowthRateResult' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'CalculationMethod' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'GrowthPeriod' if _GROWTH_CALCULATOR_AVAILABLE else None,
    
    # Core functions (when available)  
    'calculate_daily_growth_rate' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'calculate_weekly_growth_rate' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'calculate_monthly_growth_rate' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'calculate_rolling_average_growth' if _GROWTH_CALCULATOR_AVAILABLE else None,
    'calculate_compound_growth_rate' if _GROWTH_CALCULATOR_AVAILABLE else None,
    
    # Ranking system classes (when available)
    'GrowthRankingEngine' if _RANKING_SYSTEM_AVAILABLE else None,
    'RankingCategory' if _RANKING_SYSTEM_AVAILABLE else None,
    'RankingPeriod' if _RANKING_SYSTEM_AVAILABLE else None,
    'DebutCohort' if _RANKING_SYSTEM_AVAILABLE else None,
    'ArtistMetrics' if _RANKING_SYSTEM_AVAILABLE else None,
    'RankingResult' if _RANKING_SYSTEM_AVAILABLE else None,
    'CompositeIndex' if _RANKING_SYSTEM_AVAILABLE else None,
    
    # Ranking optimizer classes (when available)
    'OptimizedRankingEngine' if _RANKING_OPTIMIZER_AVAILABLE else None,
    'IntelligentCache' if _RANKING_OPTIMIZER_AVAILABLE else None,
    'BatchRankingProcessor' if _RANKING_OPTIMIZER_AVAILABLE else None,
    'CacheLevel' if _RANKING_OPTIMIZER_AVAILABLE else None,
    'OptimizationStrategy' if _RANKING_OPTIMIZER_AVAILABLE else None,
    'PerformanceMetrics' if _RANKING_OPTIMIZER_AVAILABLE else None,
    
    # Alert system classes (when available)
    'AlertEngine' if _ALERT_SYSTEM_AVAILABLE else None,
    'Alert' if _ALERT_SYSTEM_AVAILABLE else None,
    'AlertType' if _ALERT_SYSTEM_AVAILABLE else None,
    'AlertSeverity' if _ALERT_SYSTEM_AVAILABLE else None,
    'AlertStatus' if _ALERT_SYSTEM_AVAILABLE else None,
    'AlertThresholds' if _ALERT_SYSTEM_AVAILABLE else None,
    'AnomalyDetectionMethod' if _ALERT_SYSTEM_AVAILABLE else None,
    'AnomalyResult' if _ALERT_SYSTEM_AVAILABLE else None,
    
    # Model validation system classes (when available)
    'ValidationEngine' if _MODEL_VALIDATION_AVAILABLE else None,
    'ModelValidator' if _MODEL_VALIDATION_AVAILABLE else None,
    'HyperparameterTuner' if _MODEL_VALIDATION_AVAILABLE else None,
    'ModelComparator' if _MODEL_VALIDATION_AVAILABLE else None,
    'ValidationVisualizer' if _MODEL_VALIDATION_AVAILABLE else None,
    'ValidationResult' if _MODEL_VALIDATION_AVAILABLE else None,
    'TuningResult' if _MODEL_VALIDATION_AVAILABLE else None,
    'ModelComparisonResult' if _MODEL_VALIDATION_AVAILABLE else None,
    'LearningCurveResult' if _MODEL_VALIDATION_AVAILABLE else None,
    'ValidationStrategy' if _MODEL_VALIDATION_AVAILABLE else None,
    'TuningMethod' if _MODEL_VALIDATION_AVAILABLE else None,
    'ComparisonMetric' if _MODEL_VALIDATION_AVAILABLE else None,
    'get_validation_system_info' if _MODEL_VALIDATION_AVAILABLE else None,
    
    # Seasonal pattern analyzer classes (when available)
    'SeasonalPatternAnalyzer' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'SeasonalTrend' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'EventImpact' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'HolidayEffect' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'GlobalTrendCorrelation' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'SeasonalAnalysisResult' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'SeasonalComponent' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'EventType' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'HolidayType' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
    'TrendCorrelationType' if _SEASONAL_PATTERN_ANALYZER_AVAILABLE else None,
]

# Filter out None values from __all__
__all__ = [item for item in __all__ if item is not None]

# Package initialization log
logger.info(f"K-Pop Analytics Package v{__version__} initialized")
logger.info(f"Growth calculator available: {_GROWTH_CALCULATOR_AVAILABLE}")
logger.info(f"Ranking system available: {_RANKING_SYSTEM_AVAILABLE}")
logger.info(f"Alert system available: {_ALERT_SYSTEM_AVAILABLE}")
logger.info(f"Prediction models available: {_PREDICTION_MODELS_AVAILABLE}")
logger.info(f"Model validation available: {_MODEL_VALIDATION_AVAILABLE}")
logger.info(f"Seasonal pattern analyzer available: {_SEASONAL_PATTERN_ANALYZER_AVAILABLE}")
logger.info(f"Supported platforms: {len(SUPPORTED_PLATFORMS)}")
logger.info(f"Supported metrics: {len(SUPPORTED_METRICS)}")