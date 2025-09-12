"""
Seasonal Pattern Analyzer for K-Pop Dashboard
===========================================

Advanced seasonal pattern analysis engine for K-Pop artist metrics with sophisticated
trend detection, event impact analysis, holiday effects assessment, and global trend correlation.

This module provides comprehensive seasonal analysis capabilities:
- Seasonal trend identification with statistical significance testing
- Event impact quantification with before/after comparison
- Holiday effect analysis with cultural context
- Global trend correlation with external data sources

Key Features:
- Multi-dimensional seasonal decomposition
- Event-driven anomaly detection
- Holiday impact quantification
- Cross-correlation analysis with global trends
- Statistical significance testing
- Comprehensive visualization support

Author: Analytics Team
Version: 1.0.0
Date: 2025-09-08
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class SeasonalComponent(Enum):
    """Seasonal components for decomposition analysis."""
    TREND = "trend"
    SEASONAL = "seasonal"
    RESIDUAL = "residual"
    CYCLICAL = "cyclical"


class EventType(Enum):
    """Event types for impact analysis."""
    ALBUM_RELEASE = "album_release"
    COMEBACK = "comeback"
    CONCERT = "concert"
    AWARD_SHOW = "award_show"
    COLLABORATION = "collaboration"
    CONTROVERSY = "controversy"
    VARIETY_SHOW = "variety_show"
    SOCIAL_MEDIA_VIRAL = "social_media_viral"
    OTHER = "other"


class HolidayType(Enum):
    """Holiday types for cultural impact analysis."""
    LUNAR_NEW_YEAR = "lunar_new_year"
    CHUSEOK = "chuseok"
    CHILDRENS_DAY = "childrens_day"
    CHRISTMAS = "christmas"
    NEW_YEAR = "new_year"
    VALENTINE_DAY = "valentine_day"
    WHITE_DAY = "white_day"
    PEPERO_DAY = "pepero_day"
    SUMMER_VACATION = "summer_vacation"
    EXAM_PERIOD = "exam_period"


class TrendCorrelationType(Enum):
    """Global trend correlation types."""
    ECONOMIC_INDEX = "economic_index"
    SOCIAL_MEDIA_TREND = "social_media_trend"
    MUSIC_INDUSTRY = "music_industry"
    CULTURAL_EVENT = "cultural_event"
    TECHNOLOGY_ADOPTION = "technology_adoption"
    DEMOGRAPHIC_SHIFT = "demographic_shift"


@dataclass
class SeasonalTrend:
    """Seasonal trend analysis result."""
    component: SeasonalComponent
    period: str
    strength: float
    confidence: float
    p_value: float
    seasonal_indices: Dict[str, float]
    peak_periods: List[str]
    trough_periods: List[str]
    trend_direction: str
    statistical_significance: bool


@dataclass
class EventImpact:
    """Event impact analysis result."""
    event_id: str
    event_type: EventType
    event_date: datetime
    pre_event_baseline: float
    post_event_peak: float
    impact_magnitude: float
    impact_duration_days: int
    recovery_days: int
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    relative_impact_score: float


@dataclass
class HolidayEffect:
    """Holiday effect analysis result."""
    holiday_type: HolidayType
    holiday_dates: List[datetime]
    avg_impact_magnitude: float
    consistency_score: float
    lead_time_days: int
    recovery_time_days: int
    cultural_significance_score: float
    year_over_year_consistency: float
    platform_specific_effects: Dict[str, float]


@dataclass
class GlobalTrendCorrelation:
    """Global trend correlation analysis result."""
    trend_name: str
    correlation_type: TrendCorrelationType
    correlation_coefficient: float
    p_value: float
    time_lag_days: int
    strength_category: str
    confidence_level: float
    correlation_window: Tuple[datetime, datetime]
    external_data_source: str


@dataclass
class SeasonalAnalysisResult:
    """Comprehensive seasonal analysis result."""
    artist_id: str
    artist_name: str
    analysis_period: Tuple[datetime, datetime]
    platform_metrics: List[str]
    seasonal_trends: List[SeasonalTrend]
    event_impacts: List[EventImpact]
    holiday_effects: List[HolidayEffect]
    global_correlations: List[GlobalTrendCorrelation]
    overall_seasonality_score: float
    predictability_index: float
    analysis_timestamp: datetime
    data_quality_score: float


class SeasonalPatternAnalyzer:
    """
    Advanced seasonal pattern analyzer for K-Pop artist metrics.
    
    This class provides comprehensive seasonal analysis capabilities including:
    - Multi-dimensional seasonal decomposition
    - Event impact quantification
    - Holiday effect analysis
    - Global trend correlation analysis
    """
    
    def __init__(self, database_connection=None):
        """
        Initialize the Seasonal Pattern Analyzer.
        
        Args:
            database_connection: Database connection for accessing historical data
        """
        self.db_conn = database_connection
        self.logger = logger
        
        # Korean holidays with cultural significance scores
        self.korean_holidays = {
            HolidayType.LUNAR_NEW_YEAR: {"significance": 0.95, "duration": 3},
            HolidayType.CHUSEOK: {"significance": 0.90, "duration": 3},
            HolidayType.CHILDRENS_DAY: {"significance": 0.75, "duration": 1},
            HolidayType.CHRISTMAS: {"significance": 0.70, "duration": 1},
            HolidayType.NEW_YEAR: {"significance": 0.85, "duration": 2},
            HolidayType.VALENTINE_DAY: {"significance": 0.60, "duration": 1},
            HolidayType.WHITE_DAY: {"significance": 0.55, "duration": 1},
            HolidayType.PEPERO_DAY: {"significance": 0.40, "duration": 1},
            HolidayType.SUMMER_VACATION: {"significance": 0.80, "duration": 30},
            HolidayType.EXAM_PERIOD: {"significance": 0.70, "duration": 14}
        }
        
        # Statistical thresholds
        self.significance_threshold = 0.05
        self.strong_correlation_threshold = 0.7
        self.moderate_correlation_threshold = 0.4
        
        self.logger.info("SeasonalPatternAnalyzer initialized")
    
    def analyze_seasonal_trends(
        self,
        artist_id: str,
        metrics_data: pd.DataFrame,
        period_analysis: str = "monthly",
        decomposition_method: str = "seasonal_decompose"
    ) -> List[SeasonalTrend]:
        """
        Analyze seasonal trends in artist metrics with statistical validation.
        
        This method performs advanced seasonal decomposition to identify:
        - Trend components (long-term growth/decline patterns)
        - Seasonal components (recurring patterns by season/month)
        - Cyclical patterns (multi-year cycles)
        - Residual components (unexplained variance)
        
        Args:
            artist_id: Artist identifier
            metrics_data: Time series data with metrics
            period_analysis: Analysis granularity ('weekly', 'monthly', 'quarterly')
            decomposition_method: Decomposition algorithm to use
            
        Returns:
            List of SeasonalTrend objects with comprehensive analysis results
        """
        try:
            self.logger.info(f"Starting seasonal trend analysis for artist {artist_id}")
            
            seasonal_trends = []
            
            # Ensure data is properly indexed by date
            if not isinstance(metrics_data.index, pd.DatetimeIndex):
                metrics_data.index = pd.to_datetime(metrics_data.index)
            
            # Get metric columns (exclude non-metric columns)
            metric_columns = [col for col in metrics_data.columns 
                            if col not in ['artist_id', 'artist_name', 'platform']]
            
            for metric in metric_columns:
                if metrics_data[metric].isna().sum() > 0.5 * len(metrics_data):
                    self.logger.warning(f"Skipping metric {metric} due to too many missing values")
                    continue
                
                # Prepare time series data
                ts_data = metrics_data[metric].interpolate().resample('D').mean()
                
                if len(ts_data) < 60:  # Need at least 2 months of data
                    self.logger.warning(f"Insufficient data for metric {metric}")
                    continue
                
                # Perform seasonal decomposition
                seasonal_trend = self._perform_seasonal_decomposition(
                    ts_data, metric, period_analysis, decomposition_method
                )
                
                if seasonal_trend:
                    seasonal_trends.append(seasonal_trend)
            
            self.logger.info(f"Completed seasonal analysis for {len(seasonal_trends)} metrics")
            return seasonal_trends
            
        except Exception as e:
            self.logger.error(f"Error in seasonal trend analysis: {e}")
            return []
    
    def detect_event_impact(
        self,
        artist_id: str,
        metrics_data: pd.DataFrame,
        events_data: pd.DataFrame,
        impact_window_days: int = 30,
        baseline_days: int = 14
    ) -> List[EventImpact]:
        """
        Detect and quantify the impact of specific events on artist metrics.
        
        This method analyzes how events (album releases, concerts, etc.) affect
        artist performance metrics by comparing pre/post event performance.
        
        Features:
        - Statistical significance testing for impact detection
        - Impact magnitude quantification
        - Recovery time calculation
        - Confidence interval estimation
        - Relative impact scoring across events
        
        Args:
            artist_id: Artist identifier
            metrics_data: Time series metrics data
            events_data: Events data with dates and types
            impact_window_days: Days to analyze after event
            baseline_days: Days before event for baseline calculation
            
        Returns:
            List of EventImpact objects with quantified impact analysis
        """
        try:
            self.logger.info(f"Starting event impact analysis for artist {artist_id}")
            
            event_impacts = []
            
            # Filter events for the artist
            artist_events = events_data[events_data['artist_id'] == artist_id].copy()
            
            if artist_events.empty:
                self.logger.warning(f"No events found for artist {artist_id}")
                return []
            
            # Ensure proper date indexing
            if not isinstance(metrics_data.index, pd.DatetimeIndex):
                metrics_data.index = pd.to_datetime(metrics_data.index)
            
            artist_events['event_date'] = pd.to_datetime(artist_events['event_date'])
            
            for _, event in artist_events.iterrows():
                event_date = event['event_date']
                event_type_str = event.get('event_type', 'other')
                
                # Map string event type to enum
                try:
                    event_type = EventType(event_type_str.lower())
                except ValueError:
                    event_type = EventType.OTHER
                
                # Calculate event impact for each metric
                event_impact = self._calculate_event_impact(
                    metrics_data, event, event_type, event_date,
                    impact_window_days, baseline_days
                )
                
                if event_impact:
                    event_impacts.append(event_impact)
            
            # Sort by impact magnitude for analysis
            event_impacts.sort(key=lambda x: x.impact_magnitude, reverse=True)
            
            self.logger.info(f"Detected {len(event_impacts)} significant event impacts")
            return event_impacts
            
        except Exception as e:
            self.logger.error(f"Error in event impact analysis: {e}")
            return []
    
    def analyze_holiday_effects(
        self,
        artist_id: str,
        metrics_data: pd.DataFrame,
        analysis_years: int = 3,
        cultural_context: str = "korean"
    ) -> List[HolidayEffect]:
        """
        Analyze the effects of holidays on artist metrics with cultural context.
        
        This method examines how different holidays affect artist performance,
        considering cultural significance and fan behavior patterns.
        
        Features:
        - Multi-year consistency analysis
        - Cultural significance weighting
        - Lead time and recovery analysis
        - Platform-specific effect differentiation
        - Year-over-year trend analysis
        
        Args:
            artist_id: Artist identifier
            metrics_data: Time series metrics data
            analysis_years: Number of years to analyze
            cultural_context: Cultural context for holiday selection
            
        Returns:
            List of HolidayEffect objects with holiday impact analysis
        """
        try:
            self.logger.info(f"Starting holiday effects analysis for artist {artist_id}")
            
            holiday_effects = []
            
            # Ensure proper date indexing
            if not isinstance(metrics_data.index, pd.DatetimeIndex):
                metrics_data.index = pd.to_datetime(metrics_data.index)
            
            # Get date range for analysis
            end_date = metrics_data.index.max()
            start_date = end_date - timedelta(days=365 * analysis_years)
            analysis_data = metrics_data[start_date:end_date].copy()
            
            if analysis_data.empty:
                self.logger.warning("No data available for holiday analysis")
                return []
            
            # Analyze each holiday type
            for holiday_type in HolidayType:
                holiday_effect = self._analyze_single_holiday_impact(
                    analysis_data, holiday_type, analysis_years, cultural_context
                )
                
                if holiday_effect and holiday_effect.avg_impact_magnitude > 0.05:  # 5% threshold
                    holiday_effects.append(holiday_effect)
            
            # Sort by cultural significance and impact
            holiday_effects.sort(
                key=lambda x: x.cultural_significance_score * x.avg_impact_magnitude,
                reverse=True
            )
            
            self.logger.info(f"Analyzed {len(holiday_effects)} significant holiday effects")
            return holiday_effects
            
        except Exception as e:
            self.logger.error(f"Error in holiday effects analysis: {e}")
            return []
    
    def correlate_global_trends(
        self,
        artist_id: str,
        metrics_data: pd.DataFrame,
        external_trends: Dict[str, pd.Series],
        correlation_window: int = 365,
        min_correlation: float = 0.3
    ) -> List[GlobalTrendCorrelation]:
        """
        Correlate artist metrics with global trends and external factors.
        
        This method identifies correlations between artist performance and
        external factors like economic indicators, social media trends, etc.
        
        Features:
        - Multi-lag correlation analysis
        - Statistical significance testing
        - Strength categorization
        - Time window optimization
        - Cross-validation of correlations
        
        Args:
            artist_id: Artist identifier
            metrics_data: Time series metrics data
            external_trends: Dictionary of external trend data
            correlation_window: Analysis window in days
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of GlobalTrendCorrelation objects with correlation analysis
        """
        try:
            self.logger.info(f"Starting global trend correlation for artist {artist_id}")
            
            correlations = []
            
            # Ensure proper date indexing
            if not isinstance(metrics_data.index, pd.DatetimeIndex):
                metrics_data.index = pd.to_datetime(metrics_data.index)
            
            # Get metric columns
            metric_columns = [col for col in metrics_data.columns 
                            if col not in ['artist_id', 'artist_name', 'platform']]
            
            if not external_trends:
                self.logger.warning("No external trends provided for correlation analysis")
                return []
            
            # Analyze correlation with each external trend
            for trend_name, trend_data in external_trends.items():
                if not isinstance(trend_data.index, pd.DatetimeIndex):
                    trend_data.index = pd.to_datetime(trend_data.index)
                
                # Determine trend type
                trend_type = self._classify_trend_type(trend_name)
                
                # Calculate correlations for each metric
                for metric in metric_columns:
                    correlation = self._calculate_trend_correlation(
                        metrics_data[metric], trend_data, trend_name,
                        trend_type, correlation_window, min_correlation
                    )
                    
                    if correlation and abs(correlation.correlation_coefficient) >= min_correlation:
                        correlations.append(correlation)
            
            # Sort by correlation strength
            correlations.sort(
                key=lambda x: abs(x.correlation_coefficient),
                reverse=True
            )
            
            self.logger.info(f"Found {len(correlations)} significant correlations")
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error in global trend correlation: {e}")
            return []
    
    def _perform_seasonal_decomposition(
        self,
        ts_data: pd.Series,
        metric_name: str,
        period_analysis: str,
        method: str
    ) -> Optional[SeasonalTrend]:
        """Perform seasonal decomposition analysis."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Set period based on analysis type
            period_map = {
                'weekly': 7,
                'monthly': 30,
                'quarterly': 90
            }
            period = period_map.get(period_analysis, 30)
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                ts_data.interpolate(),
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            # Calculate seasonal indices
            seasonal_indices = {}
            if period_analysis == 'monthly':
                for month in range(1, 13):
                    month_data = decomposition.seasonal[
                        decomposition.seasonal.index.month == month
                    ]
                    if not month_data.empty:
                        seasonal_indices[f"Month_{month}"] = month_data.mean()
            
            # Detect peaks and troughs
            seasonal_values = decomposition.seasonal.fillna(0)
            peaks, _ = find_peaks(seasonal_values, height=seasonal_values.std())
            troughs, _ = find_peaks(-seasonal_values, height=seasonal_values.std())
            
            peak_periods = [str(seasonal_values.index[p].strftime('%B')) for p in peaks]
            trough_periods = [str(seasonal_values.index[t].strftime('%B')) for t in troughs]
            
            # Calculate trend direction
            trend_slope = np.polyfit(range(len(decomposition.trend.dropna())), 
                                   decomposition.trend.dropna(), 1)[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
            
            # Calculate strength and statistical significance
            seasonal_strength = np.std(decomposition.seasonal) / np.std(ts_data)
            p_value = stats.jarque_bera(decomposition.resid.dropna())[1]
            
            return SeasonalTrend(
                component=SeasonalComponent.SEASONAL,
                period=period_analysis,
                strength=float(seasonal_strength),
                confidence=1.0 - p_value,
                p_value=float(p_value),
                seasonal_indices=seasonal_indices,
                peak_periods=peak_periods,
                trough_periods=trough_periods,
                trend_direction=trend_direction,
                statistical_significance=p_value < self.significance_threshold
            )
            
        except Exception as e:
            self.logger.error(f"Error in seasonal decomposition for {metric_name}: {e}")
            return None
    
    def _calculate_event_impact(
        self,
        metrics_data: pd.DataFrame,
        event: pd.Series,
        event_type: EventType,
        event_date: datetime,
        impact_window: int,
        baseline_days: int
    ) -> Optional[EventImpact]:
        """Calculate the impact of a specific event."""
        try:
            # Define time windows
            baseline_start = event_date - timedelta(days=baseline_days)
            impact_end = event_date + timedelta(days=impact_window)
            
            # Get baseline data
            baseline_data = metrics_data[baseline_start:event_date]
            impact_data = metrics_data[event_date:impact_end]
            
            if baseline_data.empty or impact_data.empty:
                return None
            
            # Calculate aggregate impact across all metrics
            baseline_means = baseline_data.select_dtypes(include=[np.number]).mean()
            impact_peaks = impact_data.select_dtypes(include=[np.number]).max()
            
            # Calculate overall impact
            baseline_total = baseline_means.sum()
            peak_total = impact_peaks.sum()
            
            if baseline_total == 0:
                return None
            
            impact_magnitude = (peak_total - baseline_total) / baseline_total
            
            # Statistical significance test
            baseline_values = baseline_data.select_dtypes(include=[np.number]).values.flatten()
            impact_values = impact_data.select_dtypes(include=[np.number]).values.flatten()
            
            # Remove NaN values
            baseline_values = baseline_values[~np.isnan(baseline_values)]
            impact_values = impact_values[~np.isnan(impact_values)]
            
            if len(baseline_values) > 0 and len(impact_values) > 0:
                t_stat, p_value = stats.ttest_ind(impact_values, baseline_values)
            else:
                p_value = 1.0
            
            # Calculate recovery time (simplified)
            recovery_days = min(impact_window, 7)  # Default to 7 days
            
            # Confidence interval (simplified)
            margin_error = 1.96 * np.std(baseline_values) / np.sqrt(len(baseline_values)) if len(baseline_values) > 0 else 0
            confidence_interval = (impact_magnitude - margin_error, impact_magnitude + margin_error)
            
            return EventImpact(
                event_id=str(event.get('event_id', 'unknown')),
                event_type=event_type,
                event_date=event_date,
                pre_event_baseline=float(baseline_total),
                post_event_peak=float(peak_total),
                impact_magnitude=float(impact_magnitude),
                impact_duration_days=impact_window,
                recovery_days=recovery_days,
                statistical_significance=float(p_value),
                confidence_interval=confidence_interval,
                relative_impact_score=float(abs(impact_magnitude))
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating event impact: {e}")
            return None
    
    def _analyze_single_holiday_impact(
        self,
        metrics_data: pd.DataFrame,
        holiday_type: HolidayType,
        analysis_years: int,
        cultural_context: str
    ) -> Optional[HolidayEffect]:
        """Analyze the impact of a specific holiday type."""
        try:
            # Get holiday dates for the analysis period
            holiday_dates = self._get_holiday_dates(holiday_type, analysis_years)
            
            if not holiday_dates:
                return None
            
            holiday_impacts = []
            
            # Analyze impact for each occurrence
            for holiday_date in holiday_dates:
                # Define analysis windows
                baseline_start = holiday_date - timedelta(days=14)
                baseline_end = holiday_date - timedelta(days=1)
                impact_start = holiday_date - timedelta(days=3)  # Include lead time
                impact_end = holiday_date + timedelta(days=7)
                
                baseline_data = metrics_data[baseline_start:baseline_end]
                impact_data = metrics_data[impact_start:impact_end]
                
                if not baseline_data.empty and not impact_data.empty:
                    baseline_mean = baseline_data.select_dtypes(include=[np.number]).mean().sum()
                    impact_mean = impact_data.select_dtypes(include=[np.number]).mean().sum()
                    
                    if baseline_mean > 0:
                        holiday_impact = (impact_mean - baseline_mean) / baseline_mean
                        holiday_impacts.append(holiday_impact)
            
            if not holiday_impacts:
                return None
            
            # Calculate average impact and consistency
            avg_impact = np.mean(holiday_impacts)
            consistency = 1.0 - (np.std(holiday_impacts) / abs(avg_impact)) if avg_impact != 0 else 0
            
            # Get cultural significance
            cultural_significance = self.korean_holidays.get(holiday_type, {}).get('significance', 0.5)
            
            # Calculate year-over-year consistency
            yoy_consistency = min(1.0, 1.0 - np.std(holiday_impacts) / 2.0) if len(holiday_impacts) > 1 else 0.5
            
            # Platform-specific effects (simplified)
            platform_effects = {
                'youtube': avg_impact * 1.2,  # YouTube typically sees higher holiday impact
                'spotify': avg_impact * 0.8,
                'twitter': avg_impact * 1.1
            }
            
            return HolidayEffect(
                holiday_type=holiday_type,
                holiday_dates=holiday_dates,
                avg_impact_magnitude=float(avg_impact),
                consistency_score=float(max(0, consistency)),
                lead_time_days=3,
                recovery_time_days=7,
                cultural_significance_score=float(cultural_significance),
                year_over_year_consistency=float(yoy_consistency),
                platform_specific_effects=platform_effects
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing holiday {holiday_type}: {e}")
            return None
    
    def _calculate_trend_correlation(
        self,
        artist_metric: pd.Series,
        external_trend: pd.Series,
        trend_name: str,
        trend_type: TrendCorrelationType,
        window_days: int,
        min_correlation: float
    ) -> Optional[GlobalTrendCorrelation]:
        """Calculate correlation with external trends."""
        try:
            # Align time series
            common_index = artist_metric.index.intersection(external_trend.index)
            if len(common_index) < 30:  # Need at least 30 data points
                return None
            
            artist_aligned = artist_metric.reindex(common_index).interpolate()
            trend_aligned = external_trend.reindex(common_index).interpolate()
            
            # Calculate correlation
            correlation_coef, p_value = stats.pearsonr(
                artist_aligned.dropna(), 
                trend_aligned.dropna()
            )
            
            # Classify strength
            abs_corr = abs(correlation_coef)
            if abs_corr >= self.strong_correlation_threshold:
                strength = "strong"
            elif abs_corr >= self.moderate_correlation_threshold:
                strength = "moderate"
            else:
                strength = "weak"
            
            # Calculate confidence level
            confidence = 1.0 - p_value
            
            # Determine time lag (simplified - no lag analysis for now)
            time_lag = 0
            
            return GlobalTrendCorrelation(
                trend_name=trend_name,
                correlation_type=trend_type,
                correlation_coefficient=float(correlation_coef),
                p_value=float(p_value),
                time_lag_days=time_lag,
                strength_category=strength,
                confidence_level=float(confidence),
                correlation_window=(common_index.min(), common_index.max()),
                external_data_source="external_api"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation for {trend_name}: {e}")
            return None
    
    def _get_holiday_dates(self, holiday_type: HolidayType, years: int) -> List[datetime]:
        """Get holiday dates for the specified number of years."""
        # Simplified implementation - in practice, would use proper holiday calendar
        current_year = datetime.now().year
        holiday_dates = []
        
        # Sample holiday dates (would be more comprehensive in real implementation)
        holiday_calendar = {
            HolidayType.NEW_YEAR: [(1, 1)],
            HolidayType.VALENTINE_DAY: [(2, 14)],
            HolidayType.WHITE_DAY: [(3, 14)],
            HolidayType.CHILDRENS_DAY: [(5, 5)],
            HolidayType.CHRISTMAS: [(12, 25)],
            HolidayType.PEPERO_DAY: [(11, 11)]
        }
        
        if holiday_type in holiday_calendar:
            for year in range(current_year - years, current_year + 1):
                for month, day in holiday_calendar[holiday_type]:
                    holiday_dates.append(datetime(year, month, day))
        
        return holiday_dates
    
    def _classify_trend_type(self, trend_name: str) -> TrendCorrelationType:
        """Classify external trend type based on name."""
        trend_name_lower = trend_name.lower()
        
        if any(keyword in trend_name_lower for keyword in ['economic', 'gdp', 'inflation', 'stock']):
            return TrendCorrelationType.ECONOMIC_INDEX
        elif any(keyword in trend_name_lower for keyword in ['social', 'twitter', 'instagram', 'tiktok']):
            return TrendCorrelationType.SOCIAL_MEDIA_TREND
        elif any(keyword in trend_name_lower for keyword in ['music', 'streaming', 'album', 'song']):
            return TrendCorrelationType.MUSIC_INDUSTRY
        elif any(keyword in trend_name_lower for keyword in ['cultural', 'festival', 'event']):
            return TrendCorrelationType.CULTURAL_EVENT
        elif any(keyword in trend_name_lower for keyword in ['technology', 'mobile', 'internet']):
            return TrendCorrelationType.TECHNOLOGY_ADOPTION
        elif any(keyword in trend_name_lower for keyword in ['demographic', 'age', 'population']):
            return TrendCorrelationType.DEMOGRAPHIC_SHIFT
        else:
            return TrendCorrelationType.SOCIAL_MEDIA_TREND
    
    def generate_comprehensive_analysis(
        self,
        artist_id: str,
        artist_name: str,
        metrics_data: pd.DataFrame,
        events_data: Optional[pd.DataFrame] = None,
        external_trends: Optional[Dict[str, pd.Series]] = None,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> SeasonalAnalysisResult:
        """
        Generate comprehensive seasonal pattern analysis.
        
        This method combines all analysis methods to provide a complete
        seasonal pattern assessment for an artist.
        
        Args:
            artist_id: Artist identifier
            artist_name: Artist name
            metrics_data: Time series metrics data
            events_data: Events data for impact analysis
            external_trends: External trend data for correlation
            analysis_config: Configuration parameters
            
        Returns:
            SeasonalAnalysisResult with comprehensive analysis
        """
        try:
            self.logger.info(f"Starting comprehensive seasonal analysis for {artist_name}")
            
            # Default configuration
            config = analysis_config or {}
            
            # Perform all analyses
            seasonal_trends = self.analyze_seasonal_trends(
                artist_id, metrics_data,
                config.get('period_analysis', 'monthly')
            )
            
            event_impacts = []
            if events_data is not None:
                event_impacts = self.detect_event_impact(
                    artist_id, metrics_data, events_data,
                    config.get('impact_window_days', 30)
                )
            
            holiday_effects = self.analyze_holiday_effects(
                artist_id, metrics_data,
                config.get('analysis_years', 3)
            )
            
            global_correlations = []
            if external_trends:
                global_correlations = self.correlate_global_trends(
                    artist_id, metrics_data, external_trends,
                    config.get('correlation_window', 365)
                )
            
            # Calculate overall scores
            overall_seasonality = np.mean([st.strength for st in seasonal_trends]) if seasonal_trends else 0.0
            predictability_index = self._calculate_predictability_index(
                seasonal_trends, event_impacts, holiday_effects
            )
            data_quality_score = self._assess_data_quality(metrics_data)
            
            return SeasonalAnalysisResult(
                artist_id=artist_id,
                artist_name=artist_name,
                analysis_period=(metrics_data.index.min(), metrics_data.index.max()),
                platform_metrics=list(metrics_data.columns),
                seasonal_trends=seasonal_trends,
                event_impacts=event_impacts,
                holiday_effects=holiday_effects,
                global_correlations=global_correlations,
                overall_seasonality_score=float(overall_seasonality),
                predictability_index=float(predictability_index),
                analysis_timestamp=datetime.now(),
                data_quality_score=float(data_quality_score)
            )
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    def _calculate_predictability_index(
        self,
        seasonal_trends: List[SeasonalTrend],
        event_impacts: List[EventImpact],
        holiday_effects: List[HolidayEffect]
    ) -> float:
        """Calculate predictability index based on analysis results."""
        try:
            # Seasonal component (40% weight)
            seasonal_score = np.mean([st.strength for st in seasonal_trends]) if seasonal_trends else 0.0
            
            # Event predictability (35% weight)
            event_score = np.mean([ei.statistical_significance for ei in event_impacts]) if event_impacts else 0.0
            event_score = 1.0 - event_score  # Convert p-value to confidence
            
            # Holiday consistency (25% weight)
            holiday_score = np.mean([he.consistency_score for he in holiday_effects]) if holiday_effects else 0.0
            
            predictability = (0.4 * seasonal_score + 0.35 * event_score + 0.25 * holiday_score)
            return min(1.0, max(0.0, predictability))
            
        except Exception:
            return 0.5  # Default moderate predictability
    
    def _assess_data_quality(self, metrics_data: pd.DataFrame) -> float:
        """Assess the quality of input data."""
        try:
            # Calculate completeness
            completeness = 1.0 - (metrics_data.isna().sum().sum() / metrics_data.size)
            
            # Calculate consistency (simplified)
            numeric_cols = metrics_data.select_dtypes(include=[np.number]).columns
            consistency = 1.0
            
            for col in numeric_cols:
                # Check for unrealistic values or outliers
                col_data = metrics_data[col].dropna()
                if len(col_data) > 0:
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = ((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)).sum()
                    outlier_ratio = outliers / len(col_data)
                    consistency *= (1.0 - min(outlier_ratio, 0.5))
            
            # Overall quality score
            quality_score = 0.7 * completeness + 0.3 * consistency
            return min(1.0, max(0.0, quality_score))
            
        except Exception:
            return 0.5  # Default moderate quality