"""
Growth Ranking System Module
============================

Backend module for K-Pop artist ranking and comparative analysis based on growth metrics,
engagement performance, and multi-dimensional scoring algorithms.

Features:
- Platform-specific artist ranking with growth-based scoring
- Multi-factor composite index calculation with weighted metrics
- Intra-company artist ranking and competitive analysis
- Debut cohort comparative rankings for generational analysis
- Statistical validation and normalization methods

Author: Backend Development Team
Date: 2025-09-08
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import statistics
from collections import defaultdict

# Import from existing analytics modules
from .growth_rate_calculator import (
    MetricDataPoint, 
    GrowthRateCalculator, 
    GrowthRateResult, 
    CalculationMethod,
    GrowthPeriod
)

# Configure module logger
logger = logging.getLogger(__name__)

class RankingCategory(Enum):
    """Categories for ranking analysis."""
    GROWTH_RATE = "growth_rate"
    ABSOLUTE_VALUE = "absolute_value"
    ENGAGEMENT_RATE = "engagement_rate"
    CONSISTENCY = "consistency"
    MOMENTUM = "momentum"
    COMPOSITE = "composite"

class RankingPeriod(Enum):
    """Time periods for ranking analysis."""
    WEEKLY = 7
    MONTHLY = 30
    QUARTERLY = 90
    YEARLY = 365
    ALL_TIME = 0

class DebutCohort(Enum):
    """Debut year cohorts for generational analysis."""
    FIRST_GEN = "1990-1999"      # 1st generation K-Pop
    SECOND_GEN = "2000-2009"     # 2nd generation K-Pop
    THIRD_GEN = "2010-2019"      # 3rd generation K-Pop
    FOURTH_GEN = "2020-2029"     # 4th generation K-Pop

@dataclass
class ArtistMetrics:
    """Comprehensive metrics for a single artist."""
    artist_id: int
    artist_name: str
    company_id: int
    company_name: str
    debut_year: int
    platform: str
    metric_type: str
    current_value: int
    data_points: List[MetricDataPoint]
    quality_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass 
class RankingResult:
    """Result structure for ranking analysis."""
    artist_id: int
    artist_name: str
    company_name: str
    rank: int
    score: float
    growth_rate: float
    current_value: int
    percentile: float
    category: RankingCategory
    platform: str
    metric_type: str
    calculation_details: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0

@dataclass
class CompositeIndex:
    """Multi-factor composite index for comprehensive ranking."""
    artist_id: int
    artist_name: str
    overall_score: float
    component_scores: Dict[str, float]
    weighted_factors: Dict[str, float]
    rank: int
    percentile: float
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


class GrowthRankingEngine:
    """
    Advanced ranking engine for K-Pop artist performance analysis.
    
    Provides comprehensive ranking capabilities across multiple dimensions including
    growth rates, absolute performance, engagement metrics, and composite indices.
    """
    
    def __init__(self, 
                 growth_calculator: Optional[GrowthRateCalculator] = None,
                 default_weights: Optional[Dict[str, float]] = None,
                 outlier_threshold: float = 2.5,
                 confidence_level: float = 0.95):
        """
        Initialize the ranking engine.
        
        Args:
            growth_calculator: Instance of GrowthRateCalculator for growth analysis
            default_weights: Default weights for composite index calculation
            outlier_threshold: Threshold for outlier detection in ranking
            confidence_level: Statistical confidence level for calculations
        """
        self.growth_calculator = growth_calculator or GrowthRateCalculator()
        self.outlier_threshold = outlier_threshold
        self.confidence_level = confidence_level
        
        # Default weights for composite index
        self.default_weights = default_weights or {
            'growth_rate': 0.35,       # Growth trajectory importance
            'absolute_value': 0.25,    # Current performance scale
            'consistency': 0.20,       # Performance stability
            'momentum': 0.15,          # Recent trend strength
            'engagement_rate': 0.05    # Interaction quality
        }
        
        logger.info(f"GrowthRankingEngine initialized with {len(self.default_weights)} ranking factors")
    
    def rank_by_platform(self,
                        artists_data: List[ArtistMetrics],
                        platform: str,
                        metric_type: str,
                        category: RankingCategory = RankingCategory.GROWTH_RATE,
                        period: RankingPeriod = RankingPeriod.MONTHLY,
                        min_data_points: int = 7,
                        include_percentiles: bool = True) -> List[RankingResult]:
        """
        Rank artists by platform-specific performance metrics.
        
        Args:
            artists_data: List of artist metrics to rank
            platform: Platform to analyze ('youtube', 'spotify', etc.)
            metric_type: Metric to rank by ('subscribers', 'followers', etc.)
            category: Ranking category (growth_rate, absolute_value, etc.)
            period: Time period for analysis
            min_data_points: Minimum data points required for ranking
            include_percentiles: Whether to calculate percentile rankings
            
        Returns:
            Sorted list of RankingResult objects
        """
        logger.info(f"Ranking {len(artists_data)} artists by {category.value} on {platform} ({metric_type})")
        
        # Filter artists by platform and metric type
        filtered_artists = [
            artist for artist in artists_data 
            if artist.platform == platform and 
               artist.metric_type == metric_type and
               len(artist.data_points) >= min_data_points
        ]
        
        if not filtered_artists:
            logger.warning(f"No artists found for {platform}/{metric_type} with sufficient data")
            return []
        
        ranking_results = []
        calculation_errors = 0
        
        for artist in filtered_artists:
            try:
                # Calculate the ranking score based on category
                score, growth_rate, details = self._calculate_ranking_score(
                    artist, category, period
                )
                
                if score is not None:
                    result = RankingResult(
                        artist_id=artist.artist_id,
                        artist_name=artist.artist_name,
                        company_name=artist.company_name,
                        rank=0,  # Will be set after sorting
                        score=score,
                        growth_rate=growth_rate or 0.0,
                        current_value=artist.current_value,
                        percentile=0.0,  # Will be calculated after ranking
                        category=category,
                        platform=platform,
                        metric_type=metric_type,
                        calculation_details=details,
                        confidence_score=artist.quality_score
                    )
                    ranking_results.append(result)
                else:
                    calculation_errors += 1
                    
            except Exception as e:
                logger.error(f"Error calculating ranking for artist {artist.artist_id}: {e}")
                calculation_errors += 1
        
        if calculation_errors > 0:
            logger.warning(f"{calculation_errors} artists failed ranking calculation")
        
        # Sort by score (descending)
        ranking_results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks and percentiles
        total_artists = len(ranking_results)
        for i, result in enumerate(ranking_results):
            result.rank = i + 1
            if include_percentiles and total_artists > 1:
                result.percentile = (1 - (i / (total_artists - 1))) * 100
        
        logger.info(f"Successfully ranked {len(ranking_results)} artists by {category.value}")
        return ranking_results
    
    def calculate_composite_index(self,
                                 artists_data: List[ArtistMetrics],
                                 platforms: List[str],
                                 weights: Optional[Dict[str, float]] = None,
                                 normalization_method: str = 'z_score',
                                 include_analysis: bool = True) -> List[CompositeIndex]:
        """
        Calculate multi-dimensional composite index for comprehensive artist ranking.
        
        Args:
            artists_data: List of artist metrics across multiple platforms
            platforms: Platforms to include in composite calculation
            weights: Custom weights for different factors (overrides defaults)
            normalization_method: Method for score normalization ('z_score', 'min_max', 'rank')
            include_analysis: Whether to include strength/weakness analysis
            
        Returns:
            Sorted list of CompositeIndex objects
        """
        weights = weights or self.default_weights
        logger.info(f"Calculating composite index for {len(artists_data)} artists across {len(platforms)} platforms")
        
        # Group artist data by artist_id
        artist_groups = defaultdict(list)
        for artist in artists_data:
            if artist.platform in platforms:
                artist_groups[artist.artist_id].append(artist)
        
        composite_results = []
        
        for artist_id, artist_metrics in artist_groups.items():
            if not artist_metrics:
                continue
                
            try:
                # Get basic artist info from first metric
                primary_metric = artist_metrics[0]
                artist_name = primary_metric.artist_name
                
                # Calculate component scores for each platform/metric combination
                component_scores = {}
                raw_scores = {}
                
                for metric in artist_metrics:
                    platform_key = f"{metric.platform}_{metric.metric_type}"
                    
                    # Calculate scores for each ranking factor
                    growth_score, growth_rate, _ = self._calculate_ranking_score(
                        metric, RankingCategory.GROWTH_RATE, RankingPeriod.MONTHLY
                    )
                    
                    absolute_score, _, _ = self._calculate_ranking_score(
                        metric, RankingCategory.ABSOLUTE_VALUE, RankingPeriod.MONTHLY
                    )
                    
                    consistency_score, _, _ = self._calculate_ranking_score(
                        metric, RankingCategory.CONSISTENCY, RankingPeriod.MONTHLY
                    )
                    
                    momentum_score, _, _ = self._calculate_ranking_score(
                        metric, RankingCategory.MOMENTUM, RankingPeriod.WEEKLY
                    )
                    
                    engagement_score = metric.quality_score  # Use quality score as engagement proxy
                    
                    # Store individual platform scores
                    raw_scores[platform_key] = {
                        'growth_rate': growth_score or 0.0,
                        'absolute_value': absolute_score or 0.0,
                        'consistency': consistency_score or 0.0,
                        'momentum': momentum_score or 0.0,
                        'engagement_rate': engagement_score
                    }
                
                if not raw_scores:
                    continue
                
                # Aggregate scores across platforms for each factor
                aggregated_scores = {}
                for factor in weights.keys():
                    factor_scores = [
                        platform_scores.get(factor, 0.0) 
                        for platform_scores in raw_scores.values()
                    ]
                    # Use weighted average based on data quality
                    quality_weights = [
                        next(m.quality_score for m in artist_metrics if f"{m.platform}_{m.metric_type}" == platform)
                        for platform in raw_scores.keys()
                    ]
                    if sum(quality_weights) > 0:
                        aggregated_scores[factor] = sum(
                            score * weight for score, weight in zip(factor_scores, quality_weights)
                        ) / sum(quality_weights)
                    else:
                        aggregated_scores[factor] = statistics.mean(factor_scores) if factor_scores else 0.0
                
                # Normalize scores if requested
                if normalization_method != 'raw':
                    aggregated_scores = self._normalize_scores(aggregated_scores, normalization_method)
                
                # Calculate weighted composite score
                composite_score = sum(
                    score * weights.get(factor, 0.0) 
                    for factor, score in aggregated_scores.items()
                )
                
                # Identify strengths and weaknesses
                strengths, weaknesses = [], []
                if include_analysis:
                    strengths, weaknesses = self._analyze_performance(aggregated_scores, weights)
                
                composite_index = CompositeIndex(
                    artist_id=artist_id,
                    artist_name=artist_name,
                    overall_score=composite_score,
                    component_scores=aggregated_scores,
                    weighted_factors=weights.copy(),
                    rank=0,  # Will be assigned after sorting
                    percentile=0.0,  # Will be calculated after ranking
                    strengths=strengths,
                    weaknesses=weaknesses
                )
                
                composite_results.append(composite_index)
                
            except Exception as e:
                logger.error(f"Error calculating composite index for artist {artist_id}: {e}")
        
        # Sort by overall score (descending)
        composite_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Assign ranks and percentiles
        total_artists = len(composite_results)
        for i, result in enumerate(composite_results):
            result.rank = i + 1
            if total_artists > 1:
                result.percentile = (1 - (i / (total_artists - 1))) * 100
        
        logger.info(f"Composite index calculated for {len(composite_results)} artists")
        return composite_results
    
    def rank_within_company(self,
                           artists_data: List[ArtistMetrics],
                           company_id: int,
                           platform: str,
                           metric_type: str,
                           category: RankingCategory = RankingCategory.COMPOSITE,
                           include_company_stats: bool = True) -> Dict[str, Any]:
        """
        Rank artists within a specific company for competitive analysis.
        
        Args:
            artists_data: List of artist metrics
            company_id: Company ID to analyze
            platform: Platform for analysis
            metric_type: Metric type for analysis  
            category: Ranking category
            include_company_stats: Whether to include company-wide statistics
            
        Returns:
            Dictionary with rankings and company analysis
        """
        logger.info(f"Ranking artists within company {company_id} on {platform}/{metric_type}")
        
        # Filter artists by company
        company_artists = [
            artist for artist in artists_data 
            if artist.company_id == company_id and
               artist.platform == platform and
               artist.metric_type == metric_type
        ]
        
        if not company_artists:
            logger.warning(f"No artists found for company {company_id}")
            return {'rankings': [], 'company_stats': {}}
        
        # Get company name from first artist
        company_name = company_artists[0].company_name
        
        # Calculate rankings within company
        if category == RankingCategory.COMPOSITE:
            # Use composite index for comprehensive ranking
            composite_rankings = self.calculate_composite_index(
                company_artists, 
                [platform],
                include_analysis=True
            )
            rankings = [
                RankingResult(
                    artist_id=comp.artist_id,
                    artist_name=comp.artist_name,
                    company_name=company_name,
                    rank=comp.rank,
                    score=comp.overall_score,
                    growth_rate=comp.component_scores.get('growth_rate', 0.0),
                    current_value=next(a.current_value for a in company_artists if a.artist_id == comp.artist_id),
                    percentile=comp.percentile,
                    category=category,
                    platform=platform,
                    metric_type=metric_type,
                    calculation_details={
                        'component_scores': comp.component_scores,
                        'strengths': comp.strengths,
                        'weaknesses': comp.weaknesses
                    },
                    confidence_score=statistics.mean([a.quality_score for a in company_artists if a.artist_id == comp.artist_id])
                )
                for comp in composite_rankings
            ]
        else:
            # Use single-category ranking
            rankings = self.rank_by_platform(
                company_artists, platform, metric_type, category
            )
        
        result = {'rankings': rankings}
        
        # Add company-wide statistics if requested
        if include_company_stats:
            company_stats = self._calculate_company_stats(company_artists, rankings)
            result['company_stats'] = company_stats
        
        logger.info(f"Ranked {len(rankings)} artists within {company_name}")
        return result
    
    def rank_debut_cohort(self,
                         artists_data: List[ArtistMetrics],
                         cohort: DebutCohort,
                         platform: str,
                         metric_type: str,
                         category: RankingCategory = RankingCategory.GROWTH_RATE,
                         include_cohort_analysis: bool = True) -> Dict[str, Any]:
        """
        Rank artists within their debut cohort for generational analysis.
        
        Args:
            artists_data: List of artist metrics
            cohort: Debut cohort to analyze
            platform: Platform for analysis
            metric_type: Metric type for analysis
            category: Ranking category
            include_cohort_analysis: Whether to include cohort-wide analysis
            
        Returns:
            Dictionary with rankings and cohort analysis
        """
        logger.info(f"Ranking artists in {cohort.value} cohort on {platform}/{metric_type}")
        
        # Parse cohort years
        start_year, end_year = map(int, cohort.value.split('-'))
        
        # Filter artists by debut cohort
        cohort_artists = [
            artist for artist in artists_data
            if start_year <= artist.debut_year <= end_year and
               artist.platform == platform and
               artist.metric_type == metric_type
        ]
        
        if not cohort_artists:
            logger.warning(f"No artists found for {cohort.value} cohort")
            return {'rankings': [], 'cohort_analysis': {}}
        
        # Calculate rankings within cohort
        rankings = self.rank_by_platform(
            cohort_artists, platform, metric_type, category
        )
        
        # Enhance rankings with debut year information
        for ranking in rankings:
            artist = next(a for a in cohort_artists if a.artist_id == ranking.artist_id)
            ranking.calculation_details['debut_year'] = artist.debut_year
            ranking.calculation_details['years_active'] = datetime.now().year - artist.debut_year
        
        result = {'rankings': rankings}
        
        # Add cohort-wide analysis if requested
        if include_cohort_analysis:
            cohort_analysis = self._calculate_cohort_analysis(cohort_artists, rankings, cohort)
            result['cohort_analysis'] = cohort_analysis
        
        logger.info(f"Ranked {len(rankings)} artists in {cohort.value} cohort")
        return result
    
    # =============================================================================
    # PRIVATE HELPER METHODS
    # =============================================================================
    
    def _calculate_ranking_score(self, 
                               artist: ArtistMetrics, 
                               category: RankingCategory, 
                               period: RankingPeriod) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Calculate ranking score for a specific category."""
        try:
            data_points = artist.data_points
            details = {'calculation_method': category.value, 'period_days': period.value}
            
            if category == RankingCategory.GROWTH_RATE:
                # Calculate growth rate score
                if period == RankingPeriod.WEEKLY:
                    result = self.growth_calculator.calculate_rolling_growth(
                        data_points, window_days=7
                    )
                elif period == RankingPeriod.MONTHLY:
                    result = self.growth_calculator.calculate_growth_rate(
                        data_points, CalculationMethod.ROLLING_AVERAGE, GrowthPeriod.MONTHLY
                    )
                elif period == RankingPeriod.YEARLY:
                    result = self.growth_calculator.calculate_yoy_growth(data_points)
                else:
                    result = self.growth_calculator.calculate_daily_growth(data_points)
                
                if result:
                    score = self._normalize_growth_rate(result.growth_rate)
                    details.update({
                        'raw_growth_rate': result.growth_rate,
                        'data_quality': result.data_quality_score,
                        'is_significant': result.is_significant
                    })
                    return score, result.growth_rate, details
                
            elif category == RankingCategory.ABSOLUTE_VALUE:
                # Score based on absolute metric value (logarithmic scale)
                score = math.log10(max(artist.current_value, 1)) * 10
                details['current_value'] = artist.current_value
                return score, None, details
                
            elif category == RankingCategory.CONSISTENCY:
                # Score based on data consistency (low coefficient of variation)
                values = [p.value for p in data_points]
                if len(values) > 1:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values)
                    cv = std_val / mean_val if mean_val > 0 else float('inf')
                    score = max(0, 100 - (cv * 100))  # Lower CV = higher consistency score
                    details.update({
                        'coefficient_variation': cv,
                        'mean_value': mean_val,
                        'std_deviation': std_val
                    })
                    return score, None, details
                    
            elif category == RankingCategory.MOMENTUM:
                # Score based on recent trend strength (last week vs previous week)
                if len(data_points) >= 14:  # Need at least 2 weeks of data
                    recent_week = [p.value for p in data_points[:7]]
                    previous_week = [p.value for p in data_points[7:14]]
                    
                    recent_avg = statistics.mean(recent_week)
                    previous_avg = statistics.mean(previous_week)
                    
                    momentum = ((recent_avg - previous_avg) / previous_avg) * 100 if previous_avg > 0 else 0
                    score = self._normalize_growth_rate(momentum)
                    details.update({
                        'recent_avg': recent_avg,
                        'previous_avg': previous_avg,
                        'momentum_rate': momentum
                    })
                    return score, momentum, details
                    
            elif category == RankingCategory.ENGAGEMENT_RATE:
                # Use quality score as engagement proxy
                score = artist.quality_score * 100
                details['quality_score'] = artist.quality_score
                return score, None, details
                
            return None, None, details
            
        except Exception as e:
            logger.error(f"Error calculating {category.value} score: {e}")
            return None, None, {'error': str(e)}
    
    def _normalize_growth_rate(self, growth_rate: float) -> float:
        """Normalize growth rate to 0-100 scale."""
        # Use sigmoid function to normalize growth rates
        normalized = 100 / (1 + math.exp(-growth_rate / 10))
        return min(100, max(0, normalized))
    
    def _normalize_scores(self, scores: Dict[str, float], method: str) -> Dict[str, float]:
        """Normalize scores using specified method."""
        if not scores:
            return scores
            
        values = list(scores.values())
        
        if method == 'z_score':
            if len(values) > 1:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                if std_val > 0:
                    return {
                        key: ((value - mean_val) / std_val) * 10 + 50  # Scale to ~0-100
                        for key, value in scores.items()
                    }
        
        elif method == 'min_max':
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                return {
                    key: ((value - min_val) / (max_val - min_val)) * 100
                    for key, value in scores.items()
                }
        
        elif method == 'rank':
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            total_items = len(sorted_items)
            return {
                key: ((total_items - rank) / (total_items - 1)) * 100 if total_items > 1 else 100
                for rank, (key, _) in enumerate(sorted_items)
            }
        
        return scores
    
    def _analyze_performance(self, scores: Dict[str, float], weights: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Analyze performance to identify strengths and weaknesses."""
        if not scores:
            return [], []
        
        mean_score = statistics.mean(scores.values())
        threshold = mean_score * 0.15  # 15% threshold
        
        strengths = []
        weaknesses = []
        
        for factor, score in scores.items():
            if score > mean_score + threshold:
                strengths.append(factor.replace('_', ' ').title())
            elif score < mean_score - threshold:
                weaknesses.append(factor.replace('_', ' ').title())
        
        return strengths[:3], weaknesses[:3]  # Top 3 of each
    
    def _calculate_company_stats(self, artists: List[ArtistMetrics], rankings: List[RankingResult]) -> Dict[str, Any]:
        """Calculate company-wide statistics."""
        if not artists or not rankings:
            return {}
        
        scores = [r.score for r in rankings]
        growth_rates = [r.growth_rate for r in rankings if r.growth_rate is not None]
        current_values = [a.current_value for a in artists]
        
        return {
            'total_artists': len(artists),
            'company_name': artists[0].company_name,
            'average_score': statistics.mean(scores),
            'median_score': statistics.median(scores),
            'score_std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
            'average_growth_rate': statistics.mean(growth_rates) if growth_rates else 0,
            'total_metric_value': sum(current_values),
            'top_performer': rankings[0].artist_name if rankings else None,
            'performance_spread': max(scores) - min(scores) if len(scores) > 1 else 0
        }
    
    def _calculate_cohort_analysis(self, artists: List[ArtistMetrics], rankings: List[RankingResult], cohort: DebutCohort) -> Dict[str, Any]:
        """Calculate cohort-wide analysis."""
        if not artists or not rankings:
            return {}
        
        debut_years = [a.debut_year for a in artists]
        years_active = [datetime.now().year - year for year in debut_years]
        scores = [r.score for r in rankings]
        
        # Find peak performance years
        year_performance = defaultdict(list)
        for artist in artists:
            years_since_debut = datetime.now().year - artist.debut_year
            artist_score = next((r.score for r in rankings if r.artist_id == artist.artist_id), 0)
            year_performance[years_since_debut].append(artist_score)
        
        peak_year = max(year_performance.keys(), 
                       key=lambda y: statistics.mean(year_performance[y])) if year_performance else 0
        
        return {
            'cohort_name': cohort.value,
            'total_artists': len(artists),
            'debut_year_range': f"{min(debut_years)}-{max(debut_years)}",
            'average_years_active': statistics.mean(years_active),
            'average_cohort_score': statistics.mean(scores),
            'cohort_leader': rankings[0].artist_name if rankings else None,
            'peak_performance_year': peak_year,
            'cohort_diversity': statistics.stdev(scores) if len(scores) > 1 else 0,
            'maturity_trend': 'positive' if any(ya > 5 for ya in years_active) else 'emerging'
        }


# Export main components
__all__ = [
    'GrowthRankingEngine',
    'RankingCategory',
    'RankingPeriod', 
    'DebutCohort',
    'ArtistMetrics',
    'RankingResult',
    'CompositeIndex'
]