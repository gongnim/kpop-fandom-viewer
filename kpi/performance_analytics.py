"""
Performance Analytics Module for K-POP Dashboard
===============================================

Advanced performance identification algorithms including:
- Best Performer Automatic Selection Logic
- Attention-Needed Group Detection Logic  
- 3-Level Alert Classification System
- Performance Metrics Calculation Engine
- Anomaly Detection and Pattern Recognition

Author: Backend Development Team
Date: 2025-09-09
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, namedtuple
import math

from ..database_postgresql import (
    get_db_connection, get_companies, get_artists_by_company,
    get_platform_metrics_for_artist, get_latest_platform_metrics
)
from .kpi_engine import KPIEngine, KPICategory

# Configure module logger
logger = logging.getLogger(__name__)

# ========================================
# Enums and Constants
# ========================================

class AlertLevel(Enum):
    """3-level alert classification system."""
    GREEN = "green"      # Good performance, no issues
    YELLOW = "yellow"    # Warning, needs monitoring
    RED = "red"          # Critical, immediate attention required

class PerformanceCategory(Enum):
    """Categories for performance analysis."""
    GROWTH = "growth"
    ENGAGEMENT = "engagement"
    REACH = "reach"
    CONSISTENCY = "consistency"
    MOMENTUM = "momentum"
    OVERALL = "overall"

class TrendDirection(Enum):
    """Trend direction classification."""
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

# Performance thresholds (configurable)
PERFORMANCE_THRESHOLDS = {
    "growth": {
        "excellent": 0.15,    # 15%+ growth
        "good": 0.05,         # 5%+ growth
        "poor": -0.05,        # -5% decline
        "critical": -0.15     # -15% decline
    },
    "engagement": {
        "excellent": 0.08,    # 8%+ engagement rate
        "good": 0.04,         # 4%+ engagement rate
        "poor": 0.02,         # 2% engagement rate
        "critical": 0.01      # 1% engagement rate
    },
    "consistency": {
        "excellent": 0.1,     # Low variance (10%)
        "good": 0.2,          # Medium variance (20%)
        "poor": 0.4,          # High variance (40%)
        "critical": 0.6       # Very high variance (60%)
    }
}

# ========================================
# Data Classes
# ========================================

@dataclass
class PerformanceScore:
    """Performance score with detailed breakdown."""
    entity_id: int
    entity_name: str
    entity_type: str  # artist, group, company
    category: PerformanceCategory
    score: float      # 0-100 normalized score
    raw_value: float  # Raw metric value
    percentile: float # Percentile ranking
    trend: TrendDirection
    alert_level: AlertLevel
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'category': self.category.value,
            'score': round(self.score, 2),
            'raw_value': self.raw_value,
            'percentile': round(self.percentile, 2),
            'trend': self.trend.value,
            'alert_level': self.alert_level.value,
            'details': self.details
        }

@dataclass 
class BestPerformer:
    """Best performer identification result."""
    entity_id: int
    entity_name: str
    entity_type: str
    performance_scores: List[PerformanceScore]
    overall_score: float
    rank: int
    key_strengths: List[str]
    improvement_areas: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'performance_scores': [score.to_dict() for score in self.performance_scores],
            'overall_score': round(self.overall_score, 2),
            'rank': self.rank,
            'key_strengths': self.key_strengths,
            'improvement_areas': self.improvement_areas
        }

@dataclass
class AttentionAlert:
    """Alert for entities requiring attention."""
    entity_id: int
    entity_name: str
    entity_type: str
    alert_level: AlertLevel
    issues: List[str]
    affected_metrics: List[str]
    severity_score: float  # 0-100 scale
    recommended_actions: List[str]
    timeline: str  # when issue started
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'alert_level': self.alert_level.value,
            'issues': self.issues,
            'affected_metrics': self.affected_metrics,
            'severity_score': round(self.severity_score, 2),
            'recommended_actions': self.recommended_actions,
            'timeline': self.timeline
        }

# ========================================
# Performance Analytics Engine
# ========================================

class PerformanceAnalytics:
    """
    Advanced performance analytics engine for K-POP dashboard.
    
    Provides sophisticated algorithms for:
    - Best performer identification
    - Attention-needed detection
    - Alert level classification
    - Trend analysis and forecasting
    """
    
    def __init__(self, kpi_engine: Optional[KPIEngine] = None):
        """
        Initialize Performance Analytics Engine.
        
        Args:
            kpi_engine: KPI calculation engine instance
        """
        self.kpi_engine = kpi_engine or KPIEngine()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def identify_best_performers(
        self,
        entity_type: str = "artist",
        time_period: int = 30,
        categories: Optional[List[PerformanceCategory]] = None,
        min_data_points: int = 5,
        top_n: int = 10
    ) -> List[BestPerformer]:
        """
        Identify best performing entities using multi-dimensional analysis.
        
        Args:
            entity_type: Type of entities to analyze (artist, group, company)
            time_period: Analysis period in days
            categories: Performance categories to evaluate
            min_data_points: Minimum data points required for analysis
            top_n: Number of top performers to return
            
        Returns:
            List of BestPerformer objects ranked by overall performance
        """
        try:
            if categories is None:
                categories = [
                    PerformanceCategory.GROWTH,
                    PerformanceCategory.ENGAGEMENT,
                    PerformanceCategory.REACH,
                    PerformanceCategory.CONSISTENCY
                ]
            
            # Get entities and their performance data
            entities = self._get_entities_for_analysis(entity_type, time_period, min_data_points)
            
            best_performers = []
            all_scores = defaultdict(list)
            
            # Calculate performance scores for each entity and category
            for entity in entities:
                entity_scores = []
                
                for category in categories:
                    score = self._calculate_performance_score(
                        entity, category, time_period
                    )
                    entity_scores.append(score)
                    all_scores[category].append(score)
                
                # Calculate overall weighted score
                overall_score = self._calculate_overall_score(entity_scores)
                
                # Identify key strengths and improvement areas
                key_strengths = self._identify_key_strengths(entity_scores)
                improvement_areas = self._identify_improvement_areas(entity_scores)
                
                best_performers.append(BestPerformer(
                    entity_id=entity['id'],
                    entity_name=entity['name'],
                    entity_type=entity_type,
                    performance_scores=entity_scores,
                    overall_score=overall_score,
                    rank=0,  # Will be set after sorting
                    key_strengths=key_strengths,
                    improvement_areas=improvement_areas
                ))
            
            # Add percentile rankings to scores
            self._add_percentile_rankings(all_scores)
            
            # Sort by overall score and assign ranks
            best_performers.sort(key=lambda x: x.overall_score, reverse=True)
            for i, performer in enumerate(best_performers[:top_n], 1):
                performer.rank = i
            
            self.logger.info(f"Identified {len(best_performers[:top_n])} best performers for {entity_type}")
            return best_performers[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error identifying best performers: {e}")
            raise
    
    def detect_attention_needed(
        self,
        entity_type: str = "artist", 
        time_period: int = 30,
        severity_threshold: float = 30.0,
        min_data_points: int = 5
    ) -> List[AttentionAlert]:
        """
        Detect entities requiring immediate attention using anomaly detection.
        
        Args:
            entity_type: Type of entities to analyze
            time_period: Analysis period in days
            severity_threshold: Minimum severity score for inclusion
            min_data_points: Minimum data points required for analysis
            
        Returns:
            List of AttentionAlert objects sorted by severity
        """
        try:
            entities = self._get_entities_for_analysis(entity_type, time_period, min_data_points)
            alerts = []
            
            for entity in entities:
                # Analyze each entity for potential issues
                issues = self._analyze_entity_issues(entity, time_period)
                
                if issues:
                    # Calculate severity score
                    severity_score = self._calculate_severity_score(issues)
                    
                    if severity_score >= severity_threshold:
                        # Determine alert level
                        alert_level = self._determine_alert_level(severity_score)
                        
                        # Extract issue details
                        issue_descriptions = [issue['description'] for issue in issues]
                        affected_metrics = list(set([metric for issue in issues for metric in issue['metrics']]))
                        
                        # Generate recommendations
                        recommendations = self._generate_recommendations(issues, entity)
                        
                        # Determine timeline
                        timeline = self._determine_issue_timeline(issues)
                        
                        alerts.append(AttentionAlert(
                            entity_id=entity['id'],
                            entity_name=entity['name'],
                            entity_type=entity_type,
                            alert_level=alert_level,
                            issues=issue_descriptions,
                            affected_metrics=affected_metrics,
                            severity_score=severity_score,
                            recommended_actions=recommendations,
                            timeline=timeline
                        ))
            
            # Sort by severity score (descending)
            alerts.sort(key=lambda x: x.severity_score, reverse=True)
            
            self.logger.info(f"Detected {len(alerts)} attention items for {entity_type}")
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error detecting attention needed: {e}")
            raise
    
    def classify_alert_level(
        self,
        performance_scores: List[PerformanceScore],
        trend_analysis: Dict[str, Any]
    ) -> AlertLevel:
        """
        Classify alert level based on performance scores and trends.
        
        Args:
            performance_scores: List of performance scores
            trend_analysis: Trend analysis results
            
        Returns:
            AlertLevel classification
        """
        try:
            # Calculate weighted risk score
            risk_factors = []
            
            # Analyze performance scores
            for score in performance_scores:
                if score.score < 30:  # Poor performance
                    risk_factors.append(0.8)
                elif score.score < 50:  # Below average
                    risk_factors.append(0.4)
                elif score.trend == TrendDirection.DECLINING:
                    risk_factors.append(0.6)
                elif score.trend == TrendDirection.VOLATILE:
                    risk_factors.append(0.3)
            
            # Analyze trends
            declining_trends = trend_analysis.get('declining_metrics', 0)
            if declining_trends > 3:
                risk_factors.append(0.9)
            elif declining_trends > 1:
                risk_factors.append(0.5)
            
            # Calculate overall risk score
            if risk_factors:
                risk_score = max(risk_factors) + (sum(risk_factors) / len(risk_factors)) * 0.3
            else:
                risk_score = 0
            
            # Classify alert level
            if risk_score >= 0.7:
                return AlertLevel.RED
            elif risk_score >= 0.4:
                return AlertLevel.YELLOW
            else:
                return AlertLevel.GREEN
                
        except Exception as e:
            self.logger.error(f"Error classifying alert level: {e}")
            return AlertLevel.YELLOW
    
    # ========================================
    # Private Helper Methods
    # ========================================
    
    def _get_entities_for_analysis(
        self, 
        entity_type: str, 
        time_period: int, 
        min_data_points: int
    ) -> List[Dict[str, Any]]:
        """Get entities with sufficient data for analysis."""
        try:
            entities = []
            current_date = datetime.now()
            start_date = current_date - timedelta(days=time_period)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                if entity_type == "artist":
                    query = """
                        SELECT 
                            a.artist_id as id,
                            a.stage_name as name,
                            COUNT(pm.metric_id) as data_points
                        FROM artists a
                        LEFT JOIN platform_metrics pm ON a.artist_id = pm.artist_id
                        WHERE pm.collected_at >= %s
                        GROUP BY a.artist_id, a.stage_name
                        HAVING COUNT(pm.metric_id) >= %s
                        ORDER BY a.stage_name
                    """
                elif entity_type == "group":
                    query = """
                        SELECT 
                            g.group_id as id,
                            g.group_name as name,
                            COUNT(pm.metric_id) as data_points
                        FROM groups g
                        JOIN artists a ON g.group_id = a.group_id
                        LEFT JOIN platform_metrics pm ON a.artist_id = pm.artist_id
                        WHERE pm.collected_at >= %s
                        GROUP BY g.group_id, g.group_name
                        HAVING COUNT(pm.metric_id) >= %s
                        ORDER BY g.group_name
                    """
                elif entity_type == "company":
                    query = """
                        SELECT 
                            c.company_id as id,
                            c.company_name as name,
                            COUNT(pm.metric_id) as data_points
                        FROM companies c
                        JOIN artists a ON c.company_id = a.company_id
                        LEFT JOIN platform_metrics pm ON a.artist_id = pm.artist_id
                        WHERE pm.collected_at >= %s
                        GROUP BY c.company_id, c.company_name
                        HAVING COUNT(pm.metric_id) >= %s
                        ORDER BY c.company_name
                    """
                
                cursor.execute(query, (start_date, min_data_points))
                results = cursor.fetchall()
                
                for row in results:
                    entities.append({
                        'id': row[0],
                        'name': row[1],
                        'data_points': row[2]
                    })
                
                return entities
                
        except Exception as e:
            self.logger.error(f"Error getting entities for analysis: {e}")
            return []
    
    def _calculate_performance_score(
        self,
        entity: Dict[str, Any],
        category: PerformanceCategory,
        time_period: int
    ) -> PerformanceScore:
        """Calculate performance score for a specific category."""
        try:
            current_date = datetime.now()
            start_date = current_date - timedelta(days=time_period)
            
            # Get metrics data for the entity
            metrics_data = self._get_entity_metrics_data(entity, start_date, current_date)
            
            if category == PerformanceCategory.GROWTH:
                score, raw_value = self._calculate_growth_score(metrics_data)
            elif category == PerformanceCategory.ENGAGEMENT:
                score, raw_value = self._calculate_engagement_score(metrics_data)
            elif category == PerformanceCategory.REACH:
                score, raw_value = self._calculate_reach_score(metrics_data)
            elif category == PerformanceCategory.CONSISTENCY:
                score, raw_value = self._calculate_consistency_score(metrics_data)
            else:
                score, raw_value = 50.0, 0.0
            
            # Determine trend
            trend = self._determine_trend(metrics_data, category)
            
            # Determine alert level based on score
            if score >= 70:
                alert_level = AlertLevel.GREEN
            elif score >= 40:
                alert_level = AlertLevel.YELLOW
            else:
                alert_level = AlertLevel.RED
            
            return PerformanceScore(
                entity_id=entity['id'],
                entity_name=entity['name'],
                entity_type="artist",  # Will be updated by caller
                category=category,
                score=score,
                raw_value=raw_value,
                percentile=0,  # Will be calculated later
                trend=trend,
                alert_level=alert_level,
                details={
                    'data_points': len(metrics_data),
                    'period_days': time_period
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {e}")
            return PerformanceScore(
                entity_id=entity['id'],
                entity_name=entity['name'],
                entity_type="artist",
                category=category,
                score=0,
                raw_value=0,
                percentile=0,
                trend=TrendDirection.STABLE,
                alert_level=AlertLevel.RED
            )
    
    def _get_entity_metrics_data(
        self,
        entity: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get metrics data for an entity within date range."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        platform,
                        metric_name,
                        metric_value,
                        collected_at
                    FROM platform_metrics
                    WHERE artist_id = %s
                        AND collected_at BETWEEN %s AND %s
                    ORDER BY collected_at DESC
                """
                
                cursor.execute(query, (entity['id'], start_date, end_date))
                results = cursor.fetchall()
                
                metrics_data = []
                for row in results:
                    metrics_data.append({
                        'platform': row[0],
                        'metric_name': row[1],
                        'metric_value': float(row[2]) if row[2] else 0.0,
                        'collected_at': row[3]
                    })
                
                return metrics_data
                
        except Exception as e:
            self.logger.error(f"Error getting entity metrics data: {e}")
            return []
    
    def _calculate_growth_score(self, metrics_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate growth performance score."""
        if len(metrics_data) < 2:
            return 50.0, 0.0
        
        try:
            # Group metrics by platform and metric name
            platform_metrics = defaultdict(list)
            
            for data in metrics_data:
                key = f"{data['platform']}_{data['metric_name']}"
                platform_metrics[key].append(data)
            
            growth_rates = []
            
            for key, values in platform_metrics.items():
                if len(values) >= 2:
                    # Sort by date
                    values.sort(key=lambda x: x['collected_at'])
                    
                    # Calculate growth rate
                    initial_value = values[0]['metric_value']
                    final_value = values[-1]['metric_value']
                    
                    if initial_value > 0:
                        growth_rate = (final_value - initial_value) / initial_value
                        growth_rates.append(growth_rate)
            
            if not growth_rates:
                return 50.0, 0.0
            
            # Calculate average growth rate
            avg_growth_rate = statistics.mean(growth_rates)
            
            # Convert to 0-100 score
            # Excellent growth (15%+) = 90-100 points
            # Good growth (5%+) = 70-89 points
            # Stable (±5%) = 40-69 points
            # Declining = 0-39 points
            
            if avg_growth_rate >= 0.15:
                score = 90 + min(10, (avg_growth_rate - 0.15) * 100)
            elif avg_growth_rate >= 0.05:
                score = 70 + (avg_growth_rate - 0.05) * 200
            elif avg_growth_rate >= -0.05:
                score = 40 + (avg_growth_rate + 0.05) * 300
            else:
                score = max(0, 40 + (avg_growth_rate + 0.05) * 800)
            
            return min(100, max(0, score)), avg_growth_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating growth score: {e}")
            return 50.0, 0.0
    
    def _calculate_engagement_score(self, metrics_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate engagement performance score."""
        try:
            engagement_metrics = []
            
            # Look for engagement-related metrics
            for data in metrics_data:
                metric_name = data['metric_name'].lower()
                if any(keyword in metric_name for keyword in ['likes', 'comments', 'shares', 'views']):
                    engagement_metrics.append(data['metric_value'])
            
            if not engagement_metrics:
                return 50.0, 0.0
            
            # Calculate average engagement
            avg_engagement = statistics.mean(engagement_metrics)
            
            # Normalize to 0-100 scale (this would need platform-specific calibration)
            # For now, using a logarithmic scale
            if avg_engagement > 0:
                score = min(100, 20 * math.log10(avg_engagement + 1))
            else:
                score = 0
            
            return score, avg_engagement
            
        except Exception as e:
            self.logger.error(f"Error calculating engagement score: {e}")
            return 50.0, 0.0
    
    def _calculate_reach_score(self, metrics_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate reach performance score."""
        try:
            reach_metrics = []
            
            # Look for reach-related metrics
            for data in metrics_data:
                metric_name = data['metric_name'].lower()
                if any(keyword in metric_name for keyword in ['subscribers', 'followers', 'listeners']):
                    reach_metrics.append(data['metric_value'])
            
            if not reach_metrics:
                return 50.0, 0.0
            
            # Calculate total reach
            total_reach = sum(reach_metrics)
            
            # Normalize to 0-100 scale (logarithmic)
            if total_reach > 0:
                score = min(100, 15 * math.log10(total_reach + 1))
            else:
                score = 0
            
            return score, total_reach
            
        except Exception as e:
            self.logger.error(f"Error calculating reach score: {e}")
            return 50.0, 0.0
    
    def _calculate_consistency_score(self, metrics_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate consistency performance score."""
        try:
            # Group metrics by type and calculate variance
            metric_groups = defaultdict(list)
            
            for data in metrics_data:
                key = f"{data['platform']}_{data['metric_name']}"
                metric_groups[key].append(data['metric_value'])
            
            consistency_scores = []
            
            for key, values in metric_groups.items():
                if len(values) >= 3:
                    # Calculate coefficient of variation (CV)
                    mean_val = statistics.mean(values)
                    if mean_val > 0:
                        stdev = statistics.stdev(values)
                        cv = stdev / mean_val
                        
                        # Convert to consistency score (lower CV = higher consistency)
                        consistency_score = max(0, 100 - (cv * 100))
                        consistency_scores.append(consistency_score)
            
            if not consistency_scores:
                return 50.0, 0.0
            
            avg_consistency = statistics.mean(consistency_scores)
            
            # Calculate coefficient of variation for the raw metric
            all_values = [data['metric_value'] for data in metrics_data]
            if len(all_values) >= 2:
                mean_val = statistics.mean(all_values)
                if mean_val > 0:
                    cv = statistics.stdev(all_values) / mean_val
                else:
                    cv = 1.0
            else:
                cv = 1.0
            
            return avg_consistency, cv
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {e}")
            return 50.0, 1.0
    
    def _determine_trend(
        self,
        metrics_data: List[Dict[str, Any]],
        category: PerformanceCategory
    ) -> TrendDirection:
        """Determine trend direction for metrics."""
        if len(metrics_data) < 3:
            return TrendDirection.STABLE
        
        try:
            # Sort by date
            sorted_data = sorted(metrics_data, key=lambda x: x['collected_at'])
            
            # Take recent values
            recent_values = [data['metric_value'] for data in sorted_data[-5:]]
            
            if len(recent_values) < 3:
                return TrendDirection.STABLE
            
            # Simple trend analysis using linear regression slope
            x_values = list(range(len(recent_values)))
            
            # Calculate slope
            n = len(recent_values)
            sum_x = sum(x_values)
            sum_y = sum(recent_values)
            sum_xy = sum(x * y for x, y in zip(x_values, recent_values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Calculate variance to detect volatility
            mean_val = statistics.mean(recent_values)
            if mean_val > 0:
                variance = statistics.variance(recent_values) / (mean_val * mean_val)
            else:
                variance = 0
            
            # Classify trend
            if variance > 0.25:  # High variance indicates volatility
                return TrendDirection.VOLATILE
            elif slope > mean_val * 0.02:  # Positive trend > 2% of mean
                return TrendDirection.RISING
            elif slope < -mean_val * 0.02:  # Negative trend > -2% of mean
                return TrendDirection.DECLINING
            else:
                return TrendDirection.STABLE
                
        except Exception as e:
            self.logger.error(f"Error determining trend: {e}")
            return TrendDirection.STABLE
    
    def _calculate_overall_score(self, performance_scores: List[PerformanceScore]) -> float:
        """Calculate weighted overall performance score."""
        if not performance_scores:
            return 0.0
        
        # Define weights for different categories
        category_weights = {
            PerformanceCategory.GROWTH: 0.3,
            PerformanceCategory.ENGAGEMENT: 0.25,
            PerformanceCategory.REACH: 0.25,
            PerformanceCategory.CONSISTENCY: 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score in performance_scores:
            weight = category_weights.get(score.category, 0.1)
            weighted_sum += score.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _identify_key_strengths(self, performance_scores: List[PerformanceScore]) -> List[str]:
        """Identify key strengths from performance scores."""
        strengths = []
        
        for score in performance_scores:
            if score.score >= 75:
                category_name = score.category.value.replace('_', ' ').title()
                strengths.append(f"Excellent {category_name} ({score.score:.1f})")
            elif score.score >= 60 and score.trend == TrendDirection.RISING:
                category_name = score.category.value.replace('_', ' ').title()
                strengths.append(f"Improving {category_name} (Rising trend)")
        
        return strengths
    
    def _identify_improvement_areas(self, performance_scores: List[PerformanceScore]) -> List[str]:
        """Identify areas needing improvement from performance scores."""
        improvements = []
        
        for score in performance_scores:
            if score.score < 40:
                category_name = score.category.value.replace('_', ' ').title()
                improvements.append(f"Critical: {category_name} ({score.score:.1f})")
            elif score.score < 60:
                category_name = score.category.value.replace('_', ' ').title()
                improvements.append(f"Needs Attention: {category_name}")
            elif score.trend == TrendDirection.DECLINING:
                category_name = score.category.value.replace('_', ' ').title()
                improvements.append(f"Declining Trend: {category_name}")
        
        return improvements
    
    def _add_percentile_rankings(self, all_scores: Dict[PerformanceCategory, List[PerformanceScore]]):
        """Add percentile rankings to performance scores."""
        for category, scores in all_scores.items():
            # Sort scores for percentile calculation
            sorted_scores = sorted([s.score for s in scores])
            
            for score in scores:
                # Calculate percentile
                if len(sorted_scores) > 1:
                    rank = sorted_scores.index(score.score)
                    percentile = (rank / (len(sorted_scores) - 1)) * 100
                else:
                    percentile = 50
                
                score.percentile = percentile
    
    def _analyze_entity_issues(
        self,
        entity: Dict[str, Any],
        time_period: int
    ) -> List[Dict[str, Any]]:
        """Analyze entity for potential issues."""
        issues = []
        current_date = datetime.now()
        start_date = current_date - timedelta(days=time_period)
        
        # Get metrics data
        metrics_data = self._get_entity_metrics_data(entity, start_date, current_date)
        
        if not metrics_data:
            issues.append({
                'type': 'no_data',
                'description': 'No recent data available',
                'severity': 0.8,
                'metrics': ['all'],
                'timeline': f"{time_period} days"
            })
            return issues
        
        # Check for declining trends
        declining_issues = self._check_declining_trends(metrics_data)
        issues.extend(declining_issues)
        
        # Check for stagnation
        stagnation_issues = self._check_stagnation(metrics_data)
        issues.extend(stagnation_issues)
        
        # Check for anomalies
        anomaly_issues = self._check_anomalies(metrics_data)
        issues.extend(anomaly_issues)
        
        return issues
    
    def _check_declining_trends(self, metrics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for declining trends in metrics."""
        issues = []
        
        # Group by metric type
        metric_groups = defaultdict(list)
        for data in metrics_data:
            key = f"{data['platform']}_{data['metric_name']}"
            metric_groups[key].append(data)
        
        for key, values in metric_groups.items():
            if len(values) >= 3:
                # Sort by date
                values.sort(key=lambda x: x['collected_at'])
                
                # Check for declining trend
                recent_values = [v['metric_value'] for v in values[-3:]]
                
                if all(recent_values[i] > recent_values[i+1] for i in range(len(recent_values)-1)):
                    # Consistent decline
                    decline_rate = (recent_values[0] - recent_values[-1]) / recent_values[0]
                    
                    if decline_rate > 0.1:  # More than 10% decline
                        issues.append({
                            'type': 'declining_trend',
                            'description': f"Declining {key} ({decline_rate:.1%} drop)",
                            'severity': min(0.9, decline_rate * 2),
                            'metrics': [key],
                            'timeline': 'Recent 3 data points'
                        })
        
        return issues
    
    def _check_stagnation(self, metrics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for stagnant growth patterns."""
        issues = []
        
        # Group by metric type
        metric_groups = defaultdict(list)
        for data in metrics_data:
            key = f"{data['platform']}_{data['metric_name']}"
            metric_groups[key].append(data)
        
        for key, values in metric_groups.items():
            if len(values) >= 5:
                # Sort by date
                values.sort(key=lambda x: x['collected_at'])
                
                # Check for stagnation (low variance)
                recent_values = [v['metric_value'] for v in values[-5:]]
                
                if len(set(recent_values)) <= 2 or statistics.variance(recent_values) < (statistics.mean(recent_values) * 0.01):
                    issues.append({
                        'type': 'stagnation',
                        'description': f"Stagnant {key} (no growth)",
                        'severity': 0.6,
                        'metrics': [key],
                        'timeline': 'Recent 5 data points'
                    })
        
        return issues
    
    def _check_anomalies(self, metrics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for anomalies in metrics."""
        issues = []
        
        # Simple anomaly detection using IQR method
        metric_groups = defaultdict(list)
        for data in metrics_data:
            key = f"{data['platform']}_{data['metric_name']}"
            metric_groups[key].append(data['metric_value'])
        
        for key, values in metric_groups.items():
            if len(values) >= 5:
                # Calculate IQR
                sorted_values = sorted(values)
                q1 = np.percentile(sorted_values, 25)
                q3 = np.percentile(sorted_values, 75)
                iqr = q3 - q1
                
                # Define outlier bounds
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Check for outliers
                outliers = [v for v in values if v < lower_bound or v > upper_bound]
                
                if outliers:
                    issues.append({
                        'type': 'anomaly',
                        'description': f"Unusual values detected in {key}",
                        'severity': min(0.7, len(outliers) / len(values)),
                        'metrics': [key],
                        'timeline': 'Recent data points'
                    })
        
        return issues
    
    def _calculate_severity_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate overall severity score for issues."""
        if not issues:
            return 0.0
        
        # Weight different issue types
        type_weights = {
            'no_data': 0.9,
            'declining_trend': 0.8,
            'stagnation': 0.6,
            'anomaly': 0.4
        }
        
        weighted_severity = 0.0
        total_weight = 0.0
        
        for issue in issues:
            weight = type_weights.get(issue['type'], 0.5)
            weighted_severity += issue['severity'] * weight
            total_weight += weight
        
        base_severity = weighted_severity / total_weight if total_weight > 0 else 0
        
        # Boost severity for multiple issues
        issue_count_multiplier = min(1.5, 1 + (len(issues) - 1) * 0.2)
        
        return min(100, base_severity * 100 * issue_count_multiplier)
    
    def _determine_alert_level(self, severity_score: float) -> AlertLevel:
        """Determine alert level based on severity score."""
        if severity_score >= 70:
            return AlertLevel.RED
        elif severity_score >= 40:
            return AlertLevel.YELLOW
        else:
            return AlertLevel.GREEN
    
    def _generate_recommendations(
        self,
        issues: List[Dict[str, Any]],
        entity: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on issues."""
        recommendations = []
        
        for issue in issues:
            if issue['type'] == 'declining_trend':
                recommendations.append("Investigate factors causing metric decline")
                recommendations.append("Consider marketing campaign boost")
                recommendations.append("Review content strategy and engagement tactics")
            
            elif issue['type'] == 'stagnation':
                recommendations.append("Implement growth initiatives")
                recommendations.append("Explore new content formats or platforms")
                recommendations.append("Analyze competitor strategies")
            
            elif issue['type'] == 'anomaly':
                recommendations.append("Investigate data quality issues")
                recommendations.append("Verify measurement accuracy")
                recommendations.append("Check for external factors affecting metrics")
            
            elif issue['type'] == 'no_data':
                recommendations.append("Ensure data collection systems are operational")
                recommendations.append("Verify API connections and access")
                recommendations.append("Review data pipeline health")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]  # Limit to top 5 recommendations
    
    def _determine_issue_timeline(self, issues: List[Dict[str, Any]]) -> str:
        """Determine when issues started based on analysis."""
        timelines = [issue['timeline'] for issue in issues]
        
        # Return the most recent/specific timeline
        if 'Recent 3 data points' in timelines:
            return 'Past 3-5 days'
        elif 'Recent 5 data points' in timelines:
            return 'Past 1-2 weeks'
        elif 'Recent data points' in timelines:
            return 'Recent period'
        else:
            return 'Timeline unclear'