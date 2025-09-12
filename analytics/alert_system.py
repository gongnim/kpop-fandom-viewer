"""
K-Pop Dashboard Analytics - Alert System
=======================================

Intelligent alert engine for detecting significant changes, anomalies, and trends
in K-Pop artist performance metrics with automated alert generation and management.

This module provides:
- Rapid growth detection with configurable thresholds
- Growth decline monitoring and early warning systems  
- Statistical anomaly detection using multiple methods
- Comprehensive alert generation with severity classification
- Alert lifecycle management and resolution tracking
- Integration with ranking and growth rate calculation systems

Author: Backend Development Team
Version: 1.0.0
Date: 2025-09-08
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
import math
from collections import defaultdict, deque

# Import from existing analytics modules
from .growth_rate_calculator import (
    MetricDataPoint, 
    GrowthRateCalculator, 
    GrowthRateResult,
    CalculationMethod,
    GrowthPeriod
)
from .ranking_system import ArtistMetrics, RankingCategory

# Configure logging
logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts that can be generated."""
    RAPID_GROWTH = "rapid_growth"
    GROWTH_DECLINE = "growth_decline"
    ANOMALY_DETECTED = "anomaly_detected"
    MILESTONE_REACHED = "milestone_reached"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNUSUAL_ACTIVITY = "unusual_activity"
    DATA_QUALITY_ISSUE = "data_quality_issue"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    CRITICAL = "critical"      # Immediate attention required
    HIGH = "high"             # Important, should be addressed soon
    MEDIUM = "medium"         # Moderate importance
    LOW = "low"              # Informational
    INFO = "info"            # General information


class AlertLevel(Enum):
    """3-tier alert level system (green/yellow/red)."""
    GREEN = "green"      # Normal/Good status - informational alerts
    YELLOW = "yellow"    # Warning/Caution - requires monitoring
    RED = "red"          # Critical/Urgent - immediate action needed


class AlertStatus(Enum):
    """Status of alert in its lifecycle."""
    ACTIVE = "active"         # Alert is currently active
    ACKNOWLEDGED = "acknowledged"  # Alert has been seen by user
    INVESTIGATING = "investigating"  # Alert is being investigated
    RESOLVED = "resolved"     # Alert has been resolved
    DISMISSED = "dismissed"   # Alert was dismissed as false positive
    EXPIRED = "expired"       # Alert expired due to time limits


class AnomalyDetectionMethod(Enum):
    """Methods for detecting anomalies."""
    Z_SCORE = "z_score"               # Standard deviation based
    IQR = "interquartile_range"       # Interquartile range based
    ISOLATION_FOREST = "isolation_forest"  # Ensemble method
    MOVING_AVERAGE = "moving_average"  # Moving average deviation
    SEASONAL_DECOMPOSITION = "seasonal"  # Time series decomposition


@dataclass
class AlertThresholds:
    """Configuration thresholds for alert detection."""
    rapid_growth_percentage: float = 50.0    # 50% growth rate threshold
    rapid_growth_timeframe_hours: int = 24   # Within 24 hours
    decline_percentage: float = -20.0        # 20% decline threshold
    decline_timeframe_hours: int = 48        # Within 48 hours
    anomaly_z_score_threshold: float = 3.0   # 3 standard deviations
    anomaly_iqr_multiplier: float = 1.5      # 1.5x IQR for outliers
    minimum_data_points: int = 7             # Minimum points for analysis
    data_quality_threshold: float = 0.7      # Minimum quality score
    
    # Milestone thresholds
    subscriber_milestones: List[int] = field(default_factory=lambda: [
        1000, 10000, 100000, 500000, 1000000, 5000000, 10000000
    ])
    listener_milestones: List[int] = field(default_factory=lambda: [
        1000, 50000, 500000, 1000000, 5000000, 10000000
    ])


@dataclass
class Alert:
    """Individual alert with comprehensive metadata."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    artist_id: int
    artist_name: str
    platform: str
    metric_type: str
    
    # Alert content
    title: str
    message: str
    description: str
    
    # Alert data
    current_value: float
    
    # Alert level (green/yellow/red)
    alert_level: AlertLevel = AlertLevel.YELLOW
    previous_value: Optional[float] = None
    percentage_change: Optional[float] = None
    threshold_value: Optional[float] = None
    detection_method: Optional[str] = None
    
    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    alert_timeframe_start: Optional[datetime] = None
    alert_timeframe_end: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    data_points_analyzed: int = 0
    confidence_score: float = 0.0
    related_alerts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    false_positive: bool = False


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""
    is_anomaly: bool
    anomaly_score: float
    detection_method: AnomalyDetectionMethod
    threshold_used: float
    data_point: MetricDataPoint
    context_window: List[MetricDataPoint]
    explanation: str
    confidence: float = 0.0


@dataclass
class AlertTemplate:
    """Template for generating consistent alert messages."""
    template_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    title_template: str
    message_template: str
    description_template: str
    action_required: str
    escalation_criteria: Optional[str] = None
    auto_resolve_conditions: Optional[str] = None


class AlertResolutionEngine:
    """Engine for automatic alert resolution based on conditions."""
    
    def __init__(self):
        self.resolution_rules: Dict[AlertType, List[Callable]] = {
            AlertType.RAPID_GROWTH: [self._check_growth_stabilization],
            AlertType.GROWTH_DECLINE: [self._check_recovery],
            AlertType.ANOMALY_DETECTED: [self._check_anomaly_resolved],
            AlertType.MILESTONE_REACHED: [self._auto_resolve_milestone],
            AlertType.PERFORMANCE_DEGRADATION: [self._check_performance_recovery],
            AlertType.DATA_QUALITY_ISSUE: [self._check_data_quality_improved]
        }
    
    def check_auto_resolution(self, alert: Alert, current_metrics: ArtistMetrics) -> Tuple[bool, Optional[str]]:
        """Check if alert should be auto-resolved."""
        if alert.alert_type not in self.resolution_rules:
            return False, None
        
        for rule in self.resolution_rules[alert.alert_type]:
            should_resolve, reason = rule(alert, current_metrics)
            if should_resolve:
                return True, reason
        
        return False, None
    
    def _check_growth_stabilization(self, alert: Alert, metrics: ArtistMetrics) -> Tuple[bool, Optional[str]]:
        """Check if rapid growth has stabilized."""
        if len(metrics.data_points) < 3:
            return False, None
        
        recent_points = sorted(metrics.data_points[-3:], key=lambda x: x.timestamp)
        growth_rates = []
        
        for i in range(1, len(recent_points)):
            prev_val = recent_points[i-1].value
            curr_val = recent_points[i].value
            if prev_val > 0:
                growth_rate = ((curr_val - prev_val) / prev_val) * 100
                growth_rates.append(growth_rate)
        
        if growth_rates and max(growth_rates) < 10:  # Growth below 10%
            return True, "Growth rate has stabilized below 10%"
        
        return False, None
    
    def _check_recovery(self, alert: Alert, metrics: ArtistMetrics) -> Tuple[bool, Optional[str]]:
        """Check if decline has recovered."""
        if len(metrics.data_points) < 2:
            return False, None
        
        recent_points = sorted(metrics.data_points[-2:], key=lambda x: x.timestamp)
        if len(recent_points) == 2:
            prev_val = recent_points[0].value
            curr_val = recent_points[1].value
            if prev_val > 0:
                growth_rate = ((curr_val - prev_val) / prev_val) * 100
                if growth_rate > 5:  # Positive growth of 5%+
                    return True, f"Recovery detected with {growth_rate:.1f}% growth"
        
        return False, None
    
    def _check_anomaly_resolved(self, alert: Alert, metrics: ArtistMetrics) -> Tuple[bool, Optional[str]]:
        """Check if anomaly condition has been resolved."""
        if len(metrics.data_points) < 5:
            return False, None
        
        recent_values = [dp.value for dp in metrics.data_points[-5:]]
        mean_val = statistics.mean(recent_values)
        std_val = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        
        if std_val > 0:
            latest_z_score = abs((recent_values[-1] - mean_val) / std_val)
            if latest_z_score < 2.0:  # Within 2 standard deviations
                return True, f"Values normalized, latest z-score: {latest_z_score:.2f}"
        
        return False, None
    
    def _auto_resolve_milestone(self, alert: Alert, metrics: ArtistMetrics) -> Tuple[bool, Optional[str]]:
        """Auto-resolve milestone alerts after acknowledgment."""
        if alert.status == AlertStatus.ACKNOWLEDGED:
            return True, "Milestone alert auto-resolved after acknowledgment"
        return False, None
    
    def _check_performance_recovery(self, alert: Alert, metrics: ArtistMetrics) -> Tuple[bool, Optional[str]]:
        """Check if performance has recovered."""
        if len(metrics.data_points) < 3:
            return False, None
        
        recent_values = [dp.value for dp in metrics.data_points[-3:]]
        if len(recent_values) >= 2:
            # Check for improvement trend
            improvements = sum(1 for i in range(1, len(recent_values)) 
                             if recent_values[i] > recent_values[i-1])
            if improvements >= 2:  # At least 2 improvements
                return True, "Performance recovery trend detected"
        
        return False, None
    
    def _check_data_quality_improved(self, alert: Alert, metrics: ArtistMetrics) -> Tuple[bool, Optional[str]]:
        """Check if data quality has improved."""
        recent_points = metrics.data_points[-5:] if len(metrics.data_points) >= 5 else metrics.data_points
        if not recent_points:
            return False, None
        
        quality_scores = [dp.quality_score for dp in recent_points if dp.quality_score > 0]
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            if avg_quality > 0.8:  # Quality above 80%
                return True, f"Data quality improved to {avg_quality:.2f}"
        
        return False, None


class AlertMessageTemplates:
    """Centralized alert message templates for consistent messaging."""
    
    @staticmethod
    def get_templates() -> Dict[AlertType, AlertTemplate]:
        """Get all predefined alert templates."""
        return {
            AlertType.RAPID_GROWTH: AlertTemplate(
                template_id="rapid_growth_001",
                alert_type=AlertType.RAPID_GROWTH,
                alert_level=AlertLevel.GREEN,
                title_template="🚀 급속 성장 감지: {artist_name}",
                message_template="{artist_name}이(가) {platform}에서 {timeframe}시간 동안 {growth_rate:.1f}% 성장했습니다",
                description_template="{artist_name}의 {metric_type}이(가) {timeframe}시간 내에 {previous_value:,}에서 {current_value:,}로 증가했습니다. 성장률: {growth_rate:.1f}%",
                action_required="성장 동력 분석 및 마케팅 전략 최적화 검토",
                escalation_criteria="성장률 200% 초과 시 경영진 보고",
                auto_resolve_conditions="성장률이 10% 미만으로 안정화 시"
            ),
            
            AlertType.GROWTH_DECLINE: AlertTemplate(
                template_id="growth_decline_001",
                alert_type=AlertType.GROWTH_DECLINE,
                alert_level=AlertLevel.RED,
                title_template="📉 성장 둔화 감지: {artist_name}",
                message_template="{artist_name}이(가) {platform}에서 {timeframe}시간 동안 {decline_rate:.1f}% 감소했습니다",
                description_template="{artist_name}의 {metric_type}이(가) {timeframe}시간 내에 {previous_value:,}에서 {current_value:,}로 감소했습니다. 감소율: {decline_rate:.1f}%",
                action_required="즉시 원인 분석 및 대응 전략 수립 필요",
                escalation_criteria="감소율 30% 초과 시 긴급 대응팀 소집",
                auto_resolve_conditions="5% 이상 회복 성장 시"
            ),
            
            AlertType.ANOMALY_DETECTED: AlertTemplate(
                template_id="anomaly_001",
                alert_type=AlertType.ANOMALY_DETECTED,
                alert_level=AlertLevel.YELLOW,
                title_template="🔍 이상 패턴 감지: {artist_name}",
                message_template="{artist_name}의 {platform} 지표에서 통계적 이상 패턴이 감지되었습니다",
                description_template="{detection_method} 방법으로 이상 패턴 감지. 이상 점수: {anomaly_score:.2f}, 신뢰도: {confidence:.2f}",
                action_required="데이터 검증 및 이상 원인 조사",
                escalation_criteria="이상 점수 5.0 초과 시 데이터팀 즉시 검토",
                auto_resolve_conditions="정상 범위 (z-score < 2.0) 복귀 시"
            ),
            
            AlertType.MILESTONE_REACHED: AlertTemplate(
                template_id="milestone_001",
                alert_type=AlertType.MILESTONE_REACHED,
                alert_level=AlertLevel.GREEN,
                title_template="🎉 마일스톤 달성: {artist_name}",
                message_template="{artist_name}이(가) {milestone:,} {metric_type} 마일스톤을 달성했습니다!",
                description_template="{platform}에서 {milestone:,} {metric_type} 달성. 이전 값: {previous_value:,}",
                action_required="성과 축하 및 다음 목표 설정",
                escalation_criteria="주요 마일스톤 (1M, 10M) 달성 시 보도자료 준비",
                auto_resolve_conditions="확인 후 자동 해결"
            ),
            
            AlertType.PERFORMANCE_DEGRADATION: AlertTemplate(
                template_id="performance_001",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                alert_level=AlertLevel.RED,
                title_template="⚠️ 지속적 성과 하락: {artist_name}",
                message_template="{artist_name}의 성과가 {consecutive_periods}회 연속 하락하고 있습니다",
                description_template="{consecutive_periods}회 연속 하락 패턴 감지. 즉시 대응이 필요합니다.",
                action_required="긴급 성과 개선 전략 수립 및 실행",
                escalation_criteria="3회 연속 하락 시 경영진 보고",
                auto_resolve_conditions="2회 연속 개선 시"
            ),
            
            AlertType.DATA_QUALITY_ISSUE: AlertTemplate(
                template_id="data_quality_001",
                alert_type=AlertType.DATA_QUALITY_ISSUE,
                alert_level=AlertLevel.YELLOW,
                title_template="📊 데이터 품질 문제: {artist_name}",
                message_template="{artist_name}의 데이터 품질이 임계값 미만입니다 (품질점수: {quality_score:.2f})",
                description_template="평균 데이터 품질 점수: {quality_score:.2f}, 임계값: {threshold:.2f}",
                action_required="데이터 소스 점검 및 품질 개선",
                escalation_criteria="품질 점수 0.5 미만 시 데이터 수집 중단 검토",
                auto_resolve_conditions="품질 점수 0.8 이상 복구 시"
            ),
            
            AlertType.UNUSUAL_ACTIVITY: AlertTemplate(
                template_id="unusual_activity_001",
                alert_type=AlertType.UNUSUAL_ACTIVITY,
                alert_level=AlertLevel.YELLOW,
                title_template="🔎 비정상 활동 감지: {artist_name}",
                message_template="{artist_name}에서 평소와 다른 활동 패턴이 감지되었습니다",
                description_template="비정상 활동 패턴 감지. 상세 분석이 필요합니다.",
                action_required="활동 패턴 분석 및 원인 조사",
                escalation_criteria="의심스러운 활동 시 보안팀 신고",
                auto_resolve_conditions="정상 패턴 복귀 시"
            )
        }
    
    @staticmethod
    def get_template(alert_type: AlertType) -> Optional[AlertTemplate]:
        """Get specific template by alert type."""
        templates = AlertMessageTemplates.get_templates()
        return templates.get(alert_type)
    
    @staticmethod
    def format_alert_message(template: AlertTemplate, **kwargs) -> Dict[str, str]:
        """Format alert message using template and provided data."""
        try:
            return {
                'title': template.title_template.format(**kwargs),
                'message': template.message_template.format(**kwargs),
                'description': template.description_template.format(**kwargs),
                'action_required': template.action_required,
                'escalation_criteria': template.escalation_criteria,
                'auto_resolve_conditions': template.auto_resolve_conditions
            }
        except KeyError as e:
            logger.error(f"Missing template parameter: {e}")
            return {
                'title': f"Alert: {kwargs.get('artist_name', 'Unknown')}",
                'message': f"Alert generated for {template.alert_type.value}",
                'description': "Template formatting error - please check parameters",
                'action_required': template.action_required,
                'escalation_criteria': template.escalation_criteria,
                'auto_resolve_conditions': template.auto_resolve_conditions
            }


class AlertEngine:
    """
    Intelligent alert engine for K-Pop artist performance monitoring.
    
    Provides comprehensive alert detection, generation, and management
    capabilities with configurable thresholds and multiple detection methods.
    """
    
    def __init__(self, 
                 thresholds: Optional[AlertThresholds] = None,
                 enable_auto_resolution: bool = True,
                 alert_retention_days: int = 30):
        """
        Initialize the alert engine.
        
        Args:
            thresholds: Configuration for alert detection thresholds
            enable_auto_resolution: Whether to auto-resolve expired alerts
            alert_retention_days: How long to keep resolved alerts
        """
        self.thresholds = thresholds or AlertThresholds()
        self.enable_auto_resolution = enable_auto_resolution
        self.alert_retention_days = alert_retention_days
        
        # Initialize growth rate calculator for analysis
        self.growth_calculator = GrowthRateCalculator(
            min_data_points=self.thresholds.minimum_data_points,
            confidence_level=0.95
        )
        
        # Initialize resolution engine
        self.resolution_engine = AlertResolutionEngine()
        
        # Alert storage and management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_counter = 0
        
        # Performance tracking
        self.detection_stats = {
            'total_analyses': 0,
            'alerts_generated': 0,
            'false_positives': 0,
            'auto_resolved': 0
        }
        
        logger.info(f"AlertEngine initialized with {len(AlertType)} alert types")
    
    def detect_rapid_growth(self, 
                          artist_metrics: ArtistMetrics,
                          custom_threshold: Optional[float] = None,
                          timeframe_hours: Optional[int] = None) -> List[Alert]:
        """
        Detect rapid growth in artist metrics.
        
        Args:
            artist_metrics: Artist metrics data to analyze
            custom_threshold: Override default growth threshold
            timeframe_hours: Override default timeframe
            
        Returns:
            List of generated alerts for rapid growth detection
        """
        alerts = []
        threshold = custom_threshold or self.thresholds.rapid_growth_percentage
        timeframe = timeframe_hours or self.thresholds.rapid_growth_timeframe_hours
        
        logger.debug(f"Analyzing rapid growth for {artist_metrics.artist_name} on {artist_metrics.platform}")
        
        try:
            # Get recent data points within timeframe
            cutoff_time = datetime.now() - timedelta(hours=timeframe)
            recent_points = [
                dp for dp in artist_metrics.data_points
                if dp.timestamp >= cutoff_time
            ]
            
            if len(recent_points) < 2:
                logger.debug("Insufficient recent data for rapid growth analysis")
                return alerts
            
            # Sort by timestamp to ensure proper ordering
            recent_points.sort(key=lambda x: x.timestamp)
            
            # Calculate growth rate for the timeframe
            growth_result = self.growth_calculator.calculate_growth_rate(
                recent_points,
                method=CalculationMethod.SIMPLE,
                period=GrowthPeriod.DAILY
            )
            
            if growth_result and growth_result.growth_rate >= threshold:
                # Generate rapid growth alert
                alert = self._create_rapid_growth_alert(
                    artist_metrics, growth_result, threshold, timeframe, recent_points
                )
                alerts.append(alert)
                
                # Check for milestone achievements during rapid growth
                milestone_alerts = self._check_milestones(artist_metrics, recent_points)
                alerts.extend(milestone_alerts)
                
                logger.info(f"Rapid growth detected: {artist_metrics.artist_name} "
                          f"grew {growth_result.growth_rate:.1f}% in {timeframe}h")
            
            self.detection_stats['total_analyses'] += 1
            
        except Exception as e:
            logger.error(f"Error in rapid growth detection: {e}")
        
        return alerts
    
    def detect_growth_decline(self,
                            artist_metrics: ArtistMetrics,
                            custom_threshold: Optional[float] = None,
                            timeframe_hours: Optional[int] = None) -> List[Alert]:
        """
        Detect significant growth decline or negative trends.
        
        Args:
            artist_metrics: Artist metrics data to analyze
            custom_threshold: Override default decline threshold
            timeframe_hours: Override default timeframe
            
        Returns:
            List of generated alerts for growth decline
        """
        alerts = []
        threshold = custom_threshold or self.thresholds.decline_percentage
        timeframe = timeframe_hours or self.thresholds.decline_timeframe_hours
        
        logger.debug(f"Analyzing growth decline for {artist_metrics.artist_name}")
        
        try:
            # Get data points within timeframe
            cutoff_time = datetime.now() - timedelta(hours=timeframe)
            timeframe_points = [
                dp for dp in artist_metrics.data_points
                if dp.timestamp >= cutoff_time
            ]
            
            if len(timeframe_points) < 2:
                logger.debug("Insufficient data for decline analysis")
                return alerts
            
            timeframe_points.sort(key=lambda x: x.timestamp)
            
            # Calculate decline over the timeframe
            growth_result = self.growth_calculator.calculate_growth_rate(
                timeframe_points,
                method=CalculationMethod.SIMPLE,
                period=GrowthPeriod.DAILY
            )
            
            if growth_result and growth_result.growth_rate <= threshold:
                # Generate decline alert
                alert = self._create_decline_alert(
                    artist_metrics, growth_result, threshold, timeframe, timeframe_points
                )
                alerts.append(alert)
                
                # Check for performance degradation patterns
                degradation_alerts = self._analyze_performance_degradation(
                    artist_metrics, timeframe_points
                )
                alerts.extend(degradation_alerts)
                
                logger.warning(f"Growth decline detected: {artist_metrics.artist_name} "
                             f"declined {growth_result.growth_rate:.1f}% in {timeframe}h")
            
            self.detection_stats['total_analyses'] += 1
            
        except Exception as e:
            logger.error(f"Error in decline detection: {e}")
        
        return alerts
    
    def detect_anomalies(self,
                        artist_metrics: ArtistMetrics,
                        methods: Optional[List[AnomalyDetectionMethod]] = None,
                        lookback_days: int = 30) -> List[Alert]:
        """
        Detect statistical anomalies in artist metrics using multiple methods.
        
        Args:
            artist_metrics: Artist metrics data to analyze
            methods: List of detection methods to use
            lookback_days: Number of days to look back for analysis
            
        Returns:
            List of generated alerts for detected anomalies
        """
        alerts = []
        methods = methods or [
            AnomalyDetectionMethod.Z_SCORE,
            AnomalyDetectionMethod.IQR,
            AnomalyDetectionMethod.MOVING_AVERAGE
        ]
        
        logger.debug(f"Running anomaly detection for {artist_metrics.artist_name}")
        
        try:
            # Get data points for analysis window
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            analysis_points = [
                dp for dp in artist_metrics.data_points
                if dp.timestamp >= cutoff_time and dp.value > 0
            ]
            
            if len(analysis_points) < self.thresholds.minimum_data_points:
                logger.debug("Insufficient data for anomaly detection")
                return alerts
            
            analysis_points.sort(key=lambda x: x.timestamp)
            
            # Run each detection method
            for method in methods:
                anomaly_results = self._run_anomaly_detection(analysis_points, method)
                
                for anomaly in anomaly_results:
                    if anomaly.is_anomaly and anomaly.confidence > 0.7:
                        alert = self._create_anomaly_alert(
                            artist_metrics, anomaly, analysis_points
                        )
                        alerts.append(alert)
                        
                        logger.info(f"Anomaly detected: {artist_metrics.artist_name} "
                                  f"using {method.value} method, score: {anomaly.anomaly_score:.2f}")
            
            # Check for data quality issues
            quality_alerts = self._check_data_quality(artist_metrics, analysis_points)
            alerts.extend(quality_alerts)
            
            self.detection_stats['total_analyses'] += 1
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return alerts
    
    def generate_alert(self,
                      alert_type: AlertType,
                      artist_metrics: ArtistMetrics,
                      severity: AlertSeverity,
                      title: str = None,
                      message: str = None,
                      **kwargs) -> Alert:
        """
        Generate a new alert with comprehensive metadata using templates.
        
        Args:
            alert_type: Type of alert to generate
            artist_metrics: Associated artist metrics
            severity: Severity level of the alert
            title: Alert title (optional, will use template if not provided)
            message: Alert message (optional, will use template if not provided)
            **kwargs: Additional alert parameters for template formatting
            
        Returns:
            Generated alert object
        """
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{int(datetime.now().timestamp())}"
        
        # Get template and format message if title/message not provided
        template = AlertMessageTemplates.get_template(alert_type)
        if template and (not title or not message):
            # Prepare template data
            template_data = {
                'artist_name': artist_metrics.artist_name,
                'platform': artist_metrics.platform,
                'metric_type': artist_metrics.metric_type,
                'current_value': artist_metrics.current_value,
                **kwargs
            }
            
            formatted = AlertMessageTemplates.format_alert_message(template, **template_data)
            title = title or formatted['title']
            message = message or formatted['message']
            kwargs['description'] = kwargs.get('description', formatted['description'])
            kwargs['action_required'] = formatted['action_required']
        
        # Determine alert level from severity
        alert_level = self._map_severity_to_level(severity)
        
        # Calculate expiration time based on severity
        expiration_hours = {
            AlertSeverity.CRITICAL: 24,
            AlertSeverity.HIGH: 48,
            AlertSeverity.MEDIUM: 72,
            AlertSeverity.LOW: 168,  # 1 week
            AlertSeverity.INFO: 336   # 2 weeks
        }
        
        expires_at = datetime.now() + timedelta(hours=expiration_hours[severity])
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE,
            artist_id=artist_metrics.artist_id,
            artist_name=artist_metrics.artist_name,
            platform=artist_metrics.platform,
            metric_type=artist_metrics.metric_type,
            title=title or f"Alert: {artist_metrics.artist_name}",
            message=message or f"Alert generated for {alert_type.value}",
            description=kwargs.get('description', message or f"Alert generated for {alert_type.value}"),
            current_value=artist_metrics.current_value,
            alert_level=alert_level,
            expires_at=expires_at,
            metadata={k: v for k, v in kwargs.items() if k not in ['description', 'action_required'] and k not in ['previous_value', 'percentage_change', 'threshold_value', 'detection_method', 'data_points_analyzed', 'confidence_score', 'alert_timeframe_start', 'alert_timeframe_end']},
            previous_value=kwargs.get('previous_value'),
            percentage_change=kwargs.get('percentage_change'),
            threshold_value=kwargs.get('threshold_value'),
            detection_method=kwargs.get('detection_method'),
            data_points_analyzed=kwargs.get('data_points_analyzed', 0),
            confidence_score=kwargs.get('confidence_score', 0.0),
            alert_timeframe_start=kwargs.get('alert_timeframe_start'),
            alert_timeframe_end=kwargs.get('alert_timeframe_end')
        )
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        self.detection_stats['alerts_generated'] += 1
        
        # Add contextual tags
        self._add_alert_tags(alert, artist_metrics)
        
        logger.info(f"Generated {severity.value} {alert_type.value} alert: {title}")
        
        return alert
    
    def resolve_alert(self,
                     alert_id: str,
                     resolution_notes: Optional[str] = None,
                     resolved_by: Optional[str] = None,
                     mark_false_positive: bool = False) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: ID of alert to resolve
            resolution_notes: Notes about the resolution
            resolved_by: Who resolved the alert
            mark_false_positive: Whether to mark as false positive
            
        Returns:
            True if alert was successfully resolved, False otherwise
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Cannot resolve alert {alert_id}: not found in active alerts")
            return False
        
        alert = self.active_alerts[alert_id]
        
        # Update alert status
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.resolution_notes = resolution_notes
        alert.resolved_by = resolved_by
        alert.false_positive = mark_false_positive
        
        # Move to history and remove from active
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        if mark_false_positive:
            self.detection_stats['false_positives'] += 1
        
        logger.info(f"Resolved alert {alert_id}: {alert.title}")
        
        # Clean up old history entries
        self._cleanup_old_alerts()
        
        return True
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: Optional[str] = None) -> bool:
        """Acknowledge an alert to indicate it has been seen."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        if alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.metadata['acknowledged_by'] = acknowledged_by
            alert.metadata['acknowledged_at'] = datetime.now().isoformat()
            
            logger.info(f"Acknowledged alert {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self, 
                         severity_filter: Optional[AlertSeverity] = None,
                         alert_type_filter: Optional[AlertType] = None,
                         artist_id_filter: Optional[int] = None) -> List[Alert]:
        """Get filtered list of active alerts."""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        if alert_type_filter:
            alerts = [a for a in alerts if a.alert_type == alert_type_filter]
        
        if artist_id_filter:
            alerts = [a for a in alerts if a.artist_id == artist_id_filter]
        
        # Sort by severity and then by detection time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1, 
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4
        }
        
        alerts.sort(key=lambda x: (severity_order[x.severity], x.detected_at), reverse=True)
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert system statistics."""
        active_by_severity = defaultdict(int)
        active_by_type = defaultdict(int)
        
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
            active_by_type[alert.alert_type.value] += 1
        
        return {
            'active_alerts': len(self.active_alerts),
            'total_generated': self.detection_stats['alerts_generated'],
            'total_resolved': len(self.alert_history),
            'false_positives': self.detection_stats['false_positives'],
            'auto_resolved': self.detection_stats['auto_resolved'],
            'analyses_performed': self.detection_stats['total_analyses'],
            'active_by_severity': dict(active_by_severity),
            'active_by_type': dict(active_by_type),
            'alert_retention_days': self.alert_retention_days
        }
    
    def cleanup_expired_alerts(self) -> int:
        """Clean up expired alerts and return count of cleaned alerts."""
        if not self.enable_auto_resolution:
            return 0
        
        now = datetime.now()
        expired_alert_ids = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.expires_at and now > alert.expires_at:
                expired_alert_ids.append(alert_id)
        
        # Auto-resolve expired alerts
        for alert_id in expired_alert_ids:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.EXPIRED
            alert.resolved_at = now
            alert.resolution_notes = "Auto-resolved due to expiration"
            
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
        
        if expired_alert_ids:
            self.detection_stats['auto_resolved'] += len(expired_alert_ids)
            logger.info(f"Auto-resolved {len(expired_alert_ids)} expired alerts")
        
        return len(expired_alert_ids)
    
    def process_auto_resolution(self, artist_metrics: ArtistMetrics) -> int:
        """Process automatic resolution for alerts based on current conditions."""
        resolved_count = 0
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.artist_id == artist_metrics.artist_id:
                should_resolve, reason = self.resolution_engine.check_auto_resolution(alert, artist_metrics)
                if should_resolve:
                    alerts_to_resolve.append((alert_id, reason))
        
        # Resolve alerts
        for alert_id, reason in alerts_to_resolve:
            if self.resolve_alert(alert_id, resolution_notes=reason, resolved_by="AutoResolution"):
                resolved_count += 1
                logger.info(f"Auto-resolved alert {alert_id}: {reason}")
        
        return resolved_count
    
    def get_alerts_by_level(self, alert_level: AlertLevel) -> List[Alert]:
        """Get all active alerts by alert level (green/yellow/red)."""
        return [alert for alert in self.active_alerts.values() if alert.alert_level == alert_level]
    
    def get_alert_level_summary(self) -> Dict[str, int]:
        """Get summary count by alert level."""
        summary = {level.value: 0 for level in AlertLevel}
        
        for alert in self.active_alerts.values():
            summary[alert.alert_level.value] += 1
        
        return summary
    
    # Private helper methods
    
    def _create_rapid_growth_alert(self,
                                  artist_metrics: ArtistMetrics,
                                  growth_result: GrowthRateResult,
                                  threshold: float,
                                  timeframe: int,
                                  data_points: List[MetricDataPoint]) -> Alert:
        """Create a rapid growth alert."""
        severity = self._determine_growth_severity(growth_result.growth_rate)
        
        title = f"Rapid Growth Detected: {artist_metrics.artist_name}"
        message = (f"{artist_metrics.artist_name} experienced {growth_result.growth_rate:.1f}% "
                  f"growth in {timeframe} hours on {artist_metrics.platform}")
        
        description = (f"Rapid growth alert triggered for {artist_metrics.artist_name}. "
                      f"Growth rate of {growth_result.growth_rate:.1f}% exceeds threshold "
                      f"of {threshold}% within {timeframe} hour timeframe. "
                      f"Current value: {artist_metrics.current_value:,}")
        
        return self.generate_alert(
            alert_type=AlertType.RAPID_GROWTH,
            artist_metrics=artist_metrics,
            severity=severity,
            title=title,
            message=message,
            description=description,
            percentage_change=growth_result.growth_rate,
            threshold_value=threshold,
            detection_method="growth_rate_analysis",
            data_points_analyzed=len(data_points),
            confidence_score=growth_result.confidence_interval,
            alert_timeframe_start=data_points[0].timestamp,
            alert_timeframe_end=data_points[-1].timestamp
        )
    
    def _create_decline_alert(self,
                            artist_metrics: ArtistMetrics,
                            growth_result: GrowthRateResult,
                            threshold: float,
                            timeframe: int,
                            data_points: List[MetricDataPoint]) -> Alert:
        """Create a growth decline alert."""
        severity = self._determine_decline_severity(growth_result.growth_rate)
        
        title = f"Growth Decline Detected: {artist_metrics.artist_name}"
        message = (f"{artist_metrics.artist_name} experienced {growth_result.growth_rate:.1f}% "
                  f"decline in {timeframe} hours on {artist_metrics.platform}")
        
        description = (f"Growth decline alert triggered for {artist_metrics.artist_name}. "
                      f"Decline rate of {growth_result.growth_rate:.1f}% exceeds threshold "
                      f"of {threshold}% within {timeframe} hour timeframe. "
                      f"Current value: {artist_metrics.current_value:,}")
        
        return self.generate_alert(
            alert_type=AlertType.GROWTH_DECLINE,
            artist_metrics=artist_metrics,
            severity=severity,
            title=title,
            message=message,
            description=description,
            percentage_change=growth_result.growth_rate,
            threshold_value=threshold,
            detection_method="decline_analysis",
            data_points_analyzed=len(data_points),
            confidence_score=growth_result.confidence_interval,
            alert_timeframe_start=data_points[0].timestamp,
            alert_timeframe_end=data_points[-1].timestamp
        )
    
    def _create_anomaly_alert(self,
                            artist_metrics: ArtistMetrics,
                            anomaly: AnomalyResult,
                            data_points: List[MetricDataPoint]) -> Alert:
        """Create an anomaly detection alert."""
        severity = self._determine_anomaly_severity(anomaly.anomaly_score)
        
        title = f"Anomaly Detected: {artist_metrics.artist_name}"
        message = (f"Statistical anomaly detected for {artist_metrics.artist_name} "
                  f"on {artist_metrics.platform}")
        
        description = (f"Anomaly detection alert using {anomaly.detection_method.value} method. "
                      f"{anomaly.explanation} Anomaly score: {anomaly.anomaly_score:.2f}, "
                      f"Confidence: {anomaly.confidence:.2f}")
        
        return self.generate_alert(
            alert_type=AlertType.ANOMALY_DETECTED,
            artist_metrics=artist_metrics,
            severity=severity,
            title=title,
            message=message,
            description=description,
            threshold_value=anomaly.threshold_used,
            detection_method=anomaly.detection_method.value,
            data_points_analyzed=len(data_points),
            confidence_score=anomaly.confidence,
            metadata={
                'anomaly_score': anomaly.anomaly_score,
                'anomaly_timestamp': anomaly.data_point.timestamp.isoformat(),
                'anomaly_value': anomaly.data_point.value,
                'context_window_size': len(anomaly.context_window)
            }
        )
    
    def _run_anomaly_detection(self,
                              data_points: List[MetricDataPoint],
                              method: AnomalyDetectionMethod) -> List[AnomalyResult]:
        """Run anomaly detection using specified method."""
        results = []
        
        if len(data_points) < self.thresholds.minimum_data_points:
            return results
        
        values = [dp.value for dp in data_points]
        
        try:
            if method == AnomalyDetectionMethod.Z_SCORE:
                results = self._detect_anomalies_z_score(data_points, values)
            elif method == AnomalyDetectionMethod.IQR:
                results = self._detect_anomalies_iqr(data_points, values)
            elif method == AnomalyDetectionMethod.MOVING_AVERAGE:
                results = self._detect_anomalies_moving_average(data_points, values)
            
        except Exception as e:
            logger.error(f"Error in {method.value} anomaly detection: {e}")
        
        return results
    
    def _detect_anomalies_z_score(self,
                                 data_points: List[MetricDataPoint],
                                 values: List[float]) -> List[AnomalyResult]:
        """Detect anomalies using Z-score method."""
        results = []
        
        if len(values) < 3:
            return results
        
        mean_value = statistics.mean(values)
        stdev_value = statistics.stdev(values)
        
        if stdev_value == 0:
            return results
        
        threshold = self.thresholds.anomaly_z_score_threshold
        
        for i, (dp, value) in enumerate(zip(data_points, values)):
            z_score = abs((value - mean_value) / stdev_value)
            
            if z_score > threshold:
                # Get context window around anomaly
                start_idx = max(0, i - 5)
                end_idx = min(len(data_points), i + 6)
                context_window = data_points[start_idx:end_idx]
                
                explanation = (f"Value {value:,} deviates {z_score:.2f} standard deviations "
                             f"from mean {mean_value:,.0f}")
                
                result = AnomalyResult(
                    is_anomaly=True,
                    anomaly_score=z_score,
                    detection_method=AnomalyDetectionMethod.Z_SCORE,
                    threshold_used=threshold,
                    data_point=dp,
                    context_window=context_window,
                    explanation=explanation,
                    confidence=min(1.0, z_score / 5.0)  # Confidence increases with z-score
                )
                
                results.append(result)
        
        return results
    
    def _detect_anomalies_iqr(self,
                             data_points: List[MetricDataPoint],
                             values: List[float]) -> List[AnomalyResult]:
        """Detect anomalies using Interquartile Range method."""
        results = []
        
        if len(values) < 4:
            return results
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1
        
        if iqr == 0:
            return results
        
        multiplier = self.thresholds.anomaly_iqr_multiplier
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        for i, (dp, value) in enumerate(zip(data_points, values)):
            if value < lower_bound or value > upper_bound:
                # Get context window
                start_idx = max(0, i - 5)
                end_idx = min(len(data_points), i + 6)
                context_window = data_points[start_idx:end_idx]
                
                deviation = max(lower_bound - value, value - upper_bound, 0)
                anomaly_score = deviation / iqr if iqr > 0 else 0
                
                explanation = (f"Value {value:,} outside IQR bounds [{lower_bound:,.0f}, "
                             f"{upper_bound:,.0f}], deviation: {deviation:,.0f}")
                
                result = AnomalyResult(
                    is_anomaly=True,
                    anomaly_score=anomaly_score,
                    detection_method=AnomalyDetectionMethod.IQR,
                    threshold_used=multiplier,
                    data_point=dp,
                    context_window=context_window,
                    explanation=explanation,
                    confidence=min(1.0, anomaly_score / 3.0)
                )
                
                results.append(result)
        
        return results
    
    def _detect_anomalies_moving_average(self,
                                       data_points: List[MetricDataPoint],
                                       values: List[float]) -> List[AnomalyResult]:
        """Detect anomalies using moving average deviation."""
        results = []
        window_size = min(7, len(values) // 2)  # Adaptive window size
        
        if len(values) < window_size + 2:
            return results
        
        for i in range(window_size, len(values)):
            # Calculate moving average for window before current point
            window_values = values[i-window_size:i]
            moving_avg = statistics.mean(window_values)
            moving_std = statistics.stdev(window_values) if len(window_values) > 1 else 0
            
            if moving_std == 0:
                continue
            
            current_value = values[i]
            deviation = abs(current_value - moving_avg) / moving_std
            
            if deviation > 2.5:  # 2.5 standard deviations from moving average
                # Get context window
                start_idx = max(0, i - 5)
                end_idx = min(len(data_points), i + 6)
                context_window = data_points[start_idx:end_idx]
                
                explanation = (f"Value {current_value:,} deviates {deviation:.2f}σ from "
                             f"{window_size}-period moving average {moving_avg:,.0f}")
                
                result = AnomalyResult(
                    is_anomaly=True,
                    anomaly_score=deviation,
                    detection_method=AnomalyDetectionMethod.MOVING_AVERAGE,
                    threshold_used=2.5,
                    data_point=data_points[i],
                    context_window=context_window,
                    explanation=explanation,
                    confidence=min(1.0, deviation / 4.0)
                )
                
                results.append(result)
        
        return results
    
    def _check_milestones(self,
                         artist_metrics: ArtistMetrics,
                         recent_points: List[MetricDataPoint]) -> List[Alert]:
        """Check if any milestones were reached during recent activity."""
        alerts = []
        
        if not recent_points:
            return alerts
        
        recent_points.sort(key=lambda x: x.timestamp)
        start_value = recent_points[0].value
        end_value = recent_points[-1].value
        
        # Determine appropriate milestones based on metric type
        if artist_metrics.metric_type in ['subscribers', 'followers']:
            milestones = self.thresholds.subscriber_milestones
        elif artist_metrics.metric_type == 'monthly_listeners':
            milestones = self.thresholds.listener_milestones
        else:
            return alerts
        
        # Check which milestones were crossed
        for milestone in milestones:
            if start_value < milestone <= end_value:
                title = f"Milestone Reached: {artist_metrics.artist_name}"
                message = f"{artist_metrics.artist_name} reached {milestone:,} {artist_metrics.metric_type}"
                
                alert = self.generate_alert(
                    alert_type=AlertType.MILESTONE_REACHED,
                    artist_metrics=artist_metrics,
                    severity=AlertSeverity.INFO,
                    title=title,
                    message=message,
                    description=f"Milestone achievement: {milestone:,} {artist_metrics.metric_type} "
                               f"reached on {artist_metrics.platform}",
                    threshold_value=milestone,
                    previous_value=start_value,
                    tags=['milestone', f'milestone_{milestone}']
                )
                alerts.append(alert)
        
        return alerts
    
    def _analyze_performance_degradation(self,
                                       artist_metrics: ArtistMetrics,
                                       data_points: List[MetricDataPoint]) -> List[Alert]:
        """Analyze for sustained performance degradation patterns."""
        alerts = []
        
        if len(data_points) < 5:
            return alerts
        
        # Check for consecutive declining periods
        consecutive_declines = 0
        max_consecutive = 0
        
        values = [dp.value for dp in data_points]
        for i in range(1, len(values)):
            if values[i] < values[i-1]:
                consecutive_declines += 1
                max_consecutive = max(max_consecutive, consecutive_declines)
            else:
                consecutive_declines = 0
        
        # Generate alert if we have sustained decline
        if max_consecutive >= 3:  # 3+ consecutive declining periods
            title = f"Performance Degradation: {artist_metrics.artist_name}"
            message = (f"Sustained performance decline detected for {artist_metrics.artist_name} "
                      f"with {max_consecutive} consecutive declining periods")
            
            alert = self.generate_alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                artist_metrics=artist_metrics,
                severity=AlertSeverity.MEDIUM,
                title=title,
                message=message,
                description=f"Performance degradation pattern detected with {max_consecutive} "
                           f"consecutive declining measurements",
                metadata={'consecutive_declines': max_consecutive},
                tags=['performance', 'degradation']
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_data_quality(self,
                          artist_metrics: ArtistMetrics,
                          data_points: List[MetricDataPoint]) -> List[Alert]:
        """Check for data quality issues."""
        alerts = []
        
        if not data_points:
            return alerts
        
        # Calculate average quality score
        quality_scores = [dp.quality_score for dp in data_points if dp.quality_score > 0]
        
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            
            if avg_quality < self.thresholds.data_quality_threshold:
                title = f"Data Quality Issue: {artist_metrics.artist_name}"
                message = (f"Low data quality detected for {artist_metrics.artist_name} "
                          f"(avg quality: {avg_quality:.2f})")
                
                alert = self.generate_alert(
                    alert_type=AlertType.DATA_QUALITY_ISSUE,
                    artist_metrics=artist_metrics,
                    severity=AlertSeverity.LOW,
                    title=title,
                    message=message,
                    description=f"Data quality below threshold. Average quality score: "
                               f"{avg_quality:.2f}, threshold: {self.thresholds.data_quality_threshold}",
                    metadata={'average_quality_score': avg_quality},
                    tags=['data_quality']
                )
                alerts.append(alert)
        
        return alerts
    
    def _determine_growth_severity(self, growth_rate: float) -> AlertSeverity:
        """Determine alert severity based on growth rate."""
        if growth_rate >= 200:  # 200%+ growth
            return AlertSeverity.CRITICAL
        elif growth_rate >= 100:  # 100%+ growth
            return AlertSeverity.HIGH
        elif growth_rate >= 50:   # 50%+ growth
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _determine_decline_severity(self, decline_rate: float) -> AlertSeverity:
        """Determine alert severity based on decline rate."""
        abs_decline = abs(decline_rate)
        
        if abs_decline >= 50:    # 50%+ decline
            return AlertSeverity.CRITICAL
        elif abs_decline >= 30:  # 30%+ decline
            return AlertSeverity.HIGH
        elif abs_decline >= 15:  # 15%+ decline
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _determine_anomaly_severity(self, anomaly_score: float) -> AlertSeverity:
        """Determine alert severity based on anomaly score."""
        if anomaly_score >= 5.0:
            return AlertSeverity.CRITICAL
        elif anomaly_score >= 4.0:
            return AlertSeverity.HIGH
        elif anomaly_score >= 3.0:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _add_alert_tags(self, alert: Alert, artist_metrics: ArtistMetrics):
        """Add contextual tags to alert."""
        tags = []
        
        # Platform and metric type tags
        tags.extend([artist_metrics.platform, artist_metrics.metric_type])
        
        # Company tag if available
        if hasattr(artist_metrics, 'company_name'):
            tags.append(f"company_{artist_metrics.company_name.lower().replace(' ', '_')}")
        
        # Value range tags
        current_value = artist_metrics.current_value
        if current_value >= 10000000:  # 10M+
            tags.append('tier_mega')
        elif current_value >= 1000000:  # 1M+
            tags.append('tier_major')
        elif current_value >= 100000:   # 100K+
            tags.append('tier_emerging')
        else:
            tags.append('tier_new')
        
        alert.tags.extend(tags)
    
    def _map_severity_to_level(self, severity: AlertSeverity) -> AlertLevel:
        """Map alert severity to 3-tier level system."""
        mapping = {
            AlertSeverity.CRITICAL: AlertLevel.RED,
            AlertSeverity.HIGH: AlertLevel.RED,
            AlertSeverity.MEDIUM: AlertLevel.YELLOW,
            AlertSeverity.LOW: AlertLevel.GREEN,
            AlertSeverity.INFO: AlertLevel.GREEN
        }
        return mapping.get(severity, AlertLevel.YELLOW)
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts beyond retention period."""
        if not self.alert_history:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.alert_retention_days)
        
        # Keep only alerts within retention period
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.resolved_at and alert.resolved_at > cutoff_date
        ]


# Export main components
__all__ = [
    'AlertEngine',
    'Alert',
    'AlertType',
    'AlertSeverity',
    'AlertStatus',
    'AlertLevel',
    'AlertThresholds',
    'AlertTemplate',
    'AlertMessageTemplates',
    'AlertResolutionEngine',
    'AnomalyDetectionMethod',
    'AnomalyResult'
]