"""
K-Pop 이벤트 영향도 분석기
==========================

K-Pop 업계 이벤트들이 아티스트 성과에 미치는 영향을 정량적으로 분석하는 시스템입니다.
시상식, 컴백, 콘서트 등 다양한 이벤트의 직간접적 영향을 측정합니다.

주요 기능:
- 이벤트 전후 성과 비교 분석
- 다중 이벤트 상호작용 분석
- 시간 지연 효과 측정
- 영향도 예측 모델링

Author: Backend Analytics Team
Version: 1.0.0
Date: 2025-09-08
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from collections import defaultdict
import json

from .kpop_event_calendar import EventCategory, EventImportance, KPopEvent
from .award_shows_data import AwardShow

# Configure logging
logger = logging.getLogger(__name__)


class ImpactType(Enum):
    """영향 유형 분류"""
    IMMEDIATE = "immediate"      # 즉시 영향 (당일~3일)
    SHORT_TERM = "short_term"    # 단기 영향 (4일~2주)
    MEDIUM_TERM = "medium_term"  # 중기 영향 (2주~2개월)
    LONG_TERM = "long_term"      # 장기 영향 (2개월 이상)


class ImpactDirection(Enum):
    """영향 방향"""
    POSITIVE = "positive"        # 긍정적 영향
    NEGATIVE = "negative"        # 부정적 영향
    NEUTRAL = "neutral"          # 중립적 영향
    MIXED = "mixed"              # 복합적 영향


class ImpactMagnitude(Enum):
    """영향 크기"""
    NEGLIGIBLE = "negligible"    # 무시할 수 있는 (0-5%)
    SMALL = "small"              # 작은 영향 (5-15%)
    MODERATE = "moderate"        # 보통 영향 (15-30%)
    LARGE = "large"              # 큰 영향 (30-50%)
    MASSIVE = "massive"          # 거대한 영향 (50% 이상)


@dataclass
class ImpactMeasurement:
    """영향도 측정 결과"""
    metric_name: str
    baseline_value: float
    peak_value: float
    impact_percentage: float
    impact_magnitude: ImpactMagnitude
    impact_direction: ImpactDirection
    statistical_significance: float  # p-value
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'metric_name': self.metric_name,
            'baseline_value': self.baseline_value,
            'peak_value': self.peak_value,
            'impact_percentage': self.impact_percentage,
            'impact_magnitude': self.impact_magnitude.value,
            'impact_direction': self.impact_direction.value,
            'statistical_significance': self.statistical_significance,
            'confidence_interval': list(self.confidence_interval)
        }


@dataclass
class EventImpactAnalysis:
    """종합 이벤트 영향도 분석 결과"""
    event_id: str
    artist_id: str
    event_name: str
    event_date: date
    event_category: EventCategory
    
    # 시간별 영향도
    immediate_impacts: List[ImpactMeasurement] = field(default_factory=list)
    short_term_impacts: List[ImpactMeasurement] = field(default_factory=list)
    medium_term_impacts: List[ImpactMeasurement] = field(default_factory=list)
    long_term_impacts: List[ImpactMeasurement] = field(default_factory=list)
    
    # 종합 점수
    overall_impact_score: float = 0.0
    impact_sustainability_score: float = 0.0  # 영향 지속성
    
    # 메타데이터
    analysis_period: Tuple[date, date] = None
    confidence_level: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    # 추가 분석
    spillover_effects: Dict[str, float] = field(default_factory=dict)  # 파급 효과
    contextual_factors: List[str] = field(default_factory=list)       # 맥락 요인
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'event_id': self.event_id,
            'artist_id': self.artist_id,
            'event_name': self.event_name,
            'event_date': self.event_date.isoformat(),
            'event_category': self.event_category.value,
            'immediate_impacts': [impact.to_dict() for impact in self.immediate_impacts],
            'short_term_impacts': [impact.to_dict() for impact in self.short_term_impacts],
            'medium_term_impacts': [impact.to_dict() for impact in self.medium_term_impacts],
            'long_term_impacts': [impact.to_dict() for impact in self.long_term_impacts],
            'overall_impact_score': self.overall_impact_score,
            'impact_sustainability_score': self.impact_sustainability_score,
            'analysis_period': [self.analysis_period[0].isoformat(), self.analysis_period[1].isoformat()] if self.analysis_period else None,
            'confidence_level': self.confidence_level,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'spillover_effects': self.spillover_effects,
            'contextual_factors': self.contextual_factors
        }


class EventImpactAnalyzer:
    """
    K-Pop 이벤트 영향도 분석기
    
    다양한 이벤트가 아티스트 성과에 미치는 영향을 정량적으로 분석합니다.
    """
    
    def __init__(self, database_connection=None):
        """이벤트 영향도 분석기 초기화"""
        self.db_conn = database_connection
        self.logger = logger
        
        # 분석 결과 저장소
        self.impact_analyses: Dict[str, EventImpactAnalysis] = {}
        
        # 이벤트 유형별 기본 영향 기간 설정
        self.impact_windows = {
            EventCategory.AWARD_SHOW: {
                ImpactType.IMMEDIATE: 3,
                ImpactType.SHORT_TERM: 14,
                ImpactType.MEDIUM_TERM: 60,
                ImpactType.LONG_TERM: 180
            },
            EventCategory.COMEBACK: {
                ImpactType.IMMEDIATE: 1,
                ImpactType.SHORT_TERM: 7,
                ImpactType.MEDIUM_TERM: 30,
                ImpactType.LONG_TERM: 90
            },
            EventCategory.CONCERT: {
                ImpactType.IMMEDIATE: 2,
                ImpactType.SHORT_TERM: 10,
                ImpactType.MEDIUM_TERM: 45,
                ImpactType.LONG_TERM: 120
            }
        }
        
        # 기본 영향 기간 (다른 이벤트들)
        self.default_impact_windows = {
            ImpactType.IMMEDIATE: 2,
            ImpactType.SHORT_TERM: 7,
            ImpactType.MEDIUM_TERM: 30,
            ImpactType.LONG_TERM: 90
        }
        
        # 통계 유의성 임계값
        self.significance_threshold = 0.05
        
        self.logger.info("Event Impact Analyzer initialized")
    
    def analyze_event_impact(
        self,
        event: KPopEvent,
        metrics_data: pd.DataFrame,
        baseline_days: int = 30,
        include_long_term: bool = True
    ) -> EventImpactAnalysis:
        """개별 이벤트의 영향도 종합 분석"""
        try:
            self.logger.info(f"Analyzing impact for event: {event.name} ({event.event_date})")
            
            event_datetime = pd.to_datetime(event.event_date)
            
            # 기준선 기간 설정
            baseline_start = event_datetime - timedelta(days=baseline_days)
            baseline_end = event_datetime - timedelta(days=1)
            
            # 영향 기간별 분석
            impact_windows = self._get_impact_windows(event.category)
            
            analysis = EventImpactAnalysis(
                event_id=event.event_id,
                artist_id=event.artist_id or "unknown",
                event_name=event.name,
                event_date=event.event_date,
                event_category=event.category
            )
            
            # 즉시 영향 분석
            immediate_end = event_datetime + timedelta(days=impact_windows[ImpactType.IMMEDIATE])
            analysis.immediate_impacts = self._analyze_impact_window(
                metrics_data, baseline_start, baseline_end, event_datetime, immediate_end, "immediate"
            )
            
            # 단기 영향 분석
            short_term_end = event_datetime + timedelta(days=impact_windows[ImpactType.SHORT_TERM])
            analysis.short_term_impacts = self._analyze_impact_window(
                metrics_data, baseline_start, baseline_end, event_datetime, short_term_end, "short_term"
            )
            
            # 중기 영향 분석
            medium_term_end = event_datetime + timedelta(days=impact_windows[ImpactType.MEDIUM_TERM])
            analysis.medium_term_impacts = self._analyze_impact_window(
                metrics_data, baseline_start, baseline_end, event_datetime, medium_term_end, "medium_term"
            )
            
            # 장기 영향 분석 (선택적)
            if include_long_term:
                long_term_end = event_datetime + timedelta(days=impact_windows[ImpactType.LONG_TERM])
                analysis.long_term_impacts = self._analyze_impact_window(
                    metrics_data, baseline_start, baseline_end, event_datetime, long_term_end, "long_term"
                )
            
            # 종합 점수 계산
            analysis.overall_impact_score = self._calculate_overall_impact_score(analysis)
            analysis.impact_sustainability_score = self._calculate_sustainability_score(analysis)
            
            # 분석 기간 설정
            end_date = long_term_end.date() if include_long_term else medium_term_end.date()
            analysis.analysis_period = (baseline_start.date(), end_date)
            
            # 신뢰도 계산
            analysis.confidence_level = self._calculate_analysis_confidence(metrics_data, analysis)
            
            # 파급 효과 분석
            analysis.spillover_effects = self._analyze_spillover_effects(event, metrics_data, analysis)
            
            # 맥락 요인 분석
            analysis.contextual_factors = self._identify_contextual_factors(event, metrics_data)
            
            # 결과 저장
            self.impact_analyses[event.event_id] = analysis
            
            self.logger.info(f"Impact analysis completed: overall score {analysis.overall_impact_score:.3f}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing event impact: {e}")
            raise
    
    def compare_event_impacts(
        self,
        events: List[KPopEvent],
        metrics_data: pd.DataFrame,
        comparison_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """다중 이벤트 영향도 비교 분석"""
        try:
            self.logger.info(f"Comparing impacts of {len(events)} events")
            
            if not comparison_metrics:
                comparison_metrics = ['overall_impact_score', 'impact_sustainability_score']
            
            # 각 이벤트별 영향도 분석
            event_analyses = {}
            for event in events:
                try:
                    analysis = self.analyze_event_impact(event, metrics_data)
                    event_analyses[event.event_id] = analysis
                except Exception as e:
                    self.logger.warning(f"Failed to analyze event {event.event_id}: {e}")
                    continue
            
            if not event_analyses:
                return {"error": "No events could be analyzed"}
            
            # 비교 분석 수행
            comparison_result = {
                "events_analyzed": len(event_analyses),
                "comparison_metrics": comparison_metrics,
                "rankings": {},
                "statistical_comparison": {},
                "category_analysis": {},
                "temporal_patterns": {},
                "insights": []
            }
            
            # 지표별 순위 생성
            for metric in comparison_metrics:
                rankings = []
                for event_id, analysis in event_analyses.items():
                    value = getattr(analysis, metric, 0.0)
                    rankings.append({
                        'event_id': event_id,
                        'event_name': analysis.event_name,
                        'category': analysis.event_category.value,
                        'value': value,
                        'date': analysis.event_date.isoformat()
                    })
                
                # 값 기준 정렬
                rankings.sort(key=lambda x: x['value'], reverse=True)
                comparison_result["rankings"][metric] = rankings
            
            # 통계적 비교
            comparison_result["statistical_comparison"] = self._perform_statistical_comparison(event_analyses)
            
            # 카테고리별 분석
            comparison_result["category_analysis"] = self._analyze_by_category(event_analyses)
            
            # 시간적 패턴 분석
            comparison_result["temporal_patterns"] = self._analyze_temporal_patterns(event_analyses)
            
            # 인사이트 생성
            comparison_result["insights"] = self._generate_comparison_insights(event_analyses, comparison_result)
            
            self.logger.info(f"Event comparison completed for {len(event_analyses)} events")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Error comparing event impacts: {e}")
            return {"error": str(e)}
    
    def analyze_multi_event_interaction(
        self,
        events: List[KPopEvent],
        metrics_data: pd.DataFrame,
        interaction_window_days: int = 30
    ) -> Dict[str, Any]:
        """다중 이벤트 상호작용 분석"""
        try:
            self.logger.info(f"Analyzing multi-event interactions for {len(events)} events")
            
            # 이벤트 시간순 정렬
            sorted_events = sorted(events, key=lambda e: e.event_date)
            
            interactions = []
            
            # 연속된 이벤트 쌍별 상호작용 분석
            for i in range(len(sorted_events) - 1):
                event1 = sorted_events[i]
                event2 = sorted_events[i + 1]
                
                # 시간 간격 계산
                time_gap = (event2.event_date - event1.event_date).days
                
                if time_gap <= interaction_window_days:
                    interaction = self._analyze_event_pair_interaction(
                        event1, event2, metrics_data, time_gap
                    )
                    interactions.append(interaction)
            
            # 전체 상호작용 패턴 분석
            interaction_analysis = {
                "events_analyzed": len(sorted_events),
                "interaction_pairs": len(interactions),
                "interactions": interactions,
                "overall_patterns": self._analyze_overall_interaction_patterns(interactions),
                "synergy_effects": self._identify_synergy_effects(interactions),
                "interference_effects": self._identify_interference_effects(interactions),
                "optimal_spacing": self._calculate_optimal_event_spacing(interactions),
                "recommendations": self._generate_interaction_recommendations(interactions)
            }
            
            self.logger.info(f"Multi-event interaction analysis completed")
            return interaction_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing multi-event interactions: {e}")
            return {"error": str(e)}
    
    def predict_event_impact(
        self,
        upcoming_event: KPopEvent,
        historical_data: pd.DataFrame,
        similar_events: List[EventImpactAnalysis] = None
    ) -> Dict[str, Any]:
        """이벤트 영향도 예측"""
        try:
            self.logger.info(f"Predicting impact for upcoming event: {upcoming_event.name}")
            
            if similar_events is None:
                similar_events = self._find_similar_events(upcoming_event)
            
            if not similar_events:
                return {"error": "No similar events found for prediction"}
            
            # 예측 모델 생성
            prediction_model = self._build_prediction_model(similar_events)
            
            # 기본 예측
            base_prediction = self._calculate_base_prediction(upcoming_event, similar_events)
            
            # 맥락적 조정
            contextual_adjustments = self._apply_contextual_adjustments(upcoming_event, historical_data)
            
            # 최종 예측
            final_prediction = self._combine_predictions(base_prediction, contextual_adjustments)
            
            # 불확실성 범위 계산
            uncertainty_range = self._calculate_prediction_uncertainty(similar_events)
            
            prediction_result = {
                "event_id": upcoming_event.event_id,
                "event_name": upcoming_event.name,
                "event_date": upcoming_event.event_date.isoformat(),
                "prediction_timestamp": datetime.now().isoformat(),
                
                "predicted_impacts": {
                    "immediate_impact": final_prediction.get("immediate", 0.0),
                    "short_term_impact": final_prediction.get("short_term", 0.0),
                    "medium_term_impact": final_prediction.get("medium_term", 0.0),
                    "long_term_impact": final_prediction.get("long_term", 0.0)
                },
                
                "overall_prediction": {
                    "impact_score": final_prediction.get("overall_score", 0.0),
                    "sustainability_score": final_prediction.get("sustainability", 0.0),
                    "confidence_level": final_prediction.get("confidence", 0.5)
                },
                
                "uncertainty_analysis": {
                    "prediction_range": uncertainty_range,
                    "key_risk_factors": self._identify_prediction_risks(upcoming_event),
                    "sensitivity_factors": self._identify_sensitivity_factors(similar_events)
                },
                
                "comparative_analysis": {
                    "similar_events_count": len(similar_events),
                    "historical_performance_range": self._calculate_historical_range(similar_events),
                    "category_average": self._calculate_category_average(upcoming_event.category, similar_events)
                },
                
                "recommendations": {
                    "optimization_suggestions": self._generate_optimization_suggestions(upcoming_event, final_prediction),
                    "timing_considerations": self._analyze_timing_considerations(upcoming_event),
                    "risk_mitigation": self._suggest_risk_mitigation(upcoming_event, uncertainty_range)
                }
            }
            
            self.logger.info(f"Impact prediction completed: predicted overall score {final_prediction.get('overall_score', 0):.3f}")
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error predicting event impact: {e}")
            return {"error": str(e)}
    
    def generate_impact_report(
        self,
        artist_id: str,
        analysis_period: Tuple[date, date],
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """종합 이벤트 영향도 보고서 생성"""
        try:
            self.logger.info(f"Generating impact report for artist {artist_id}")
            
            # 해당 기간 이벤트 분석 결과 수집
            period_analyses = [
                analysis for analysis in self.impact_analyses.values()
                if (analysis.artist_id == artist_id and
                    analysis_period[0] <= analysis.event_date <= analysis_period[1])
            ]
            
            if not period_analyses:
                return {"error": f"No impact analyses found for artist {artist_id} in the specified period"}
            
            # 보고서 구조 생성
            impact_report = {
                "artist_id": artist_id,
                "analysis_period": [analysis_period[0].isoformat(), analysis_period[1].isoformat()],
                "report_generated": datetime.now().isoformat(),
                
                "executive_summary": {
                    "total_events_analyzed": len(period_analyses),
                    "average_impact_score": np.mean([a.overall_impact_score for a in period_analyses]),
                    "highest_impact_event": self._find_highest_impact_event(period_analyses),
                    "most_sustainable_impact": self._find_most_sustainable_event(period_analyses),
                    "key_insights": self._generate_executive_insights(period_analyses)
                },
                
                "detailed_analysis": {
                    "event_breakdown": [analysis.to_dict() for analysis in period_analyses],
                    "category_performance": self._analyze_category_performance(period_analyses),
                    "temporal_trends": self._analyze_temporal_trends(period_analyses),
                    "impact_patterns": self._identify_impact_patterns(period_analyses)
                },
                
                "performance_metrics": {
                    "impact_distribution": self._calculate_impact_distribution(period_analyses),
                    "sustainability_analysis": self._analyze_sustainability_patterns(period_analyses),
                    "statistical_summary": self._generate_statistical_summary(period_analyses)
                },
                
                "strategic_recommendations": {
                    "event_optimization": self._recommend_event_optimization(period_analyses),
                    "timing_strategies": self._recommend_timing_strategies(period_analyses),
                    "impact_maximization": self._recommend_impact_maximization(period_analyses)
                }
            }
            
            # 예측 정보 포함 (선택적)
            if include_predictions:
                impact_report["future_predictions"] = self._generate_future_predictions(artist_id, period_analyses)
            
            self.logger.info(f"Impact report generated successfully for artist {artist_id}")
            return impact_report
            
        except Exception as e:
            self.logger.error(f"Error generating impact report: {e}")
            return {"error": str(e)}
    
    def _get_impact_windows(self, event_category: EventCategory) -> Dict[ImpactType, int]:
        """이벤트 카테고리별 영향 기간 조회"""
        return self.impact_windows.get(event_category, self.default_impact_windows)
    
    def _analyze_impact_window(
        self,
        metrics_data: pd.DataFrame,
        baseline_start: pd.Timestamp,
        baseline_end: pd.Timestamp,
        impact_start: pd.Timestamp,
        impact_end: pd.Timestamp,
        window_type: str
    ) -> List[ImpactMeasurement]:
        """특정 시간 윈도우의 영향도 분석"""
        try:
            measurements = []
            
            # 기준선 데이터
            baseline_data = metrics_data[
                (metrics_data.index >= baseline_start) & (metrics_data.index <= baseline_end)
            ]
            
            # 영향 기간 데이터
            impact_data = metrics_data[
                (metrics_data.index >= impact_start) & (metrics_data.index <= impact_end)
            ]
            
            if baseline_data.empty or impact_data.empty:
                return measurements
            
            # 각 지표별 영향도 측정
            numeric_columns = metrics_data.select_dtypes(include=[np.number]).columns
            
            for metric in numeric_columns:
                if metric in ['artist_id', 'event_id']:
                    continue
                
                baseline_values = baseline_data[metric].dropna()
                impact_values = impact_data[metric].dropna()
                
                if len(baseline_values) == 0 or len(impact_values) == 0:
                    continue
                
                # 기본 통계
                baseline_mean = baseline_values.mean()
                impact_peak = impact_values.max()
                
                # 영향 퍼센트 계산
                if baseline_mean != 0:
                    impact_percentage = ((impact_peak - baseline_mean) / baseline_mean) * 100
                else:
                    impact_percentage = 0.0
                
                # 영향 크기 분류
                magnitude = self._classify_impact_magnitude(abs(impact_percentage))
                
                # 영향 방향 결정
                direction = ImpactDirection.POSITIVE if impact_percentage > 0 else ImpactDirection.NEGATIVE
                if abs(impact_percentage) < 1.0:
                    direction = ImpactDirection.NEUTRAL
                
                # 통계적 유의성 검정
                if len(baseline_values) > 1 and len(impact_values) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(impact_values, baseline_values)
                    except:
                        p_value = 1.0
                else:
                    p_value = 1.0
                
                # 신뢰구간 계산 (간소화)
                std_error = np.std(impact_values) / np.sqrt(len(impact_values)) if len(impact_values) > 0 else 0
                margin_error = 1.96 * std_error
                confidence_interval = (impact_percentage - margin_error, impact_percentage + margin_error)
                
                measurement = ImpactMeasurement(
                    metric_name=metric,
                    baseline_value=float(baseline_mean),
                    peak_value=float(impact_peak),
                    impact_percentage=float(impact_percentage),
                    impact_magnitude=magnitude,
                    impact_direction=direction,
                    statistical_significance=float(p_value),
                    confidence_interval=confidence_interval
                )
                
                measurements.append(measurement)
            
            return measurements
            
        except Exception as e:
            self.logger.error(f"Error analyzing impact window: {e}")
            return []
    
    def _classify_impact_magnitude(self, abs_percentage: float) -> ImpactMagnitude:
        """영향 크기 분류"""
        if abs_percentage < 5:
            return ImpactMagnitude.NEGLIGIBLE
        elif abs_percentage < 15:
            return ImpactMagnitude.SMALL
        elif abs_percentage < 30:
            return ImpactMagnitude.MODERATE
        elif abs_percentage < 50:
            return ImpactMagnitude.LARGE
        else:
            return ImpactMagnitude.MASSIVE
    
    def _calculate_overall_impact_score(self, analysis: EventImpactAnalysis) -> float:
        """종합 영향도 점수 계산"""
        try:
            all_impacts = (
                analysis.immediate_impacts + 
                analysis.short_term_impacts + 
                analysis.medium_term_impacts + 
                analysis.long_term_impacts
            )
            
            if not all_impacts:
                return 0.0
            
            # 가중 평균 계산
            weights = {
                'immediate': 0.4,
                'short_term': 0.3,
                'medium_term': 0.2,
                'long_term': 0.1
            }
            
            weighted_scores = []
            
            # 각 시기별 평균 영향도
            for impact_list, weight_key in [
                (analysis.immediate_impacts, 'immediate'),
                (analysis.short_term_impacts, 'short_term'),
                (analysis.medium_term_impacts, 'medium_term'),
                (analysis.long_term_impacts, 'long_term')
            ]:
                if impact_list:
                    avg_impact = np.mean([abs(imp.impact_percentage) for imp in impact_list])
                    # 0-1 범위로 정규화
                    normalized_impact = min(avg_impact / 100.0, 1.0)
                    weighted_scores.append(normalized_impact * weights[weight_key])
            
            return sum(weighted_scores) if weighted_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall impact score: {e}")
            return 0.0
    
    def _calculate_sustainability_score(self, analysis: EventImpactAnalysis) -> float:
        """영향 지속성 점수 계산"""
        try:
            # 시간대별 영향 크기 비교
            impact_by_period = {
                'immediate': analysis.immediate_impacts,
                'short_term': analysis.short_term_impacts,
                'medium_term': analysis.medium_term_impacts,
                'long_term': analysis.long_term_impacts
            }
            
            period_scores = []
            for period, impacts in impact_by_period.items():
                if impacts:
                    avg_impact = np.mean([abs(imp.impact_percentage) for imp in impacts])
                    period_scores.append(avg_impact)
                else:
                    period_scores.append(0.0)
            
            if len(period_scores) < 2:
                return 0.5  # 기본값
            
            # 지속성 = 시간이 지나도 영향이 유지되는 정도
            # 초기 영향 대비 장기 영향의 비율
            if period_scores[0] > 0:  # 즉시 영향이 있는 경우
                sustainability = np.mean(period_scores[1:]) / period_scores[0]
                return min(sustainability, 1.0)
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating sustainability score: {e}")
            return 0.5
    
    def _calculate_analysis_confidence(
        self, 
        metrics_data: pd.DataFrame, 
        analysis: EventImpactAnalysis
    ) -> float:
        """분석 신뢰도 계산"""
        try:
            # 데이터 품질 점수
            data_completeness = 1.0 - (metrics_data.isna().sum().sum() / metrics_data.size)
            
            # 통계적 유의성 점수
            all_impacts = (
                analysis.immediate_impacts + analysis.short_term_impacts + 
                analysis.medium_term_impacts + analysis.long_term_impacts
            )
            
            if all_impacts:
                significant_impacts = [
                    imp for imp in all_impacts 
                    if imp.statistical_significance < self.significance_threshold
                ]
                significance_ratio = len(significant_impacts) / len(all_impacts)
            else:
                significance_ratio = 0.0
            
            # 전체 신뢰도
            confidence = (data_completeness * 0.6 + significance_ratio * 0.4)
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    def _analyze_spillover_effects(
        self,
        event: KPopEvent,
        metrics_data: pd.DataFrame,
        analysis: EventImpactAnalysis
    ) -> Dict[str, float]:
        """파급 효과 분석"""
        spillover_effects = {}
        
        # 플랫폼간 파급 효과 (간소화)
        if analysis.immediate_impacts:
            primary_impact = max(analysis.immediate_impacts, key=lambda x: abs(x.impact_percentage))
            spillover_effects["cross_platform"] = abs(primary_impact.impact_percentage) * 0.3
        
        # 동종 업계 파급 효과
        if event.category == EventCategory.AWARD_SHOW:
            spillover_effects["industry_wide"] = analysis.overall_impact_score * 0.5
        
        return spillover_effects
    
    def _identify_contextual_factors(self, event: KPopEvent, metrics_data: pd.DataFrame) -> List[str]:
        """맥락 요인 식별"""
        factors = []
        
        # 시기적 요인
        month = event.event_date.month
        if month in [11, 12, 1]:
            factors.append("시상식 시즌")
        elif month in [7, 8]:
            factors.append("여름 페스티벌 시즌")
        
        # 이벤트 유형별 요인
        if event.category == EventCategory.COMEBACK:
            factors.append("컴백 이벤트")
        elif event.category == EventCategory.AWARD_SHOW:
            factors.append("시상식 참여")
        
        return factors
    
    # 비교 분석 관련 메서드들
    def _perform_statistical_comparison(self, event_analyses: Dict[str, EventImpactAnalysis]) -> Dict[str, Any]:
        """통계적 비교 수행"""
        impact_scores = [analysis.overall_impact_score for analysis in event_analyses.values()]
        
        return {
            "mean_impact": float(np.mean(impact_scores)),
            "median_impact": float(np.median(impact_scores)),
            "std_deviation": float(np.std(impact_scores)),
            "min_impact": float(np.min(impact_scores)),
            "max_impact": float(np.max(impact_scores)),
            "variance": float(np.var(impact_scores))
        }
    
    def _analyze_by_category(self, event_analyses: Dict[str, EventImpactAnalysis]) -> Dict[str, Any]:
        """카테고리별 분석"""
        category_stats = defaultdict(list)
        
        for analysis in event_analyses.values():
            category_stats[analysis.event_category.value].append(analysis.overall_impact_score)
        
        category_analysis = {}
        for category, scores in category_stats.items():
            category_analysis[category] = {
                "count": len(scores),
                "average_impact": float(np.mean(scores)),
                "max_impact": float(np.max(scores)),
                "min_impact": float(np.min(scores))
            }
        
        return category_analysis
    
    def _analyze_temporal_patterns(self, event_analyses: Dict[str, EventImpactAnalysis]) -> Dict[str, Any]:
        """시간적 패턴 분석"""
        monthly_impacts = defaultdict(list)
        
        for analysis in event_analyses.values():
            month = analysis.event_date.month
            monthly_impacts[month].append(analysis.overall_impact_score)
        
        temporal_patterns = {}
        for month, impacts in monthly_impacts.items():
            temporal_patterns[f"month_{month}"] = {
                "average_impact": float(np.mean(impacts)),
                "event_count": len(impacts)
            }
        
        return temporal_patterns
    
    def _generate_comparison_insights(
        self, 
        event_analyses: Dict[str, EventImpactAnalysis], 
        comparison_result: Dict[str, Any]
    ) -> List[str]:
        """비교 분석 인사이트 생성"""
        insights = []
        
        # 최고 성과 이벤트
        if comparison_result["rankings"].get("overall_impact_score"):
            top_event = comparison_result["rankings"]["overall_impact_score"][0]
            insights.append(f"최고 영향도: {top_event['event_name']} ({top_event['value']:.3f})")
        
        # 카테고리별 성과
        category_analysis = comparison_result.get("category_analysis", {})
        if category_analysis:
            best_category = max(category_analysis.items(), key=lambda x: x[1]["average_impact"])
            insights.append(f"최고 성과 카테고리: {best_category[0]} (평균 {best_category[1]['average_impact']:.3f})")
        
        return insights
    
    # 다중 이벤트 상호작용 관련 메서드들 (간소화)
    def _analyze_event_pair_interaction(
        self, 
        event1: KPopEvent, 
        event2: KPopEvent, 
        metrics_data: pd.DataFrame, 
        time_gap: int
    ) -> Dict[str, Any]:
        """이벤트 쌍 상호작용 분석"""
        return {
            "event1_id": event1.event_id,
            "event2_id": event2.event_id,
            "time_gap_days": time_gap,
            "interaction_type": "sequential",
            "synergy_score": 0.5,  # 실제로는 복잡한 계산 필요
            "interference_score": 0.2
        }
    
    def _analyze_overall_interaction_patterns(self, interactions: List[Dict]) -> Dict[str, Any]:
        """전체 상호작용 패턴 분석"""
        return {
            "average_synergy": np.mean([i["synergy_score"] for i in interactions]) if interactions else 0.0,
            "average_interference": np.mean([i["interference_score"] for i in interactions]) if interactions else 0.0
        }
    
    def _identify_synergy_effects(self, interactions: List[Dict]) -> List[Dict[str, Any]]:
        """시너지 효과 식별"""
        return [i for i in interactions if i["synergy_score"] > 0.7]
    
    def _identify_interference_effects(self, interactions: List[Dict]) -> List[Dict[str, Any]]:
        """간섭 효과 식별"""
        return [i for i in interactions if i["interference_score"] > 0.5]
    
    def _calculate_optimal_event_spacing(self, interactions: List[Dict]) -> Dict[str, Any]:
        """최적 이벤트 간격 계산"""
        if not interactions:
            return {"optimal_days": 30, "confidence": 0.5}
        
        # 간소화된 계산
        high_synergy_gaps = [i["time_gap_days"] for i in interactions if i["synergy_score"] > 0.6]
        
        if high_synergy_gaps:
            optimal_gap = int(np.mean(high_synergy_gaps))
        else:
            optimal_gap = 30  # 기본값
        
        return {
            "optimal_days": optimal_gap,
            "confidence": 0.7,
            "recommended_range": (optimal_gap - 7, optimal_gap + 7)
        }
    
    def _generate_interaction_recommendations(self, interactions: List[Dict]) -> List[str]:
        """상호작용 분석 기반 추천사항 생성"""
        recommendations = []
        
        if not interactions:
            recommendations.append("단일 이벤트 집중 전략 권장")
            return recommendations
        
        avg_synergy = np.mean([i["synergy_score"] for i in interactions])
        
        if avg_synergy > 0.6:
            recommendations.append("이벤트 연계 시너지 효과 높음 - 연속 이벤트 전략 권장")
        else:
            recommendations.append("이벤트 간격 조정을 통한 간섭 최소화 필요")
        
        return recommendations
    
    # 예측 관련 메서드들 (간소화)
    def _find_similar_events(self, upcoming_event: KPopEvent) -> List[EventImpactAnalysis]:
        """유사 이벤트 찾기"""
        similar_events = []
        
        for analysis in self.impact_analyses.values():
            if analysis.event_category == upcoming_event.category:
                similar_events.append(analysis)
        
        return similar_events[-5:]  # 최근 5개만 반환
    
    def _build_prediction_model(self, similar_events: List[EventImpactAnalysis]) -> Dict[str, Any]:
        """예측 모델 구축"""
        return {
            "model_type": "historical_average",
            "training_data_size": len(similar_events),
            "confidence": 0.7
        }
    
    def _calculate_base_prediction(
        self, 
        upcoming_event: KPopEvent, 
        similar_events: List[EventImpactAnalysis]
    ) -> Dict[str, float]:
        """기본 예측 계산"""
        if not similar_events:
            return {"overall_score": 0.5, "sustainability": 0.5}
        
        avg_impact = np.mean([e.overall_impact_score for e in similar_events])
        avg_sustainability = np.mean([e.impact_sustainability_score for e in similar_events])
        
        return {
            "overall_score": float(avg_impact),
            "sustainability": float(avg_sustainability),
            "immediate": float(avg_impact * 1.2),
            "short_term": float(avg_impact * 1.0),
            "medium_term": float(avg_impact * 0.8),
            "long_term": float(avg_impact * 0.6)
        }
    
    def _apply_contextual_adjustments(
        self, 
        upcoming_event: KPopEvent, 
        historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """맥락적 조정 적용"""
        adjustments = {"multiplier": 1.0}
        
        # 시기적 조정
        month = upcoming_event.event_date.month
        if month in [11, 12, 1]:  # 시상식 시즌
            adjustments["multiplier"] *= 1.2
        elif month in [6, 7, 8]:  # 여름 시즌
            adjustments["multiplier"] *= 1.1
        
        return adjustments
    
    def _combine_predictions(
        self, 
        base_prediction: Dict[str, float], 
        contextual_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """예측 결합"""
        multiplier = contextual_adjustments.get("multiplier", 1.0)
        
        combined = {}
        for key, value in base_prediction.items():
            combined[key] = value * multiplier
        
        combined["confidence"] = 0.7  # 기본 신뢰도
        
        return combined
    
    def _calculate_prediction_uncertainty(self, similar_events: List[EventImpactAnalysis]) -> Dict[str, float]:
        """예측 불확실성 계산"""
        if not similar_events:
            return {"low": 0.3, "high": 0.7}
        
        impacts = [e.overall_impact_score for e in similar_events]
        std_dev = np.std(impacts)
        mean_impact = np.mean(impacts)
        
        return {
            "low": max(0.0, mean_impact - 1.96 * std_dev),
            "high": min(1.0, mean_impact + 1.96 * std_dev),
            "standard_deviation": float(std_dev)
        }
    
    # 보고서 생성 관련 메서드들 (간소화)
    def _find_highest_impact_event(self, analyses: List[EventImpactAnalysis]) -> Dict[str, Any]:
        """최고 영향도 이벤트 찾기"""
        if not analyses:
            return {}
        
        highest = max(analyses, key=lambda x: x.overall_impact_score)
        return {
            "event_name": highest.event_name,
            "impact_score": highest.overall_impact_score,
            "date": highest.event_date.isoformat()
        }
    
    def _find_most_sustainable_event(self, analyses: List[EventImpactAnalysis]) -> Dict[str, Any]:
        """가장 지속가능한 영향 이벤트 찾기"""
        if not analyses:
            return {}
        
        most_sustainable = max(analyses, key=lambda x: x.impact_sustainability_score)
        return {
            "event_name": most_sustainable.event_name,
            "sustainability_score": most_sustainable.impact_sustainability_score,
            "date": most_sustainable.event_date.isoformat()
        }
    
    def _generate_executive_insights(self, analyses: List[EventImpactAnalysis]) -> List[str]:
        """경영진 인사이트 생성"""
        insights = []
        
        if analyses:
            avg_impact = np.mean([a.overall_impact_score for a in analyses])
            if avg_impact > 0.7:
                insights.append("전반적으로 높은 이벤트 영향도 달성")
            elif avg_impact < 0.3:
                insights.append("이벤트 효과 개선이 필요한 상황")
        
        return insights
    
    # 추가 헬퍼 메서드들
    def _identify_prediction_risks(self, upcoming_event: KPopEvent) -> List[str]:
        return ["시장 경쟁 심화", "팬덤 피로도", "외부 환경 변화"]
    
    def _identify_sensitivity_factors(self, similar_events: List[EventImpactAnalysis]) -> List[str]:
        return ["이벤트 타이밍", "마케팅 예산", "경쟁 상황"]
    
    def _calculate_historical_range(self, similar_events: List[EventImpactAnalysis]) -> Dict[str, float]:
        if not similar_events:
            return {"min": 0.0, "max": 1.0}
        
        impacts = [e.overall_impact_score for e in similar_events]
        return {"min": float(np.min(impacts)), "max": float(np.max(impacts))}
    
    def _calculate_category_average(self, category: EventCategory, similar_events: List[EventImpactAnalysis]) -> float:
        category_events = [e for e in similar_events if e.event_category == category]
        if not category_events:
            return 0.5
        
        return float(np.mean([e.overall_impact_score for e in category_events]))
    
    def _generate_optimization_suggestions(self, event: KPopEvent, prediction: Dict[str, float]) -> List[str]:
        return ["마케팅 예산 최적화", "타이밍 조정 검토", "팬 참여 이벤트 기획"]
    
    def _analyze_timing_considerations(self, event: KPopEvent) -> List[str]:
        return ["경쟁 이벤트 회피", "시즌 특성 활용", "팬덤 활동 주기 고려"]
    
    def _suggest_risk_mitigation(self, event: KPopEvent, uncertainty: Dict[str, float]) -> List[str]:
        return ["다양한 시나리오 준비", "백업 계획 수립", "실시간 모니터링 체계 구축"]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        return {
            "total_impact_analyses": len(self.impact_analyses),
            "average_impact_score": np.mean([
                analysis.overall_impact_score for analysis in self.impact_analyses.values()
            ]) if self.impact_analyses else 0.0,
            "categories_analyzed": list(set([
                analysis.event_category.value for analysis in self.impact_analyses.values()
            ])),
            "analysis_period_coverage": {
                "earliest": min([a.event_date for a in self.impact_analyses.values()]).isoformat() if self.impact_analyses else None,
                "latest": max([a.event_date for a in self.impact_analyses.values()]).isoformat() if self.impact_analyses else None
            },
            "system_initialized": datetime.now().isoformat()
        }