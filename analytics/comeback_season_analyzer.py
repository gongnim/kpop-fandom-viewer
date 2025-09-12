"""
K-Pop 컴백 시즌 패턴 분석기
===========================

K-Pop 아티스트들의 컴백 패턴을 분석하고 최적 타이밍을 예측하는 시스템입니다.
시즌별 경쟁도, 성과 패턴, 업계 트렌드를 종합적으로 분석합니다.

주요 기능:
- 시즌별 컴백 밀도 분석
- 경쟁도 기반 최적 타이밍 예측
- 과거 성과 패턴 분석
- 시상식 연계 전략 수립

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
from collections import defaultdict, Counter
import logging
import json

from .kpop_event_calendar import ComebackSeason, EventCategory, KPopEvent, EventImportance
from .award_shows_data import AwardShow

# Configure logging
logger = logging.getLogger(__name__)


class CompetitionLevel(Enum):
    """경쟁도 수준"""
    VERY_LOW = "very_low"      # 매우 낮음 (0-2개 컴백)
    LOW = "low"                # 낮음 (3-5개 컴백)  
    MODERATE = "moderate"      # 보통 (6-10개 컴백)
    HIGH = "high"              # 높음 (11-15개 컴백)
    VERY_HIGH = "very_high"    # 매우 높음 (16-20개 컴백)
    EXTREME = "extreme"        # 극도로 높음 (20개 이상)


class PerformanceMetric(Enum):
    """성과 측정 지표"""
    STREAMING_GROWTH = "streaming_growth"
    FOLLOWER_INCREASE = "follower_increase"
    CHART_PERFORMANCE = "chart_performance"
    SOCIAL_ENGAGEMENT = "social_engagement"
    AWARD_NOMINATIONS = "award_nominations"
    MEDIA_COVERAGE = "media_coverage"


@dataclass
class ComebackAnalysis:
    """컴백 분석 결과"""
    artist_id: str
    comeback_date: date
    season: ComebackSeason
    competition_level: CompetitionLevel
    
    # 성과 지표
    performance_metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    overall_performance_score: float = 0.0
    
    # 컨텍스트 정보
    concurrent_comebacks: List[str] = field(default_factory=list)  # 동시기 컴백 아티스트
    major_events_nearby: List[str] = field(default_factory=list)   # 인근 주요 이벤트
    
    # 메타데이터
    analysis_date: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'artist_id': self.artist_id,
            'comeback_date': self.comeback_date.isoformat(),
            'season': self.season.value,
            'competition_level': self.competition_level.value,
            'performance_metrics': {k.value: v for k, v in self.performance_metrics.items()},
            'overall_performance_score': self.overall_performance_score,
            'concurrent_comebacks': self.concurrent_comebacks,
            'major_events_nearby': self.major_events_nearby,
            'analysis_date': self.analysis_date.isoformat(),
            'confidence_score': self.confidence_score
        }


@dataclass
class SeasonalPattern:
    """시즌별 패턴 분석 결과"""
    season: ComebackSeason
    period_analyzed: Tuple[date, date]
    
    # 통계 정보
    total_comebacks: int = 0
    average_competition_level: str = ""
    success_rate: float = 0.0
    
    # 성과 패턴
    average_performance: float = 0.0
    performance_variance: float = 0.0
    top_performers: List[str] = field(default_factory=list)
    
    # 타이밍 분석
    optimal_timing_windows: List[Dict[str, Any]] = field(default_factory=list)
    peak_weeks: List[int] = field(default_factory=list)  # 주차별 분석
    
    # 특성 분석
    genre_preferences: Dict[str, int] = field(default_factory=dict)
    concept_trends: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'season': self.season.value,
            'period_analyzed': [self.period_analyzed[0].isoformat(), self.period_analyzed[1].isoformat()],
            'total_comebacks': self.total_comebacks,
            'average_competition_level': self.average_competition_level,
            'success_rate': self.success_rate,
            'average_performance': self.average_performance,
            'performance_variance': self.performance_variance,
            'top_performers': self.top_performers,
            'optimal_timing_windows': self.optimal_timing_windows,
            'peak_weeks': self.peak_weeks,
            'genre_preferences': self.genre_preferences,
            'concept_trends': self.concept_trends
        }


class ComebackSeasonAnalyzer:
    """
    K-Pop 컴백 시즌 패턴 분석기
    
    컴백 타이밍 최적화와 시즌별 전략 수립을 위한 고급 분석 도구입니다.
    """
    
    def __init__(self, database_connection=None):
        """컴백 시즌 분석기 초기화"""
        self.db_conn = database_connection
        self.logger = logger
        
        # 분석 데이터 저장소
        self.comeback_analyses: Dict[str, ComebackAnalysis] = {}
        self.seasonal_patterns: Dict[ComebackSeason, SeasonalPattern] = {}
        
        # 시즌별 특성 정의
        self.season_characteristics = self._initialize_season_characteristics()
        
        # 경쟁도 임계값 설정
        self.competition_thresholds = {
            CompetitionLevel.VERY_LOW: (0, 2),
            CompetitionLevel.LOW: (3, 5),
            CompetitionLevel.MODERATE: (6, 10),
            CompetitionLevel.HIGH: (11, 15),
            CompetitionLevel.VERY_HIGH: (16, 20),
            CompetitionLevel.EXTREME: (21, float('inf'))
        }
        
        self.logger.info("Comeback Season Analyzer initialized")
    
    def _initialize_season_characteristics(self) -> Dict[ComebackSeason, Dict[str, Any]]:
        """시즌별 특성 초기화"""
        return {
            ComebackSeason.SPRING: {
                'months': [3, 4, 5],
                'typical_concepts': ['청량', '로맨틱', '밝은'],
                'fan_activity_level': 0.8,
                'media_attention': 0.7,
                'streaming_preference': 'moderate',
                'award_show_proximity': 0.3,
                'school_season_impact': 0.9,  # 신학기 영향
                'weather_correlation': 'positive'
            },
            ComebackSeason.SUMMER: {
                'months': [6, 7, 8],
                'typical_concepts': ['청량', '트로피컬', '파티'],
                'fan_activity_level': 0.9,  # 방학 시즌
                'media_attention': 0.8,
                'streaming_preference': 'high',
                'award_show_proximity': 0.2,
                'school_season_impact': 0.3,
                'weather_correlation': 'positive'
            },
            ComebackSeason.AUTUMN: {
                'months': [9, 10, 11],
                'typical_concepts': ['성숙', '감성', '다양'],
                'fan_activity_level': 0.85,
                'media_attention': 0.9,  # 시상식 준비
                'streaming_preference': 'high',
                'award_show_proximity': 0.8,
                'school_season_impact': 0.7,
                'weather_correlation': 'neutral'
            },
            ComebackSeason.WINTER: {
                'months': [12, 1, 2],
                'typical_concepts': ['발라드', '감성', '따뜻한'],
                'fan_activity_level': 0.7,  # 연말연시 외출 감소
                'media_attention': 1.0,     # 시상식 시즌
                'streaming_preference': 'moderate',
                'award_show_proximity': 1.0,
                'school_season_impact': 0.4,
                'weather_correlation': 'negative'
            }
        }
    
    def analyze_comeback_performance(
        self,
        artist_id: str,
        comeback_date: date,
        metrics_data: pd.DataFrame,
        baseline_days: int = 30,
        analysis_window: int = 60
    ) -> ComebackAnalysis:
        """개별 컴백 성과 분석"""
        try:
            self.logger.info(f"Analyzing comeback performance for artist {artist_id} on {comeback_date}")
            
            # 시즌 결정
            season = self._determine_season(comeback_date)
            
            # 경쟁도 분석
            competition_level = self._analyze_competition_level(comeback_date, artist_id)
            
            # 성과 지표 계산
            performance_metrics = self._calculate_performance_metrics(
                artist_id, comeback_date, metrics_data, baseline_days, analysis_window
            )
            
            # 전체 성과 점수 계산
            overall_score = self._calculate_overall_performance_score(performance_metrics)
            
            # 컨텍스트 정보 수집
            concurrent_comebacks = self._find_concurrent_comebacks(comeback_date, artist_id)
            major_events = self._find_nearby_major_events(comeback_date)
            
            # 신뢰도 점수 계산
            confidence = self._calculate_confidence_score(metrics_data, analysis_window)
            
            analysis = ComebackAnalysis(
                artist_id=artist_id,
                comeback_date=comeback_date,
                season=season,
                competition_level=competition_level,
                performance_metrics=performance_metrics,
                overall_performance_score=overall_score,
                concurrent_comebacks=concurrent_comebacks,
                major_events_nearby=major_events,
                confidence_score=confidence
            )
            
            # 분석 결과 저장
            analysis_key = f"{artist_id}_{comeback_date.isoformat()}"
            self.comeback_analyses[analysis_key] = analysis
            
            self.logger.info(f"Comeback analysis completed: overall score {overall_score:.3f}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing comeback performance: {e}")
            raise
    
    def analyze_seasonal_patterns(
        self,
        start_date: date,
        end_date: date,
        min_comebacks_threshold: int = 5
    ) -> Dict[ComebackSeason, SeasonalPattern]:
        """시즌별 패턴 종합 분석"""
        try:
            self.logger.info(f"Analyzing seasonal patterns from {start_date} to {end_date}")
            
            patterns = {}
            
            for season in ComebackSeason:
                if season == ComebackSeason.AWARD_SEASON or season == ComebackSeason.FESTIVAL_SEASON:
                    continue  # 특수 시즌은 별도 처리
                
                # 시즌별 컴백 데이터 필터링
                season_comebacks = self._filter_comebacks_by_season(
                    season, start_date, end_date
                )
                
                if len(season_comebacks) < min_comebacks_threshold:
                    self.logger.warning(f"Insufficient data for {season.value}: {len(season_comebacks)} comebacks")
                    continue
                
                # 패턴 분석 수행
                pattern = self._analyze_single_season_pattern(
                    season, season_comebacks, start_date, end_date
                )
                
                patterns[season] = pattern
                self.seasonal_patterns[season] = pattern
            
            self.logger.info(f"Seasonal pattern analysis completed for {len(patterns)} seasons")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in seasonal pattern analysis: {e}")
            return {}
    
    def predict_optimal_comeback_timing(
        self,
        artist_id: str,
        target_year: int,
        historical_data: Optional[pd.DataFrame] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """최적 컴백 타이밍 예측"""
        try:
            self.logger.info(f"Predicting optimal comeback timing for artist {artist_id} in {target_year}")
            
            # 아티스트 과거 성과 분석
            artist_history = self._analyze_artist_comeback_history(artist_id)
            
            # 업계 전체 패턴 분석
            industry_patterns = self._analyze_industry_patterns(target_year)
            
            # 시즌별 예측 점수 계산
            season_predictions = {}
            
            for season in [ComebackSeason.SPRING, ComebackSeason.SUMMER, 
                          ComebackSeason.AUTUMN, ComebackSeason.WINTER]:
                
                prediction = self._calculate_season_prediction_score(
                    season, artist_id, target_year, artist_history, industry_patterns, constraints
                )
                
                season_predictions[season.value] = prediction
            
            # 최적 추천 선정
            best_season = max(season_predictions.items(), key=lambda x: x[1]['total_score'])
            
            # 상세 타이밍 추천
            detailed_timing = self._generate_detailed_timing_recommendation(
                best_season[0], target_year, artist_history
            )
            
            prediction_result = {
                "artist_id": artist_id,
                "target_year": target_year,
                "season_predictions": season_predictions,
                "recommended_season": best_season[0],
                "recommended_score": best_season[1]['total_score'],
                "detailed_timing": detailed_timing,
                "confidence_level": self._calculate_prediction_confidence(artist_history),
                "key_factors": self._identify_key_prediction_factors(season_predictions),
                "risk_factors": self._identify_risk_factors(best_season[0], target_year),
                "generated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Optimal timing prediction completed: {best_season[0]} with score {best_season[1]['total_score']:.3f}")
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error predicting optimal comeback timing: {e}")
            return {"error": str(e)}
    
    def analyze_competition_landscape(
        self,
        analysis_date: date,
        window_days: int = 30,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """컴백 경쟁도 환경 분석"""
        try:
            self.logger.info(f"Analyzing competition landscape around {analysis_date}")
            
            # 분석 기간 설정
            start_date = analysis_date - timedelta(days=window_days // 2)
            end_date = analysis_date + timedelta(days=window_days // 2)
            
            # 해당 기간 컴백 데이터 수집
            period_comebacks = self._get_comebacks_in_period(start_date, end_date)
            
            # 경쟁도 지표 계산
            competition_metrics = {
                "total_comebacks": len(period_comebacks),
                "major_artists": self._count_major_artists(period_comebacks),
                "tier_distribution": self._analyze_artist_tiers(period_comebacks),
                "genre_diversity": self._analyze_genre_diversity(period_comebacks),
                "competition_intensity": self._calculate_competition_intensity(period_comebacks)
            }
            
            # 일간 밀도 분석
            daily_density = self._calculate_daily_comeback_density(period_comebacks, start_date, end_date)
            
            # 최적/최악 타이밍 식별
            optimal_windows = self._identify_optimal_windows(daily_density)
            high_risk_periods = self._identify_high_risk_periods(daily_density, period_comebacks)
            
            # 예측 정보 포함
            predictions = {}
            if include_predictions:
                predictions = self._predict_future_competition(end_date)
            
            landscape_analysis = {
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "focus_date": analysis_date.isoformat()
                },
                "competition_metrics": competition_metrics,
                "daily_density": daily_density,
                "optimal_windows": optimal_windows,
                "high_risk_periods": high_risk_periods,
                "strategic_recommendations": self._generate_competition_recommendations(
                    competition_metrics, optimal_windows
                ),
                "predictions": predictions,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Competition landscape analysis completed")
            return landscape_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing competition landscape: {e}")
            return {"error": str(e)}
    
    def generate_comeback_strategy_report(
        self,
        artist_id: str,
        target_comeback_date: date,
        strategy_objectives: List[str] = None
    ) -> Dict[str, Any]:
        """종합 컴백 전략 보고서 생성"""
        try:
            self.logger.info(f"Generating comeback strategy report for artist {artist_id}")
            
            if not strategy_objectives:
                strategy_objectives = ["chart_performance", "fan_engagement", "award_eligibility"]
            
            # 기본 분석
            season = self._determine_season(target_comeback_date)
            competition_analysis = self.analyze_competition_landscape(target_comeback_date)
            
            # 아티스트 특성 분석
            artist_profile = self._analyze_artist_profile(artist_id)
            
            # 전략별 분석
            strategy_analysis = {}
            for objective in strategy_objectives:
                strategy_analysis[objective] = self._analyze_strategy_objective(
                    objective, artist_id, target_comeback_date, season
                )
            
            # 타이밍 최적화 분석
            timing_analysis = self._analyze_timing_optimization(
                target_comeback_date, competition_analysis
            )
            
            # 리스크 평가
            risk_assessment = self._conduct_risk_assessment(
                artist_id, target_comeback_date, competition_analysis
            )
            
            # 액션 플랜 생성
            action_plan = self._generate_action_plan(
                artist_id, target_comeback_date, strategy_analysis, timing_analysis
            )
            
            strategy_report = {
                "artist_id": artist_id,
                "target_comeback_date": target_comeback_date.isoformat(),
                "season": season.value,
                "strategy_objectives": strategy_objectives,
                
                "executive_summary": {
                    "recommended_action": action_plan["primary_recommendation"],
                    "key_opportunities": action_plan["opportunities"],
                    "critical_risks": risk_assessment["critical_risks"],
                    "success_probability": self._calculate_success_probability(
                        strategy_analysis, timing_analysis, risk_assessment
                    )
                },
                
                "detailed_analysis": {
                    "artist_profile": artist_profile,
                    "competition_landscape": competition_analysis["competition_metrics"],
                    "strategy_analysis": strategy_analysis,
                    "timing_optimization": timing_analysis,
                    "risk_assessment": risk_assessment
                },
                
                "recommendations": {
                    "timing_adjustments": action_plan["timing_recommendations"],
                    "strategic_focus": action_plan["strategic_priorities"],
                    "preparation_timeline": action_plan["preparation_schedule"],
                    "contingency_plans": action_plan["contingency_options"]
                },
                
                "success_metrics": self._define_success_metrics(strategy_objectives),
                "generated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Comeback strategy report generated successfully")
            return strategy_report
            
        except Exception as e:
            self.logger.error(f"Error generating comeback strategy report: {e}")
            return {"error": str(e)}
    
    def _determine_season(self, comeback_date: date) -> ComebackSeason:
        """컴백 날짜로부터 시즌 결정"""
        month = comeback_date.month
        
        if month in [3, 4, 5]:
            return ComebackSeason.SPRING
        elif month in [6, 7, 8]:
            return ComebackSeason.SUMMER
        elif month in [9, 10, 11]:
            return ComebackSeason.AUTUMN
        else:  # 12, 1, 2
            return ComebackSeason.WINTER
    
    def _analyze_competition_level(self, comeback_date: date, artist_id: str) -> CompetitionLevel:
        """컴백 날짜의 경쟁도 분석"""
        # ±2주 내 컴백 수 계산
        start_date = comeback_date - timedelta(days=14)
        end_date = comeback_date + timedelta(days=14)
        
        concurrent_comebacks = self._get_comebacks_in_period(start_date, end_date, exclude_artist=artist_id)
        comeback_count = len(concurrent_comebacks)
        
        # 경쟁도 수준 결정
        for level, (min_val, max_val) in self.competition_thresholds.items():
            if min_val <= comeback_count <= max_val:
                return level
        
        return CompetitionLevel.MODERATE  # 기본값
    
    def _calculate_performance_metrics(
        self,
        artist_id: str,
        comeback_date: date,
        metrics_data: pd.DataFrame,
        baseline_days: int,
        analysis_window: int
    ) -> Dict[PerformanceMetric, float]:
        """성과 지표 계산"""
        try:
            comeback_datetime = pd.to_datetime(comeback_date)
            
            # 기준선 기간
            baseline_start = comeback_datetime - timedelta(days=baseline_days)
            baseline_end = comeback_datetime - timedelta(days=1)
            
            # 분석 기간
            analysis_start = comeback_datetime
            analysis_end = comeback_datetime + timedelta(days=analysis_window)
            
            # 데이터 필터링
            baseline_data = metrics_data[
                (metrics_data.index >= baseline_start) & (metrics_data.index <= baseline_end)
            ]
            analysis_data = metrics_data[
                (metrics_data.index >= analysis_start) & (metrics_data.index <= analysis_end)
            ]
            
            if baseline_data.empty or analysis_data.empty:
                return {}
            
            performance_metrics = {}
            
            # 스트리밍 성장률
            if 'streaming_count' in metrics_data.columns:
                baseline_streaming = baseline_data['streaming_count'].mean()
                peak_streaming = analysis_data['streaming_count'].max()
                if baseline_streaming > 0:
                    growth_rate = (peak_streaming - baseline_streaming) / baseline_streaming
                    performance_metrics[PerformanceMetric.STREAMING_GROWTH] = growth_rate
            
            # 팔로워 증가율
            follower_cols = [col for col in metrics_data.columns if 'followers' in col or 'subscribers' in col]
            if follower_cols:
                total_baseline = sum(baseline_data[col].mean() for col in follower_cols)
                total_peak = sum(analysis_data[col].max() for col in follower_cols)
                if total_baseline > 0:
                    follower_growth = (total_peak - total_baseline) / total_baseline
                    performance_metrics[PerformanceMetric.FOLLOWER_INCREASE] = follower_growth
            
            # 차트 성과 (간소화)
            if 'chart_position' in metrics_data.columns:
                best_position = analysis_data['chart_position'].min()  # 낮을수록 좋음
                chart_score = max(0, (100 - best_position) / 100)
                performance_metrics[PerformanceMetric.CHART_PERFORMANCE] = chart_score
            
            # 소셜 참여도 (간소화)
            engagement_cols = [col for col in metrics_data.columns if 'likes' in col or 'comments' in col]
            if engagement_cols:
                baseline_engagement = sum(baseline_data[col].mean() for col in engagement_cols)
                peak_engagement = sum(analysis_data[col].max() for col in engagement_cols)
                if baseline_engagement > 0:
                    engagement_growth = (peak_engagement - baseline_engagement) / baseline_engagement
                    performance_metrics[PerformanceMetric.SOCIAL_ENGAGEMENT] = engagement_growth
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_overall_performance_score(self, metrics: Dict[PerformanceMetric, float]) -> float:
        """전체 성과 점수 계산"""
        if not metrics:
            return 0.0
        
        # 가중치 설정
        weights = {
            PerformanceMetric.STREAMING_GROWTH: 0.3,
            PerformanceMetric.FOLLOWER_INCREASE: 0.25,
            PerformanceMetric.CHART_PERFORMANCE: 0.25,
            PerformanceMetric.SOCIAL_ENGAGEMENT: 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                # 값 정규화 (0-1 범위)
                normalized_value = max(0, min(value, 2.0)) / 2.0
                total_score += normalized_value * weights[metric]
                total_weight += weights[metric]
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _find_concurrent_comebacks(self, comeback_date: date, exclude_artist: str) -> List[str]:
        """동시기 컴백 아티스트 찾기 (±1주)"""
        start_date = comeback_date - timedelta(days=7)
        end_date = comeback_date + timedelta(days=7)
        
        concurrent = self._get_comebacks_in_period(start_date, end_date, exclude_artist)
        return [cb.get('artist_name', cb.get('artist_id', 'Unknown')) for cb in concurrent[:5]]  # 상위 5개만
    
    def _find_nearby_major_events(self, comeback_date: date, window_days: int = 14) -> List[str]:
        """인근 주요 이벤트 찾기"""
        # 실제 구현에서는 이벤트 데이터베이스 조회
        major_events = []
        
        # 시상식 시즌 체크
        month = comeback_date.month
        if month in [11, 12, 1]:
            major_events.append("시상식 시즌")
        
        # 특별 시기 체크
        if month == 12 and comeback_date.day > 20:
            major_events.append("연말 특집 시즌")
        
        return major_events
    
    def _calculate_confidence_score(self, metrics_data: pd.DataFrame, analysis_window: int) -> float:
        """신뢰도 점수 계산"""
        if metrics_data.empty:
            return 0.0
        
        # 데이터 완성도
        completeness = 1.0 - (metrics_data.isna().sum().sum() / metrics_data.size)
        
        # 데이터 기간의 적절성
        period_adequacy = min(len(metrics_data) / analysis_window, 1.0)
        
        # 전체 신뢰도
        confidence = (completeness * 0.6 + period_adequacy * 0.4)
        return min(confidence, 1.0)
    
    def _filter_comebacks_by_season(
        self, 
        season: ComebackSeason, 
        start_date: date, 
        end_date: date
    ) -> List[Dict[str, Any]]:
        """시즌별 컴백 데이터 필터링"""
        season_months = self.season_characteristics[season]['months']
        
        # 실제 구현에서는 데이터베이스에서 조회
        # 여기서는 모의 데이터 생성
        comebacks = []
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.month in season_months:
                # 모의 컴백 데이터
                if np.random.random() > 0.7:  # 30% 확률로 컴백
                    comebacks.append({
                        'artist_id': f'artist_{len(comebacks)}',
                        'artist_name': f'Artist {len(comebacks)}',
                        'comeback_date': current_date,
                        'season': season.value,
                        'performance_score': np.random.uniform(0.3, 0.9)
                    })
            current_date += timedelta(days=1)
        
        return comebacks
    
    def _analyze_single_season_pattern(
        self,
        season: ComebackSeason,
        comebacks: List[Dict[str, Any]],
        start_date: date,
        end_date: date
    ) -> SeasonalPattern:
        """단일 시즌 패턴 분석"""
        try:
            # 기본 통계
            total_comebacks = len(comebacks)
            performance_scores = [cb.get('performance_score', 0.5) for cb in comebacks]
            
            avg_performance = np.mean(performance_scores) if performance_scores else 0.0
            performance_variance = np.var(performance_scores) if performance_scores else 0.0
            
            # 성공률 계산 (성과 점수 0.6 이상을 성공으로 가정)
            successful_comebacks = [score for score in performance_scores if score >= 0.6]
            success_rate = len(successful_comebacks) / total_comebacks if total_comebacks > 0 else 0.0
            
            # 상위 수행자
            top_performers = []
            if comebacks:
                sorted_comebacks = sorted(comebacks, key=lambda x: x.get('performance_score', 0), reverse=True)
                top_performers = [cb['artist_name'] for cb in sorted_comebacks[:3]]
            
            # 최적 타이밍 윈도우 (간소화)
            optimal_windows = self._identify_optimal_timing_windows(comebacks, season)
            
            # 주차별 분석
            peak_weeks = self._analyze_weekly_patterns(comebacks, season)
            
            pattern = SeasonalPattern(
                season=season,
                period_analyzed=(start_date, end_date),
                total_comebacks=total_comebacks,
                average_competition_level=self._calculate_avg_competition_level(comebacks),
                success_rate=success_rate,
                average_performance=avg_performance,
                performance_variance=performance_variance,
                top_performers=top_performers,
                optimal_timing_windows=optimal_windows,
                peak_weeks=peak_weeks
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error analyzing season pattern for {season}: {e}")
            raise
    
    def _analyze_artist_comeback_history(self, artist_id: str) -> Dict[str, Any]:
        """아티스트 컴백 이력 분석"""
        # 실제 구현에서는 데이터베이스에서 해당 아티스트의 과거 컴백 데이터 조회
        return {
            'total_comebacks': 5,
            'average_performance': 0.7,
            'preferred_seasons': ['autumn', 'spring'],
            'best_performing_season': 'autumn',
            'typical_gap_months': 8,
            'success_rate_by_season': {
                'spring': 0.8,
                'summer': 0.6,
                'autumn': 0.9,
                'winter': 0.7
            }
        }
    
    def _analyze_industry_patterns(self, target_year: int) -> Dict[str, Any]:
        """업계 전체 패턴 분석"""
        return {
            'expected_total_comebacks': 150,
            'seasonal_distribution': {
                'spring': 0.25,
                'summer': 0.30,
                'autumn': 0.30,
                'winter': 0.15
            },
            'competition_forecast': {
                'spring': 'moderate',
                'summer': 'high',
                'autumn': 'very_high',
                'winter': 'moderate'
            }
        }
    
    def _calculate_season_prediction_score(
        self,
        season: ComebackSeason,
        artist_id: str,
        target_year: int,
        artist_history: Dict[str, Any],
        industry_patterns: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """시즌별 예측 점수 계산"""
        
        # 아티스트 과거 성과
        artist_season_performance = artist_history.get('success_rate_by_season', {}).get(season.value, 0.5)
        
        # 업계 경쟁도
        competition_forecast = industry_patterns.get('competition_forecast', {}).get(season.value, 'moderate')
        competition_penalty = {
            'low': 0.0, 'moderate': 0.1, 'high': 0.2, 'very_high': 0.3, 'extreme': 0.4
        }.get(competition_forecast, 0.1)
        
        # 시즌 특성
        season_characteristics = self.season_characteristics.get(season, {})
        season_bonus = season_characteristics.get('fan_activity_level', 0.5) * 0.2
        
        # 제약 조건 체크
        constraint_penalty = 0.0
        if constraints:
            # 예: 특정 시즌 선호/비선호
            if season.value in constraints.get('avoid_seasons', []):
                constraint_penalty = 0.5
            elif season.value in constraints.get('prefer_seasons', []):
                constraint_penalty = -0.2  # 보너스
        
        # 총 점수 계산
        total_score = (
            artist_season_performance * 0.4 +
            (1.0 - competition_penalty) * 0.3 +
            season_bonus * 0.2 +
            (1.0 - constraint_penalty) * 0.1
        )
        
        return {
            'total_score': round(total_score, 3),
            'artist_performance_factor': round(artist_season_performance, 3),
            'competition_penalty': round(competition_penalty, 3),
            'season_bonus': round(season_bonus, 3),
            'constraint_penalty': round(constraint_penalty, 3),
            'predicted_competition': competition_forecast
        }
    
    def _generate_detailed_timing_recommendation(
        self,
        recommended_season: str,
        target_year: int,
        artist_history: Dict[str, Any]
    ) -> Dict[str, Any]:
        """상세 타이밍 추천 생성"""
        season = ComebackSeason(recommended_season)
        season_months = self.season_characteristics[season]['months']
        
        # 최적 월 선택
        optimal_month = season_months[1]  # 시즌 중간 월 선택
        
        # 권장 주차 (월 중순 권장)
        recommended_weeks = [2, 3]  # 2-3주차
        
        return {
            'recommended_season': recommended_season,
            'optimal_months': season_months,
            'primary_month_recommendation': optimal_month,
            'recommended_weeks_in_month': recommended_weeks,
            'preparation_timeline': {
                'concept_planning': f"{optimal_month - 4}월",
                'recording_production': f"{optimal_month - 3}월",
                'marketing_preparation': f"{optimal_month - 2}월",
                'pre_release_promotion': f"{optimal_month - 1}월",
                'comeback_execution': f"{optimal_month}월"
            },
            'alternative_options': [
                {
                    'month': month,
                    'pros': f"{month}월의 장점",
                    'cons': f"{month}월의 단점"
                }
                for month in season_months if month != optimal_month
            ]
        }
    
    def _calculate_prediction_confidence(self, artist_history: Dict[str, Any]) -> str:
        """예측 신뢰도 계산"""
        total_comebacks = artist_history.get('total_comebacks', 0)
        
        if total_comebacks >= 5:
            return "높음"
        elif total_comebacks >= 3:
            return "보통"
        else:
            return "낮음"
    
    def _identify_key_prediction_factors(self, season_predictions: Dict) -> List[str]:
        """주요 예측 요인 식별"""
        factors = []
        
        # 가장 높은 점수를 받은 시즌의 요인들 분석
        best_season = max(season_predictions.items(), key=lambda x: x[1]['total_score'])
        best_prediction = best_season[1]
        
        if best_prediction['artist_performance_factor'] > 0.7:
            factors.append("아티스트 해당 시즌 과거 성과 우수")
        
        if best_prediction['competition_penalty'] < 0.2:
            factors.append("예상 경쟁도 낮음")
        
        if best_prediction['season_bonus'] > 0.15:
            factors.append("시즌 특성상 팬 활동 활발")
        
        return factors
    
    def _identify_risk_factors(self, recommended_season: str, target_year: int) -> List[str]:
        """리스크 요인 식별"""
        risk_factors = []
        
        season = ComebackSeason(recommended_season)
        
        if season == ComebackSeason.AUTUMN:
            risk_factors.append("시상식 시즌으로 경쟁 심화 예상")
        
        if season == ComebackSeason.SUMMER:
            risk_factors.append("여름 페스티벌 시즌으로 컴백 집중 예상")
        
        # 연도별 특수 상황 (실제로는 더 복잡한 로직)
        if target_year % 2 == 0:  # 짝수 해
            risk_factors.append("올림픽/월드컵 등 대형 이벤트 가능성")
        
        return risk_factors
    
    def _get_comebacks_in_period(
        self, 
        start_date: date, 
        end_date: date, 
        exclude_artist: str = None
    ) -> List[Dict[str, Any]]:
        """특정 기간의 컴백 데이터 조회 (모의 데이터)"""
        comebacks = []
        
        # 실제 구현에서는 데이터베이스 조회
        # 여기서는 모의 데이터 생성
        current_date = start_date
        comeback_id = 0
        
        while current_date <= end_date:
            if np.random.random() > 0.85:  # 15% 확률로 컴백
                artist_id = f"artist_{comeback_id}"
                if artist_id != exclude_artist:
                    comebacks.append({
                        'comeback_id': comeback_id,
                        'artist_id': artist_id,
                        'artist_name': f"Artist {comeback_id}",
                        'comeback_date': current_date,
                        'tier': np.random.choice(['major', 'mid', 'rookie'], p=[0.2, 0.5, 0.3])
                    })
                    comeback_id += 1
            current_date += timedelta(days=1)
        
        return comebacks
    
    # 추가 헬퍼 메서드들 (간소화된 구현)
    def _count_major_artists(self, comebacks: List[Dict]) -> int:
        return len([cb for cb in comebacks if cb.get('tier') == 'major'])
    
    def _analyze_artist_tiers(self, comebacks: List[Dict]) -> Dict[str, int]:
        tier_count = {'major': 0, 'mid': 0, 'rookie': 0}
        for cb in comebacks:
            tier = cb.get('tier', 'rookie')
            tier_count[tier] += 1
        return tier_count
    
    def _analyze_genre_diversity(self, comebacks: List[Dict]) -> float:
        # 장르 다양성 점수 (0-1)
        return min(len(comebacks) * 0.1, 1.0)
    
    def _calculate_competition_intensity(self, comebacks: List[Dict]) -> float:
        major_count = self._count_major_artists(comebacks)
        total_count = len(comebacks)
        return min((major_count * 2 + total_count) / 20.0, 1.0)
    
    def _calculate_daily_comeback_density(
        self, 
        comebacks: List[Dict], 
        start_date: date, 
        end_date: date
    ) -> Dict[str, int]:
        daily_count = defaultdict(int)
        for cb in comebacks:
            daily_count[cb['comeback_date'].isoformat()] += 1
        return dict(daily_count)
    
    def _identify_optimal_windows(self, daily_density: Dict[str, int]) -> List[Dict[str, Any]]:
        # 저밀도 기간을 최적 윈도우로 식별
        optimal_windows = []
        for date_str, count in daily_density.items():
            if count <= 1:  # 하루 1개 이하 컴백
                optimal_windows.append({
                    'date': date_str,
                    'competition_level': 'low',
                    'comeback_count': count
                })
        return optimal_windows[:5]  # 상위 5개만 반환
    
    def _identify_high_risk_periods(self, daily_density: Dict[str, int], comebacks: List[Dict]) -> List[Dict[str, Any]]:
        high_risk = []
        for date_str, count in daily_density.items():
            if count >= 3:  # 하루 3개 이상 컴백
                high_risk.append({
                    'date': date_str,
                    'risk_level': 'high',
                    'comeback_count': count,
                    'major_artists_count': len([
                        cb for cb in comebacks 
                        if cb['comeback_date'].isoformat() == date_str and cb.get('tier') == 'major'
                    ])
                })
        return high_risk
    
    def _predict_future_competition(self, from_date: date) -> Dict[str, Any]:
        # 미래 경쟁도 예측 (간소화)
        return {
            'next_30_days': {
                'expected_comebacks': 8,
                'major_artists_expected': 2,
                'competition_level': 'moderate'
            },
            'seasonal_forecast': {
                'current_season_outlook': 'increasing_competition',
                'next_season_prediction': 'high_competition'
            }
        }
    
    def _generate_competition_recommendations(
        self, 
        competition_metrics: Dict, 
        optimal_windows: List[Dict]
    ) -> List[str]:
        recommendations = []
        
        if competition_metrics['competition_intensity'] > 0.7:
            recommendations.append("경쟁도 높음 - 차별화된 컨셉 필요")
        
        if len(optimal_windows) > 0:
            recommendations.append(f"최적 타이밍: {optimal_windows[0]['date']} 주변")
        
        if competition_metrics['major_artists'] > 3:
            recommendations.append("메이저 아티스트 집중 - 마케팅 강화 필요")
        
        return recommendations
    
    def _calculate_avg_competition_level(self, comebacks: List[Dict]) -> str:
        comeback_count = len(comebacks)
        
        if comeback_count <= 5:
            return "low"
        elif comeback_count <= 15:
            return "moderate"
        else:
            return "high"
    
    def _identify_optimal_timing_windows(self, comebacks: List[Dict], season: ComebackSeason) -> List[Dict[str, Any]]:
        # 시즌 내 최적 타이밍 윈도우 식별
        return [
            {
                'window_start': '첫째 주',
                'window_end': '둘째 주',
                'competition_level': 'low',
                'success_rate': 0.8
            }
        ]
    
    def _analyze_weekly_patterns(self, comebacks: List[Dict], season: ComebackSeason) -> List[int]:
        # 주차별 패턴 분석 (월의 몇째 주가 좋은지)
        return [2, 3]  # 2주차, 3주차가 최적
    
    # 전략 보고서 관련 헬퍼 메서드들
    def _analyze_artist_profile(self, artist_id: str) -> Dict[str, Any]:
        return {
            'career_stage': 'established',
            'fanbase_size': 'large',
            'international_presence': 'moderate',
            'typical_genre': 'pop',
            'marketing_strength': 'high'
        }
    
    def _analyze_strategy_objective(
        self, 
        objective: str, 
        artist_id: str, 
        target_date: date, 
        season: ComebackSeason
    ) -> Dict[str, Any]:
        return {
            'objective': objective,
            'feasibility_score': 0.7,
            'required_resources': 'high',
            'timeline_compatibility': 'good',
            'expected_outcome': 'positive'
        }
    
    def _analyze_timing_optimization(self, target_date: date, competition_analysis: Dict) -> Dict[str, Any]:
        return {
            'current_timing_score': 0.8,
            'alternative_dates': [
                {
                    'date': (target_date + timedelta(days=7)).isoformat(),
                    'score': 0.9,
                    'reason': '경쟁도 감소'
                }
            ],
            'timing_risks': ['주요 아티스트 동시 컴백'],
            'timing_opportunities': ['페스티벌 시즌 활용 가능']
        }
    
    def _conduct_risk_assessment(
        self, 
        artist_id: str, 
        target_date: date, 
        competition_analysis: Dict
    ) -> Dict[str, Any]:
        return {
            'overall_risk_level': 'moderate',
            'critical_risks': [
                '메이저 아티스트 동시 컴백',
                '시상식 시즌 경쟁 심화'
            ],
            'manageable_risks': [
                '신인 아티스트 경쟁',
                '계절적 선호도 변화'
            ],
            'risk_mitigation_strategies': [
                '마케팅 예산 증액',
                '독특한 컨셉 차별화',
                '팬 참여 이벤트 강화'
            ]
        }
    
    def _generate_action_plan(
        self, 
        artist_id: str, 
        target_date: date, 
        strategy_analysis: Dict, 
        timing_analysis: Dict
    ) -> Dict[str, Any]:
        return {
            'primary_recommendation': '계획된 일정 유지, 마케팅 강화',
            'opportunities': [
                '시즌 특성 활용한 컨셉',
                '경쟁자 부재 기간 활용',
                '팬덤 활동 시기 연계'
            ],
            'timing_recommendations': [
                '현재 타이밍 적절',
                '±1주 조정 고려 가능'
            ],
            'strategic_priorities': [
                '차별화된 음악적 완성도',
                '효과적 프로모션 전략',
                '팬 커뮤니케이션 강화'
            ],
            'preparation_schedule': {
                '3개월 전': '컨셉 확정',
                '2개월 전': '음원 제작 완료',
                '1개월 전': '프로모션 시작',
                '컴백 당일': '활동 시작'
            },
            'contingency_options': [
                '타이밍 조정 옵션',
                '마케팅 전략 대안',
                '위기 대응 계획'
            ]
        }
    
    def _calculate_success_probability(
        self, 
        strategy_analysis: Dict, 
        timing_analysis: Dict, 
        risk_assessment: Dict
    ) -> float:
        # 성공 확률 계산 (간소화)
        base_probability = 0.6
        
        timing_bonus = timing_analysis['current_timing_score'] * 0.2
        risk_penalty = 0.1 if risk_assessment['overall_risk_level'] == 'high' else 0.0
        
        return min(base_probability + timing_bonus - risk_penalty, 1.0)
    
    def _define_success_metrics(self, objectives: List[str]) -> Dict[str, str]:
        metric_definitions = {
            'chart_performance': '차트 Top 10 진입',
            'fan_engagement': '팬 참여도 30% 증가',
            'award_eligibility': '주요 시상식 후보 선정',
            'streaming_growth': '스트리밍 수 50% 증가',
            'international_expansion': '해외 차트 진입'
        }
        
        return {obj: metric_definitions.get(obj, '목표 달성') for obj in objectives}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        return {
            "total_comeback_analyses": len(self.comeback_analyses),
            "seasonal_patterns_analyzed": len(self.seasonal_patterns),
            "seasons_covered": [season.value for season in self.seasonal_patterns.keys()],
            "average_confidence_score": np.mean([
                analysis.confidence_score for analysis in self.comeback_analyses.values()
            ]) if self.comeback_analyses else 0.0,
            "system_initialized": datetime.now().isoformat()
        }