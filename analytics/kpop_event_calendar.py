"""
K-Pop 업계 이벤트 달력 시스템
============================

K-Pop 업계의 주요 이벤트들을 체계적으로 관리하고 분석하는 시스템입니다.
시상식, 컴백 시즌, 콘서트 등 다양한 이벤트의 영향도를 분석합니다.

주요 기능:
- 시상식 일정 및 영향도 관리
- 컴백 시즌 패턴 분석
- 이벤트 영향도 정량화
- 업계 전반적인 이벤트 트렌드 분석

Author: Backend Analytics Team
Version: 1.0.0
Date: 2025-09-08
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import json

# Configure logging
logger = logging.getLogger(__name__)


class EventCategory(Enum):
    """이벤트 카테고리 분류"""
    AWARD_SHOW = "award_show"          # 시상식
    COMEBACK = "comeback"              # 컴백
    DEBUT = "debut"                    # 데뷔
    CONCERT = "concert"                # 콘서트
    FANMEETING = "fanmeeting"          # 팬미팅
    VARIETY_SHOW = "variety_show"      # 예능 출연
    COLLABORATION = "collaboration"     # 콜라보레이션
    ANNIVERSARY = "anniversary"        # 기념일
    TOUR = "tour"                      # 투어
    FESTIVAL = "festival"              # 페스티벌
    ALBUM_RELEASE = "album_release"    # 앨범 발매
    SINGLE_RELEASE = "single_release"  # 싱글 발매
    SPECIAL_EVENT = "special_event"    # 특별 이벤트
    HIATUS = "hiatus"                  # 활동 휴식
    CONTROVERSY = "controversy"        # 논란/이슈


class EventImportance(Enum):
    """이벤트 중요도 분류"""
    CRITICAL = 5      # 매우 높음 (MAMA, 골든디스크 등 주요 시상식)
    HIGH = 4          # 높음 (음악방송 1위, 주요 콘서트)
    MEDIUM = 3        # 보통 (일반 컴백, 예능 출연)
    LOW = 2           # 낮음 (SNS 이벤트, 소규모 팬미팅)
    MINIMAL = 1       # 최소 (일상적 활동)


class AwardShow(Enum):
    """주요 시상식 목록"""
    # 대형 시상식
    MAMA = "mama"                          # Mnet Asian Music Awards
    GOLDEN_DISC = "golden_disc"            # 골든 디스크 어워드
    SEOUL_MUSIC_AWARDS = "seoul_music"     # 서울뮤직어워드
    GAON_CHART = "gaon_chart"             # 가온차트뮤직어워드 (현 써클차트)
    
    # TV 방송사 시상식
    KBS_GAYO_DAECHUKJE = "kbs_gayo"       # KBS 가요대축제
    MBC_GAYO_DAEJEJEON = "mbc_gayo"       # MBC 가요대제전
    SBS_GAYO_DAEJEON = "sbs_gayo"         # SBS 가요대전
    
    # 음악방송 연말 특집
    MUSIC_BANK_YEAR_END = "music_bank_ye"  # 뮤직뱅크 연말결산
    INKIGAYO_YEAR_END = "inkigayo_ye"      # 인기가요 연말결산
    
    # 전문 시상식
    KOREAN_MUSIC_AWARDS = "korean_music"  # 한국대중음악상
    KOREAN_HIP_HOP_AWARDS = "khha"        # 한국힙합어워드
    
    # 해외 시상식
    AMERICAN_MUSIC_AWARDS = "ama"         # AMA
    BILLBOARD_MUSIC_AWARDS = "bbma"       # BBMA


class ComebackSeason(Enum):
    """컴백 시즌 분류"""
    SPRING = "spring"          # 봄 시즌 (3-5월)
    SUMMER = "summer"          # 여름 시즌 (6-8월)  
    AUTUMN = "autumn"          # 가을 시즌 (9-11월)
    WINTER = "winter"          # 겨울 시즌 (12-2월)
    AWARD_SEASON = "award"     # 시상식 시즌 (11-1월)
    FESTIVAL_SEASON = "festival" # 페스티벌 시즌 (7-9월)


@dataclass
class EventImpactMetrics:
    """이벤트 영향도 측정 지표"""
    pre_event_avg: float = 0.0           # 이벤트 전 평균값
    post_event_peak: float = 0.0         # 이벤트 후 최고값
    impact_magnitude: float = 0.0        # 영향도 크기
    impact_duration_days: int = 0        # 영향 지속 기간
    recovery_days: int = 0               # 원상 복구 기간
    significance_score: float = 0.0      # 통계적 유의성
    relative_importance: float = 0.0     # 상대적 중요도
    platform_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass  
class KPopEvent:
    """K-Pop 이벤트 데이터 클래스"""
    event_id: str
    name: str
    event_date: date
    category: EventCategory
    importance: EventImportance
    description: str = ""
    
    # 관련 엔티티
    artist_id: Optional[str] = None
    group_id: Optional[str] = None
    company_id: Optional[str] = None
    
    # 시상식 관련 정보
    award_show: Optional[AwardShow] = None
    awards_won: List[str] = field(default_factory=list)
    nominations: List[str] = field(default_factory=list)
    
    # 컴백 관련 정보
    comeback_season: Optional[ComebackSeason] = None
    album_name: Optional[str] = None
    title_track: Optional[str] = None
    
    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 영향도 데이터
    impact_metrics: Optional[EventImpactMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'event_id': self.event_id,
            'name': self.name,
            'event_date': self.event_date.isoformat(),
            'category': self.category.value,
            'importance': self.importance.value,
            'description': self.description,
            'artist_id': self.artist_id,
            'group_id': self.group_id,
            'company_id': self.company_id,
            'award_show': self.award_show.value if self.award_show else None,
            'awards_won': self.awards_won,
            'nominations': self.nominations,
            'comeback_season': self.comeback_season.value if self.comeback_season else None,
            'album_name': self.album_name,
            'title_track': self.title_track,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class KPopEventCalendar:
    """
    K-Pop 업계 이벤트 달력 관리 시스템
    
    주요 기능:
    1. 이벤트 등록 및 관리
    2. 시상식 일정 자동 생성
    3. 컴백 시즌 분석
    4. 이벤트 영향도 분석
    5. 업계 트렌드 분석
    """
    
    def __init__(self, database_connection=None):
        """이벤트 달력 시스템 초기화"""
        self.db_conn = database_connection
        self.events: Dict[str, KPopEvent] = {}
        self.logger = logger
        
        # 주요 시상식 일정 (매년 반복)
        self.annual_award_shows = self._initialize_annual_awards()
        
        # 컴백 시즌 패턴
        self.comeback_seasons = self._initialize_comeback_seasons()
        
        self.logger.info("K-Pop Event Calendar System initialized")
    
    def _initialize_annual_awards(self) -> Dict[AwardShow, Dict[str, Any]]:
        """연례 시상식 일정 초기화"""
        return {
            # 11-12월 시상식 시즌
            AwardShow.MAMA: {
                'typical_month': 11,
                'typical_day_range': (20, 30),
                'importance': EventImportance.CRITICAL,
                'description': '아시아 최대 음악 시상식',
                'location': '다양한 아시아 도시',
                'duration_days': 2
            },
            AwardShow.GOLDEN_DISC: {
                'typical_month': 1,
                'typical_day_range': (5, 15),
                'importance': EventImportance.CRITICAL,
                'description': '한국 대표 음악 시상식',
                'location': '서울',
                'duration_days': 2
            },
            AwardShow.SEOUL_MUSIC_AWARDS: {
                'typical_month': 1,
                'typical_day_range': (20, 31),
                'importance': EventImportance.HIGH,
                'description': '서울뮤직어워드',
                'location': '서울',
                'duration_days': 1
            },
            AwardShow.GAON_CHART: {
                'typical_month': 1,
                'typical_day_range': (25, 31),
                'importance': EventImportance.HIGH,
                'description': '차트 기반 시상식',
                'location': '서울',
                'duration_days': 1
            },
            
            # 연말 방송사 가요제
            AwardShow.KBS_GAYO_DAECHUKJE: {
                'typical_month': 12,
                'typical_day_range': (25, 31),
                'importance': EventImportance.HIGH,
                'description': 'KBS 연말 가요 축제',
                'location': '서울',
                'duration_days': 1
            },
            AwardShow.MBC_GAYO_DAEJEJEON: {
                'typical_month': 12,
                'typical_day_range': (28, 31),
                'importance': EventImportance.HIGH,
                'description': 'MBC 연말 가요 대제전',
                'location': '서울',
                'duration_days': 1
            },
            AwardShow.SBS_GAYO_DAEJEON: {
                'typical_month': 12,
                'typical_day_range': (25, 30),
                'importance': EventImportance.HIGH,
                'description': 'SBS 연말 가요 대전',
                'location': '서울',
                'duration_days': 1
            }
        }
    
    def _initialize_comeback_seasons(self) -> Dict[ComebackSeason, Dict[str, Any]]:
        """컴백 시즌 패턴 초기화"""
        return {
            ComebackSeason.SPRING: {
                'months': [3, 4, 5],
                'characteristics': '신학기 시즌, 밝은 컨셉 선호',
                'competition_level': 'medium',
                'expected_impact_boost': 1.1
            },
            ComebackSeason.SUMMER: {
                'months': [6, 7, 8],
                'characteristics': '여름 페스티벌, 청량한 컨셉',
                'competition_level': 'high',
                'expected_impact_boost': 1.3
            },
            ComebackSeason.AUTUMN: {
                'months': [9, 10, 11],
                'characteristics': '시상식 준비, 다양한 컨셉',
                'competition_level': 'very_high',
                'expected_impact_boost': 1.4
            },
            ComebackSeason.WINTER: {
                'months': [12, 1, 2],
                'characteristics': '시상식 시즌, 발라드 인기',
                'competition_level': 'high',
                'expected_impact_boost': 1.2
            },
            ComebackSeason.AWARD_SEASON: {
                'months': [11, 12, 1],
                'characteristics': '시상식 집중 시기',
                'competition_level': 'extreme',
                'expected_impact_boost': 1.5
            }
        }
    
    def add_event(self, event: KPopEvent) -> bool:
        """이벤트 추가"""
        try:
            # 중복 검사
            if event.event_id in self.events:
                self.logger.warning(f"Event {event.event_id} already exists")
                return False
            
            # 데이터 검증
            if not self._validate_event(event):
                return False
            
            # 컴백 시즌 자동 설정
            if event.category == EventCategory.COMEBACK and not event.comeback_season:
                event.comeback_season = self._determine_comeback_season(event.event_date)
            
            self.events[event.event_id] = event
            
            # 데이터베이스에 저장
            if self.db_conn:
                self._save_event_to_db(event)
            
            self.logger.info(f"Event added: {event.name} ({event.event_date})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add event {event.event_id}: {e}")
            return False
    
    def generate_annual_award_calendar(self, year: int) -> List[KPopEvent]:
        """연례 시상식 달력 생성"""
        annual_events = []
        
        for award_show, config in self.annual_award_shows.items():
            try:
                # 일정 추정 (실제로는 공식 발표 일정을 사용해야 함)
                month = config['typical_month']
                day_range = config['typical_day_range']
                estimated_day = (day_range[0] + day_range[1]) // 2
                
                event_date = date(year, month, estimated_day)
                
                event = KPopEvent(
                    event_id=f"award_{award_show.value}_{year}",
                    name=f"{award_show.value.upper()} {year}",
                    event_date=event_date,
                    category=EventCategory.AWARD_SHOW,
                    importance=config['importance'],
                    description=config['description'],
                    award_show=award_show
                )
                
                annual_events.append(event)
                
            except Exception as e:
                self.logger.error(f"Failed to generate event for {award_show}: {e}")
        
        self.logger.info(f"Generated {len(annual_events)} annual award events for {year}")
        return annual_events
    
    def analyze_comeback_season_patterns(
        self, 
        start_date: date, 
        end_date: date,
        artist_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """컴백 시즌 패턴 분석"""
        try:
            # 컴백 이벤트 필터링
            comeback_events = [
                event for event in self.events.values()
                if (event.category == EventCategory.COMEBACK and
                    start_date <= event.event_date <= end_date and
                    (not artist_id or event.artist_id == artist_id))
            ]
            
            if not comeback_events:
                return {"error": "No comeback events found in the specified period"}
            
            # 시즌별 분석
            season_analysis = defaultdict(list)
            for event in comeback_events:
                season = self._determine_comeback_season(event.event_date)
                season_analysis[season.value].append(event)
            
            # 통계 계산
            analysis_result = {
                "period": f"{start_date} to {end_date}",
                "total_comebacks": len(comeback_events),
                "season_distribution": {},
                "competition_analysis": {},
                "recommendations": []
            }
            
            for season_name, events in season_analysis.items():
                season_info = self.comeback_seasons[ComebackSeason(season_name)]
                
                analysis_result["season_distribution"][season_name] = {
                    "count": len(events),
                    "percentage": round(len(events) / len(comeback_events) * 100, 1),
                    "characteristics": season_info['characteristics'],
                    "competition_level": season_info['competition_level']
                }
                
                # 월별 분포
                monthly_dist = defaultdict(int)
                for event in events:
                    monthly_dist[event.event_date.month] += 1
                
                analysis_result["season_distribution"][season_name]["monthly_distribution"] = dict(monthly_dist)
            
            # 경쟁도 분석
            analysis_result["competition_analysis"] = self._analyze_competition_levels(comeback_events)
            
            # 추천사항 생성
            analysis_result["recommendations"] = self._generate_comeback_recommendations(season_analysis)
            
            self.logger.info(f"Comeback season analysis completed: {len(comeback_events)} events analyzed")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in comeback season analysis: {e}")
            return {"error": str(e)}
    
    def calculate_event_impact(
        self,
        event: KPopEvent,
        metrics_data: pd.DataFrame,
        baseline_days: int = 14,
        impact_window_days: int = 30
    ) -> EventImpactMetrics:
        """이벤트 영향도 계산"""
        try:
            event_date = pd.to_datetime(event.event_date)
            
            # 기준선 기간 설정
            baseline_start = event_date - timedelta(days=baseline_days)
            baseline_end = event_date - timedelta(days=1)
            
            # 영향도 측정 기간 설정
            impact_start = event_date
            impact_end = event_date + timedelta(days=impact_window_days)
            
            # 데이터 필터링
            baseline_data = metrics_data[
                (metrics_data.index >= baseline_start) & 
                (metrics_data.index <= baseline_end)
            ]
            impact_data = metrics_data[
                (metrics_data.index >= impact_start) & 
                (metrics_data.index <= impact_end)
            ]
            
            if baseline_data.empty or impact_data.empty:
                self.logger.warning(f"Insufficient data for event impact analysis: {event.event_id}")
                return EventImpactMetrics()
            
            # 지표별 영향도 계산
            platform_breakdown = {}
            total_impact = 0.0
            metric_count = 0
            
            for column in metrics_data.columns:
                if column in ['artist_id', 'event_date']:
                    continue
                
                baseline_avg = baseline_data[column].mean()
                impact_peak = impact_data[column].max()
                
                if baseline_avg > 0:
                    impact_ratio = (impact_peak - baseline_avg) / baseline_avg
                    platform_breakdown[column] = impact_ratio
                    total_impact += impact_ratio
                    metric_count += 1
            
            # 전체 영향도 평균
            avg_impact = total_impact / metric_count if metric_count > 0 else 0.0
            
            # 영향 지속 기간 계산 (간소화)
            impact_duration = self._calculate_impact_duration(impact_data, baseline_data.mean().mean())
            
            # 회복 기간 계산
            recovery_days = min(impact_window_days, 14)  # 기본값 14일
            
            # 통계적 유의성 (간소화)
            significance = self._calculate_statistical_significance(baseline_data, impact_data)
            
            # 상대적 중요도 (이벤트 중요도 기반)
            relative_importance = event.importance.value / 5.0
            
            impact_metrics = EventImpactMetrics(
                pre_event_avg=float(baseline_data.mean().mean()),
                post_event_peak=float(impact_data.max().max()),
                impact_magnitude=float(avg_impact),
                impact_duration_days=impact_duration,
                recovery_days=recovery_days,
                significance_score=significance,
                relative_importance=relative_importance,
                platform_breakdown=platform_breakdown
            )
            
            # 이벤트에 영향도 정보 저장
            event.impact_metrics = impact_metrics
            
            self.logger.info(f"Impact calculated for event {event.event_id}: {avg_impact:.3f}")
            return impact_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating event impact for {event.event_id}: {e}")
            return EventImpactMetrics()
    
    def get_events_by_period(
        self,
        start_date: date,
        end_date: date,
        category: Optional[EventCategory] = None,
        importance: Optional[EventImportance] = None
    ) -> List[KPopEvent]:
        """기간별 이벤트 조회"""
        filtered_events = []
        
        for event in self.events.values():
            # 날짜 필터
            if not (start_date <= event.event_date <= end_date):
                continue
            
            # 카테고리 필터
            if category and event.category != category:
                continue
            
            # 중요도 필터
            if importance and event.importance != importance:
                continue
            
            filtered_events.append(event)
        
        # 날짜순 정렬
        filtered_events.sort(key=lambda x: x.event_date)
        
        self.logger.info(f"Retrieved {len(filtered_events)} events for period {start_date} to {end_date}")
        return filtered_events
    
    def analyze_award_show_impact(
        self,
        award_show: AwardShow,
        year: int,
        metrics_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """시상식 영향도 분석"""
        try:
            # 해당 시상식 이벤트 찾기
            award_events = [
                event for event in self.events.values()
                if (event.award_show == award_show and
                    event.event_date.year == year)
            ]
            
            if not award_events:
                return {"error": f"No {award_show.value} events found for {year}"}
            
            award_event = award_events[0]  # 첫 번째 이벤트 사용
            
            # 영향도 계산
            impact_metrics = self.calculate_event_impact(
                award_event, metrics_data, 
                baseline_days=21, impact_window_days=45
            )
            
            # 시상식별 특수 분석
            analysis_result = {
                "award_show": award_show.value,
                "year": year,
                "event_date": award_event.event_date.isoformat(),
                "overall_impact": impact_metrics.impact_magnitude,
                "platform_breakdown": impact_metrics.platform_breakdown,
                "significance": impact_metrics.significance_score,
                "comparative_analysis": self._compare_with_historical_awards(award_show, impact_metrics)
            }
            
            # 수상자별 영향도 (만약 수상 정보가 있다면)
            if award_event.awards_won:
                analysis_result["awards_impact"] = self._analyze_individual_award_impact(
                    award_event, metrics_data
                )
            
            self.logger.info(f"Award show impact analysis completed for {award_show.value} {year}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in award show impact analysis: {e}")
            return {"error": str(e)}
    
    def predict_optimal_comeback_timing(
        self,
        artist_id: str,
        target_year: int,
        historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """최적 컴백 타이밍 예측"""
        try:
            # 아티스트의 과거 컴백 패턴 분석
            artist_events = [
                event for event in self.events.values()
                if (event.artist_id == artist_id and 
                    event.category == EventCategory.COMEBACK)
            ]
            
            # 시즌별 성과 분석
            season_performance = defaultdict(list)
            for event in artist_events:
                season = self._determine_comeback_season(event.event_date)
                if event.impact_metrics:
                    season_performance[season].append(event.impact_metrics.impact_magnitude)
            
            # 업계 전체 컴백 밀도 분석
            industry_density = self._analyze_industry_comeback_density(target_year)
            
            # 시상식 일정과의 관계 분석
            award_calendar = self.generate_annual_award_calendar(target_year)
            
            # 최적 타이밍 계산
            recommendations = []
            
            for season in ComebackSeason:
                if season == ComebackSeason.FESTIVAL_SEASON:
                    continue  # 별도 처리
                
                season_info = self.comeback_seasons[season]
                
                # 성과 예측 점수 계산
                historical_performance = np.mean(season_performance[season]) if season_performance[season] else 0.5
                competition_penalty = self._calculate_competition_penalty(season, industry_density)
                timing_bonus = self._calculate_timing_bonus(season, award_calendar)
                
                total_score = (
                    historical_performance * 0.4 +
                    season_info['expected_impact_boost'] * 0.3 -
                    competition_penalty * 0.2 +
                    timing_bonus * 0.1
                )
                
                recommendations.append({
                    "season": season.value,
                    "months": season_info['months'],
                    "predicted_score": round(total_score, 3),
                    "historical_performance": round(historical_performance, 3),
                    "competition_level": season_info['competition_level'],
                    "characteristics": season_info['characteristics'],
                    "recommendation_reason": self._generate_recommendation_reason(
                        season, total_score, historical_performance, competition_penalty
                    )
                })
            
            # 점수순 정렬
            recommendations.sort(key=lambda x: x['predicted_score'], reverse=True)
            
            result = {
                "artist_id": artist_id,
                "target_year": target_year,
                "recommendations": recommendations,
                "top_recommendation": recommendations[0] if recommendations else None,
                "analysis_metadata": {
                    "historical_events_analyzed": len(artist_events),
                    "industry_density_considered": True,
                    "award_calendar_integrated": True
                }
            }
            
            self.logger.info(f"Optimal comeback timing predicted for artist {artist_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in comeback timing prediction: {e}")
            return {"error": str(e)}
    
    def _validate_event(self, event: KPopEvent) -> bool:
        """이벤트 데이터 검증"""
        if not event.name or not event.event_date:
            return False
        
        if event.event_date > date.today() + timedelta(days=365*2):
            self.logger.warning(f"Event date too far in future: {event.event_date}")
            return False
        
        return True
    
    def _determine_comeback_season(self, event_date: date) -> ComebackSeason:
        """이벤트 날짜로부터 컴백 시즌 결정"""
        month = event_date.month
        
        if month in [3, 4, 5]:
            return ComebackSeason.SPRING
        elif month in [6, 7, 8]:
            return ComebackSeason.SUMMER
        elif month in [9, 10, 11]:
            return ComebackSeason.AUTUMN
        else:  # 12, 1, 2
            return ComebackSeason.WINTER
    
    def _analyze_competition_levels(self, events: List[KPopEvent]) -> Dict[str, Any]:
        """경쟁도 분석"""
        # 월별 컴백 밀도
        monthly_density = defaultdict(int)
        for event in events:
            monthly_density[event.event_date.month] += 1
        
        # 고경쟁 기간 식별
        high_competition_months = [
            month for month, count in monthly_density.items()
            if count > np.mean(list(monthly_density.values())) + np.std(list(monthly_density.values()))
        ]
        
        return {
            "monthly_density": dict(monthly_density),
            "high_competition_months": high_competition_months,
            "average_monthly_comebacks": round(np.mean(list(monthly_density.values())), 1),
            "peak_month": max(monthly_density.keys(), key=lambda x: monthly_density[x]) if monthly_density else None
        }
    
    def _generate_comeback_recommendations(self, season_analysis: Dict) -> List[str]:
        """컴백 추천사항 생성"""
        recommendations = []
        
        # 가장 경쟁이 적은 시즌 찾기
        if season_analysis:
            season_counts = {season: len(events) for season, events in season_analysis.items()}
            least_competitive = min(season_counts.keys(), key=lambda x: season_counts[x])
            recommendations.append(f"가장 경쟁이 적은 시즌: {least_competitive}")
        
        # 시즌별 특성 기반 추천
        for season_name in season_analysis.keys():
            season = ComebackSeason(season_name)
            if season in self.comeback_seasons:
                characteristics = self.comeback_seasons[season]['characteristics']
                recommendations.append(f"{season_name}: {characteristics}")
        
        return recommendations
    
    def _calculate_impact_duration(self, impact_data: pd.DataFrame, baseline_avg: float) -> int:
        """영향 지속 기간 계산 (간소화)"""
        try:
            # 기준선의 120% 이상 유지되는 기간 계산
            threshold = baseline_avg * 1.2
            
            for i, (_, row) in enumerate(impact_data.iterrows()):
                if row.mean() < threshold:
                    return max(1, i)
            
            return len(impact_data)
            
        except Exception:
            return 7  # 기본값
    
    def _calculate_statistical_significance(self, baseline: pd.DataFrame, impact: pd.DataFrame) -> float:
        """통계적 유의성 계산 (간소화)"""
        try:
            from scipy import stats
            
            baseline_means = baseline.mean(axis=1)
            impact_means = impact.mean(axis=1)
            
            if len(baseline_means) > 1 and len(impact_means) > 1:
                _, p_value = stats.ttest_ind(impact_means, baseline_means)
                return 1.0 - p_value  # 신뢰도로 변환
            
            return 0.5  # 기본값
            
        except Exception:
            return 0.5  # 기본값
    
    def _compare_with_historical_awards(self, award_show: AwardShow, current_metrics: EventImpactMetrics) -> Dict[str, Any]:
        """과거 시상식과의 비교 분석"""
        # 실제 구현에서는 과거 데이터와 비교
        return {
            "vs_previous_year": "similar",  # 예시
            "vs_industry_average": "above_average",
            "trend": "increasing"
        }
    
    def _analyze_individual_award_impact(self, event: KPopEvent, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        """개별 수상 영향도 분석"""
        # 실제 구현에서는 수상자별로 세분화된 분석
        return {
            "major_awards_impact": 0.3,  # 예시
            "popularity_awards_impact": 0.2,
            "rookie_awards_impact": 0.4
        }
    
    def _analyze_industry_comeback_density(self, year: int) -> Dict[str, Any]:
        """업계 전체 컴백 밀도 분석"""
        # 실제 구현에서는 해당 연도의 전체 컴백 일정 분석
        return {
            "high_density_months": [3, 9, 10],
            "low_density_months": [1, 2, 6],
            "average_monthly_comebacks": 15
        }
    
    def _calculate_competition_penalty(self, season: ComebackSeason, industry_density: Dict) -> float:
        """경쟁도에 따른 패널티 계산"""
        season_info = self.comeback_seasons[season]
        competition_level = season_info['competition_level']
        
        penalty_map = {
            'low': 0.0,
            'medium': 0.1,
            'high': 0.2,
            'very_high': 0.3,
            'extreme': 0.4
        }
        
        return penalty_map.get(competition_level, 0.1)
    
    def _calculate_timing_bonus(self, season: ComebackSeason, award_calendar: List[KPopEvent]) -> float:
        """시상식 타이밍에 따른 보너스 계산"""
        # 시상식 전 2-3개월 컴백시 보너스
        season_months = self.comeback_seasons[season]['months']
        
        for award_event in award_calendar:
            award_month = award_event.event_date.month
            
            # 주요 시상식 전 적절한 타이밍 체크
            if award_event.importance == EventImportance.CRITICAL:
                gap_months = [(award_month - month) % 12 for month in season_months]
                if any(2 <= gap <= 4 for gap in gap_months):  # 2-4개월 전
                    return 0.2
        
        return 0.0
    
    def _generate_recommendation_reason(
        self, 
        season: ComebackSeason, 
        total_score: float, 
        historical_performance: float, 
        competition_penalty: float
    ) -> str:
        """추천 이유 생성"""
        reasons = []
        
        if historical_performance > 0.7:
            reasons.append("과거 성과가 우수함")
        
        if competition_penalty < 0.2:
            reasons.append("경쟁도가 낮음")
        
        if total_score > 1.0:
            reasons.append("전반적으로 유리한 조건")
        
        season_char = self.comeback_seasons[season]['characteristics']
        reasons.append(f"시즌 특성: {season_char}")
        
        return ", ".join(reasons)
    
    def _save_event_to_db(self, event: KPopEvent) -> bool:
        """이벤트를 데이터베이스에 저장"""
        try:
            # 실제 데이터베이스 저장 로직은 데이터베이스 연결에 따라 구현
            self.logger.info(f"Event {event.event_id} would be saved to database")
            return True
        except Exception as e:
            self.logger.error(f"Database save failed: {e}")
            return False
    
    def export_calendar(self, format_type: str = "json") -> Union[str, Dict]:
        """이벤트 달력 내보내기"""
        try:
            events_data = [event.to_dict() for event in self.events.values()]
            
            if format_type == "json":
                return json.dumps(events_data, indent=2, ensure_ascii=False)
            elif format_type == "dict":
                return {"events": events_data, "total_count": len(events_data)}
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return {"error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        return {
            "total_events": len(self.events),
            "events_by_category": {
                cat.value: len([e for e in self.events.values() if e.category == cat])
                for cat in EventCategory
            },
            "events_by_importance": {
                imp.value: len([e for e in self.events.values() if e.importance == imp])
                for imp in EventImportance
            },
            "date_range": {
                "earliest": min([e.event_date for e in self.events.values()]).isoformat() if self.events else None,
                "latest": max([e.event_date for e in self.events.values()]).isoformat() if self.events else None
            }
        }