"""
K-Pop 주요 시상식 데이터 매니저
==============================

한국 및 아시아 주요 음악 시상식 데이터를 체계적으로 관리하는 모듈입니다.
시상식 일정, 시상 부문, 과거 수상 이력 등을 포함합니다.

주요 기능:
- 연례 시상식 일정 관리
- 시상 부문별 데이터 구조화
- 과거 수상 이력 분석
- 시상식 영향도 평가

Author: Backend Analytics Team
Version: 1.0.0
Date: 2025-09-08
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# 조건부 import - pandas가 없을 때를 대비
try:
    from .kpop_event_calendar import AwardShow, EventImportance, KPopEvent, EventCategory
except ImportError:
    # pandas 없이도 작동하도록 필요한 enum들을 직접 정의
    class AwardShow(Enum):
        MAMA = "mama"
        GOLDEN_DISC = "golden_disc"
        SEOUL_MUSIC_AWARDS = "seoul_music_awards"
        GAON_CHART = "gaon_chart"
        KBS_GAYO_DAECHUKJE = "kbs_gayo_daechukje"
        MBC_GAYO_DAEJEJEON = "mbc_gayo_daejejeon"
        SBS_GAYO_DAEJEON = "sbs_gayo_daejeon"
        KOREAN_MUSIC_AWARDS = "korean_music_awards"
        MUSIC_BANK_YEAR_END = "music_bank_year_end"
        INKIGAYO_YEAR_END = "inkigayo_year_end"
        KOREAN_HIP_HOP_AWARDS = "korean_hip_hop_awards"
    
    class EventImportance(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    class EventCategory(Enum):
        AWARD_SHOW = "award_show"
        COMEBACK = "comeback"
        CONCERT = "concert"
        COLLABORATION = "collaboration"
        OTHER = "other"

# Configure logging
logger = logging.getLogger(__name__)


class AwardCategory(Enum):
    """시상 부문 분류"""
    # 주요 음악상
    ARTIST_OF_THE_YEAR = "artist_of_the_year"           # 올해의 아티스트
    ALBUM_OF_THE_YEAR = "album_of_the_year"             # 올해의 앨범
    SONG_OF_THE_YEAR = "song_of_the_year"               # 올해의 노래
    
    # 장르별 상
    MALE_ARTIST = "male_artist"                          # 남자 아티스트상
    FEMALE_ARTIST = "female_artist"                      # 여자 아티스트상
    GROUP_ARTIST = "group_artist"                        # 그룹 아티스트상
    
    # 신인상
    ROOKIE_OF_THE_YEAR = "rookie_of_the_year"           # 신인상
    BEST_NEW_MALE_ARTIST = "best_new_male"              # 최우수 신인 남자 아티스트
    BEST_NEW_FEMALE_ARTIST = "best_new_female"          # 최우수 신인 여자 아티스트
    BEST_NEW_GROUP = "best_new_group"                   # 최우수 신인 그룹
    
    # 인기상
    POPULARITY_AWARD = "popularity"                      # 인기상
    GLOBAL_ARTIST = "global_artist"                     # 글로벌 아티스트상
    SOCIAL_MEDIA_AWARD = "social_media"                 # 소셜미디어상
    
    # 음악성 관련
    BEST_VOCAL_PERFORMANCE = "best_vocal"                # 최우수 보컬
    BEST_DANCE_PERFORMANCE = "best_dance"               # 최우수 댄스
    BEST_RAP_PERFORMANCE = "best_rap"                   # 최우수 랩
    
    # 제작 관련
    PRODUCER_AWARD = "producer"                          # 프로듀서상
    SONGWRITER_AWARD = "songwriter"                      # 작곡가상
    BEST_MUSIC_VIDEO = "best_mv"                        # 최우수 뮤직비디오
    
    # 특별상
    SPECIAL_ACHIEVEMENT = "special_achievement"         # 공로상
    HALL_OF_FAME = "hall_of_fame"                       # 명예의 전당


@dataclass
class AwardResult:
    """시상 결과 데이터 클래스"""
    award_category: AwardCategory
    winner_id: str                    # artist_id 또는 group_id
    winner_name: str
    winner_type: str                  # 'artist', 'group', 'song', 'album'
    
    # 수상 작품 정보 (해당하는 경우)
    winning_work: Optional[str] = None    # 앨범명, 곡명 등
    winning_work_type: Optional[str] = None  # 'album', 'single', 'song'
    
    # 메타데이터
    year: int = 0
    award_show: Optional[AwardShow] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'award_category': self.award_category.value,
            'winner_id': self.winner_id,
            'winner_name': self.winner_name,
            'winner_type': self.winner_type,
            'winning_work': self.winning_work,
            'winning_work_type': self.winning_work_type,
            'year': self.year,
            'award_show': self.award_show.value if self.award_show else None,
            'notes': self.notes
        }


@dataclass
class AwardShowInfo:
    """시상식 상세 정보"""
    award_show: AwardShow
    full_name_ko: str                     # 한국어 정식 명칭
    full_name_en: str                     # 영어 정식 명칭
    organizer: str                        # 주최기관
    first_held: int                       # 첫 개최 연도
    importance_level: EventImportance     # 중요도
    
    # 일정 정보
    typical_month: int                    # 보통 개최되는 월
    typical_duration_days: int            # 행사 기간
    location_pattern: str                 # 개최 장소 패턴
    
    # 특징
    characteristics: List[str] = field(default_factory=list)
    award_categories: List[AwardCategory] = field(default_factory=list)
    
    # 영향력 정보
    expected_viewership: int = 0          # 예상 시청자 수
    industry_significance: float = 0.0    # 업계 영향력 (0-1)
    international_reach: float = 0.0      # 국제적 영향력 (0-1)
    
    # 방송 정보
    broadcast_channels: List[str] = field(default_factory=list)  # 방송 채널
    venue_capacity: int = 0               # 회장 수용 인원
    
    # 투표 정보
    voting_start_date: Optional[str] = None
    voting_end_date: Optional[str] = None
    
    # 지표 정보
    impact_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'award_show': self.award_show.value,
            'full_name_ko': self.full_name_ko,
            'full_name_en': self.full_name_en,
            'organizer': self.organizer,
            'first_held': self.first_held,
            'importance_level': self.importance_level.value,
            'typical_month': self.typical_month,
            'typical_duration_days': self.typical_duration_days,
            'location_pattern': self.location_pattern,
            'characteristics': self.characteristics,
            'award_categories': [cat.value for cat in self.award_categories],
            'expected_viewership': self.expected_viewership,
            'industry_significance': self.industry_significance,
            'broadcast_channels': self.broadcast_channels,            'venue_capacity': self.venue_capacity,            'voting_start_date': self.voting_start_date,            'voting_end_date': self.voting_end_date,            'impact_metrics': self.impact_metrics,
            'international_reach': self.international_reach
        }


class AwardShowDataManager:
    """
    K-Pop 주요 시상식 데이터 매니저
    
    시상식 정보를 체계적으로 관리하고 분석 데이터를 제공합니다.
    """
    
    def __init__(self):
        """데이터 매니저 초기화"""
        self.award_shows_info: Dict[AwardShow, AwardShowInfo] = {}
        self.award_results: Dict[str, List[AwardResult]] = {}  # year_award_show -> results
        self.logger = logger
        
        # 시상식 정보 초기화
        self._initialize_award_shows_data()
        
        self.logger.info("Award Show Data Manager initialized")
    
    def _initialize_award_shows_data(self):
        """시상식 데이터 초기화"""
        
        # MAMA (Mnet Asian Music Awards)
        self.award_shows_info[AwardShow.MAMA] = AwardShowInfo(
            award_show=AwardShow.MAMA,
            full_name_ko="엠넷 아시안 뮤직 어워드",
            full_name_en="Mnet Asian Music Awards",
            organizer="CJ ENM",
            first_held=1999,
            importance_level=EventImportance.CRITICAL,
            typical_month=11,
            typical_duration_days=2,
            location_pattern="아시아 주요 도시 순회",
            characteristics=[
                "아시아 최대 규모 음악 시상식",
                "K-Pop의 글로벌 확산 견인",
                "화려한 무대와 콜라보레이션",
                "실시간 글로벌 스트리밍"
            ],
            award_categories=[
                AwardCategory.ARTIST_OF_THE_YEAR,
                AwardCategory.ALBUM_OF_THE_YEAR,
                AwardCategory.SONG_OF_THE_YEAR,
                AwardCategory.MALE_ARTIST,
                AwardCategory.FEMALE_ARTIST,
                AwardCategory.GROUP_ARTIST,
                AwardCategory.ROOKIE_OF_THE_YEAR,
                AwardCategory.BEST_DANCE_PERFORMANCE,
                AwardCategory.BEST_VOCAL_PERFORMANCE,
                AwardCategory.BEST_RAP_PERFORMANCE
            ],
            expected_viewership=50000000,
            industry_significance=0.95,
            international_reach=0.90,
            broadcast_channels=["Mnet", "YouTube", "Twitter"],
            venue_capacity=20000,
            voting_start_date=None,
            voting_end_date=None,
            impact_metrics={"global_impact_score": 9.5, "social_buzz": 9.8}
        )
        
        # 골든디스크 어워드
        self.award_shows_info[AwardShow.GOLDEN_DISC] = AwardShowInfo(
            award_show=AwardShow.GOLDEN_DISC,
            full_name_ko="골든디스크어워드",
            full_name_en="Golden Disc Awards",
            organizer="한국음반산업협회",
            first_held=1986,
            importance_level=EventImportance.CRITICAL,
            typical_month=1,
            typical_duration_days=2,
            location_pattern="서울 및 주요 도시",
            characteristics=[
                "한국 음반산업 권위있는 시상식",
                "음반/디지털 부문 구분 시상",
                "실제 판매량 기반 평가",
                "업계 전문가 참여"
            ],
            award_categories=[
                AwardCategory.ALBUM_OF_THE_YEAR,
                AwardCategory.SONG_OF_THE_YEAR,
                AwardCategory.ROOKIE_OF_THE_YEAR,
                AwardCategory.BEST_NEW_MALE_ARTIST,
                AwardCategory.BEST_NEW_FEMALE_ARTIST,
                AwardCategory.POPULARITY_AWARD
            ],
            expected_viewership=20000000,
            industry_significance=0.90,
            international_reach=0.60,
            broadcast_channels=["JTBC", "YouTube"],
            venue_capacity=10000,
            voting_start_date=None,
            voting_end_date=None,
            impact_metrics={"global_impact_score": 8.5, "industry_authority": 9.2}
        )
        
        # 서울뮤직어워드
        self.award_shows_info[AwardShow.SEOUL_MUSIC_AWARDS] = AwardShowInfo(
            award_show=AwardShow.SEOUL_MUSIC_AWARDS,
            full_name_ko="서울뮤직어워드",
            full_name_en="Seoul Music Awards",
            organizer="스포츠서울",
            first_held=1990,
            importance_level=EventImportance.HIGH,
            typical_month=1,
            typical_duration_days=1,
            location_pattern="서울 주요 공연장",
            characteristics=[
                "언론사 주최 공신력 있는 시상식",
                "음원+음반+전문가 평가 종합",
                "신인 발굴에 중점",
                "팬 투표 부문 존재"
            ],
            award_categories=[
                AwardCategory.ARTIST_OF_THE_YEAR,
                AwardCategory.ALBUM_OF_THE_YEAR,
                AwardCategory.SONG_OF_THE_YEAR,
                AwardCategory.ROOKIE_OF_THE_YEAR,
                AwardCategory.POPULARITY_AWARD,
                AwardCategory.SOCIAL_MEDIA_AWARD
            ],
            expected_viewership=15000000,
            industry_significance=0.80,
            international_reach=0.50,
            broadcast_channels=["KBS2", "YouTube"],
            venue_capacity=8000,
            voting_start_date=None,
            voting_end_date=None,
            impact_metrics={"global_impact_score": 7.8, "fan_engagement": 8.5}
        )
        
        # 가온차트 뮤직어워드 (현 써클차트)
        self.award_shows_info[AwardShow.GAON_CHART] = AwardShowInfo(
            award_show=AwardShow.GAON_CHART,
            full_name_ko="써클차트 뮤직어워드",
            full_name_en="Circle Chart Music Awards",
            organizer="한국음악콘텐츠산업협회",
            first_held=2012,
            importance_level=EventImportance.HIGH,
            typical_month=1,
            typical_duration_days=1,
            location_pattern="서울 공연장",
            characteristics=[
                "공식 음악차트 기반 시상",
                "데이터 기반 객관적 평가",
                "월간/연간 차트 종합",
                "업계 공신력 높음"
            ],
            award_categories=[
                AwardCategory.ARTIST_OF_THE_YEAR,
                AwardCategory.ALBUM_OF_THE_YEAR,
                AwardCategory.SONG_OF_THE_YEAR,
                AwardCategory.ROOKIE_OF_THE_YEAR,
                AwardCategory.GLOBAL_ARTIST
            ],
            expected_viewership=10000000,
            industry_significance=0.85,
            international_reach=0.40,
            broadcast_channels=["MBC", "온라인 스트리밍"],
            venue_capacity=7000,
            voting_start_date=None,
            voting_end_date=None,
            impact_metrics={"global_impact_score": 7.2, "chart_influence": 9.0}
        )
        
        # KBS 가요대축제
        self.award_shows_info[AwardShow.KBS_GAYO_DAECHUKJE] = AwardShowInfo(
            award_show=AwardShow.KBS_GAYO_DAECHUKJE,
            full_name_ko="KBS 가요대축제",
            full_name_en="KBS Song Festival",
            organizer="KBS",
            first_held=1965,
            importance_level=EventImportance.HIGH,
            typical_month=12,
            typical_duration_days=1,
            location_pattern="KBS홀, 서울 주요 공연장",
            characteristics=[
                "한국 최장수 가요 프로그램",
                "연말 대표 음악 축제",
                "특별 무대와 콜라보",
                "공영방송의 권위"
            ],
            award_categories=[
                AwardCategory.ARTIST_OF_THE_YEAR,
                AwardCategory.SONG_OF_THE_YEAR,
                AwardCategory.POPULARITY_AWARD,
                AwardCategory.SPECIAL_ACHIEVEMENT
            ],
            expected_viewership=25000000,
            industry_significance=0.75,
            international_reach=0.30,
            broadcast_channels=["KBS2"],
            venue_capacity=9000,
            voting_start_date=None,
            voting_end_date=None,
            impact_metrics={"global_impact_score": 6.8, "tradition_value": 8.0}
        )
        
        # MBC 가요대제전
        self.award_shows_info[AwardShow.MBC_GAYO_DAEJEJEON] = AwardShowInfo(
            award_show=AwardShow.MBC_GAYO_DAEJEJEON,
            full_name_ko="MBC 가요대제전",
            full_name_en="MBC Gayo Daejejeon",
            organizer="MBC",
            first_held=1966,
            importance_level=EventImportance.HIGH,
            typical_month=12,
            typical_duration_days=1,
            location_pattern="MBC 공개홀, 서울 주요 공연장",
            characteristics=[
                "연말 대표 음악 축제",
                "창의적 무대 연출",
                "아티스트 간 콜라보",
                "시청자 참여 이벤트"
            ],
            award_categories=[
                AwardCategory.ARTIST_OF_THE_YEAR,
                AwardCategory.SONG_OF_THE_YEAR,
                AwardCategory.BEST_DANCE_PERFORMANCE,
                AwardCategory.POPULARITY_AWARD
            ],
            expected_viewership=22000000,
            industry_significance=0.75,
            international_reach=0.30,
            broadcast_channels=["MBC"],
            venue_capacity=8500,
            voting_start_date=None,
            voting_end_date=None,
            impact_metrics={"global_impact_score": 6.5, "entertainment_value": 8.5}
        )
        
        # SBS 가요대전
        self.award_shows_info[AwardShow.SBS_GAYO_DAEJEON] = AwardShowInfo(
            award_show=AwardShow.SBS_GAYO_DAEJEON,
            full_name_ko="SBS 가요대전",
            full_name_en="SBS Gayo Daejeon",
            organizer="SBS",
            first_held=1996,
            importance_level=EventImportance.HIGH,
            typical_month=12,
            typical_duration_days=1,
            location_pattern="SBS 프리즘타워, 고척스카이돔",
            characteristics=[
                "대형 무대와 스펙터클",
                "해외 아티스트 초청",
                "트렌드 반영 빠름",
                "젊은 층 타겟팅"
            ],
            award_categories=[
                AwardCategory.ARTIST_OF_THE_YEAR,
                AwardCategory.SONG_OF_THE_YEAR,
                AwardCategory.GLOBAL_ARTIST,
                AwardCategory.POPULARITY_AWARD
            ],
            expected_viewership=20000000,
            industry_significance=0.75,
            international_reach=0.35,
            broadcast_channels=["SBS"],
            venue_capacity=15000,
            voting_start_date=None,
            voting_end_date=None,
            impact_metrics={"global_impact_score": 7.0, "star_power": 8.8}
        )
        
        self.logger.info(f"Initialized data for {len(self.award_shows_info)} award shows")
    
    def get_award_show_info(self, award_show: AwardShow) -> Optional[AwardShowInfo]:
        """시상식 정보 조회"""
        return self.award_shows_info.get(award_show)
    
    def get_annual_calendar(self, year: int) -> List[Dict[str, Any]]:
        """연간 시상식 달력 생성"""
        annual_calendar = []
        
        for award_show, info in self.award_shows_info.items():
            try:
                # 예상 일정 계산 (실제로는 공식 발표 일정 사용 권장)
                estimated_date = self._estimate_award_date(year, info)
                
                calendar_entry = {
                    'award_show': award_show.value,
                    'name_ko': info.full_name_ko,
                    'name_en': info.full_name_en,
                    'estimated_date': estimated_date.isoformat(),
                    'month': estimated_date.month,
                    'importance': info.importance_level.value,
                    'organizer': info.organizer,
                    'expected_impact': self._calculate_expected_impact(info),
                    'characteristics': info.characteristics
                }
                
                annual_calendar.append(calendar_entry)
                
            except Exception as e:
                self.logger.error(f"Error generating calendar entry for {award_show}: {e}")
        
        # 날짜순 정렬
        annual_calendar.sort(key=lambda x: x['estimated_date'])
        
        self.logger.info(f"Generated annual calendar for {year} with {len(annual_calendar)} events")
        return annual_calendar
    
    def add_award_result(
        self,
        year: int,
        award_show: AwardShow,
        results: List[AwardResult]
    ) -> bool:
        """시상 결과 추가"""
        try:
            key = f"{year}_{award_show.value}"
            
            if key in self.award_results:
                self.logger.warning(f"Award results for {key} already exist, overwriting")
            
            # 결과 검증
            for result in results:
                result.year = year
                result.award_show = award_show
            
            self.award_results[key] = results
            
            self.logger.info(f"Added {len(results)} award results for {award_show.value} {year}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add award results: {e}")
            return False
    
    def get_award_results(
        self,
        year: int,
        award_show: Optional[AwardShow] = None,
        category: Optional[AwardCategory] = None
    ) -> List[AwardResult]:
        """시상 결과 조회"""
        results = []
        
        if award_show:
            # 특정 시상식 결과
            key = f"{year}_{award_show.value}"
            results = self.award_results.get(key, [])
        else:
            # 해당 연도 모든 시상식 결과
            for key, award_results in self.award_results.items():
                if key.startswith(str(year)):
                    results.extend(award_results)
        
        # 카테고리 필터링
        if category:
            results = [r for r in results if r.award_category == category]
        
        self.logger.info(f"Retrieved {len(results)} award results")
        return results
    
    def analyze_award_trends(
        self,
        start_year: int,
        end_year: int,
        category: Optional[AwardCategory] = None
    ) -> Dict[str, Any]:
        """시상 트렌드 분석"""
        try:
            analysis_results = {
                "period": f"{start_year}-{end_year}",
                "category_analyzed": category.value if category else "all",
                "winner_trends": {},
                "award_show_analysis": {},
                "yearly_statistics": {}
            }
            
            # 연도별 수상 데이터 수집
            all_results = []
            for year in range(start_year, end_year + 1):
                year_results = self.get_award_results(year, category=category)
                all_results.extend(year_results)
                
                # 연도별 통계
                analysis_results["yearly_statistics"][str(year)] = {
                    "total_awards": len(year_results),
                    "unique_winners": len(set(r.winner_id for r in year_results)),
                    "award_shows_held": len(set(r.award_show for r in year_results if r.award_show))
                }
            
            if not all_results:
                return {"error": "No award data found for the specified period"}
            
            # 수상자 트렌드 분석
            winner_counts = {}
            for result in all_results:
                winner_counts[result.winner_name] = winner_counts.get(result.winner_name, 0) + 1
            
            # 상위 수상자
            top_winners = sorted(winner_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis_results["winner_trends"]["top_winners"] = [
                {"name": name, "award_count": count} for name, count in top_winners
            ]
            
            # 시상식별 분석
            award_show_stats = {}
            for result in all_results:
                if result.award_show:
                    show_name = result.award_show.value
                    if show_name not in award_show_stats:
                        award_show_stats[show_name] = {
                            "total_awards": 0,
                            "unique_winners": set(),
                            "categories": set()
                        }
                    
                    award_show_stats[show_name]["total_awards"] += 1
                    award_show_stats[show_name]["unique_winners"].add(result.winner_id)
                    award_show_stats[show_name]["categories"].add(result.award_category)
            
            # 시상식별 통계 정리
            for show_name, stats in award_show_stats.items():
                analysis_results["award_show_analysis"][show_name] = {
                    "total_awards": stats["total_awards"],
                    "unique_winners": len(stats["unique_winners"]),
                    "categories_covered": len(stats["categories"]),
                    "average_per_year": round(stats["total_awards"] / (end_year - start_year + 1), 1)
                }
            
            self.logger.info(f"Award trends analysis completed for {start_year}-{end_year}")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in award trends analysis: {e}")
            return {"error": str(e)}
    
    def predict_award_impact(
        self,
        award_show: AwardShow,
        year: int,
        artist_id: str
    ) -> Dict[str, Any]:
        """시상식 수상 영향도 예측"""
        try:
            show_info = self.get_award_show_info(award_show)
            if not show_info:
                return {"error": f"No information found for {award_show.value}"}
            
            # 기본 영향도 계산
            base_impact = show_info.industry_significance * show_info.importance_level.value / 5.0
            
            # 국제적 영향력 고려
            international_boost = show_info.international_reach * 0.3
            
            # 시청자 수 기반 노출 효과
            viewership_impact = min(show_info.expected_viewership / 50000000, 1.0) * 0.2
            
            # 총 예상 영향도
            total_predicted_impact = base_impact + international_boost + viewership_impact
            
            # 예상 효과별 분석
            predicted_effects = {
                "immediate_impact": {
                    "follower_increase_rate": f"{total_predicted_impact * 15:.1f}%",
                    "streaming_boost": f"{total_predicted_impact * 25:.1f}%",
                    "search_trend_spike": f"{total_predicted_impact * 50:.1f}%"
                },
                "duration_impact": {
                    "peak_effect_days": int(7 + total_predicted_impact * 10),
                    "sustained_effect_weeks": int(2 + total_predicted_impact * 4)
                },
                "career_impact": {
                    "brand_value_increase": f"{total_predicted_impact * 10:.1f}%",
                    "collaboration_opportunities": "증가" if total_predicted_impact > 0.7 else "보통",
                    "international_recognition": "높음" if show_info.international_reach > 0.7 else "보통"
                }
            }
            
            return {
                "award_show": award_show.value,
                "year": year,
                "artist_id": artist_id,
                "predicted_impact_score": round(total_predicted_impact, 3),
                "impact_category": self._categorize_impact_level(total_predicted_impact),
                "predicted_effects": predicted_effects,
                "analysis_factors": {
                    "industry_significance": show_info.industry_significance,
                    "international_reach": show_info.international_reach,
                    "expected_viewership": show_info.expected_viewership,
                    "award_show_importance": show_info.importance_level.value
                },
                "recommendations": self._generate_award_recommendations(show_info, total_predicted_impact)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting award impact: {e}")
            return {"error": str(e)}
    
    def generate_award_season_calendar(self, year: int) -> Dict[str, Any]:
        """시상식 시즌 종합 달력 생성"""
        try:
            annual_calendar = self.get_annual_calendar(year)
            
            # 월별 그룹화
            monthly_calendar = {}
            for event in annual_calendar:
                month = event['month']
                if month not in monthly_calendar:
                    monthly_calendar[month] = []
                monthly_calendar[month].append(event)
            
            # 시상식 시즌 분석
            season_analysis = {
                "peak_months": [],
                "total_events": len(annual_calendar),
                "highest_impact_month": None,
                "preparation_timeline": {},
                "strategic_considerations": []
            }
            
            # 월별 영향도 계산
            monthly_impact = {}
            for month, events in monthly_calendar.items():
                total_impact = sum(event['expected_impact'] for event in events)
                monthly_impact[month] = total_impact
                
                if total_impact > 2.0:  # 높은 영향도 기준
                    season_analysis["peak_months"].append({
                        "month": month,
                        "events_count": len(events),
                        "total_impact": round(total_impact, 2),
                        "major_events": [e['name_ko'] for e in events if e['importance'] >= 4]
                    })
            
            # 최고 영향도 월 식별
            if monthly_impact:
                peak_month = max(monthly_impact.keys(), key=lambda x: monthly_impact[x])
                season_analysis["highest_impact_month"] = {
                    "month": peak_month,
                    "impact_score": round(monthly_impact[peak_month], 2),
                    "events": monthly_calendar[peak_month]
                }
            
            # 준비 타임라인 생성
            for month, events in monthly_calendar.items():
                prep_start_month = month - 3 if month > 3 else month + 9
                season_analysis["preparation_timeline"][f"month_{month}"] = {
                    "award_events": len(events),
                    "preparation_start": f"month_{prep_start_month}",
                    "key_preparation_tasks": [
                        "음반/싱글 발매 계획 수립",
                        "프로모션 전략 기획",
                        "무대 준비 및 연습"
                    ]
                }
            
            # 전략적 고려사항
            if len(monthly_calendar.get(11, [])) > 0 or len(monthly_calendar.get(12, [])) > 0:
                season_analysis["strategic_considerations"].append(
                    "11-12월 시상식 집중 기간, 연말 프로모션 전략 필수"
                )
            
            if len(monthly_calendar.get(1, [])) > 0:
                season_analysis["strategic_considerations"].append(
                    "1월 신년 시상식 시즌, 전년도 성과 어필 중요"
                )
            
            return {
                "year": year,
                "monthly_calendar": monthly_calendar,
                "season_analysis": season_analysis,
                "total_events": len(annual_calendar),
                "peak_season_months": [11, 12, 1],  # 일반적인 시상식 시즌
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating award season calendar: {e}")
            return {"error": str(e)}
    
    def _estimate_award_date(self, year: int, info: AwardShowInfo) -> date:
        """시상식 예상 일정 계산"""
        # 간단한 추정 로직 (실제로는 과거 패턴 분석 필요)
        month = info.typical_month
        
        # 월별 대표 일정
        typical_day_map = {
            1: 15,   # 1월 중순
            11: 25,  # 11월 말
            12: 28   # 12월 말
        }
        
        day = typical_day_map.get(month, 15)
        
        try:
            return date(year, month, day)
        except ValueError:
            # 날짜가 유효하지 않은 경우 (예: 2월 30일)
            return date(year, month, min(day, 28))
    
    def _calculate_expected_impact(self, info: AwardShowInfo) -> float:
        """시상식 예상 영향도 계산"""
        # 중요도, 업계 영향력, 국제적 영향력을 종합
        impact = (
            info.importance_level.value / 5.0 * 0.4 +
            info.industry_significance * 0.3 +
            info.international_reach * 0.3
        )
        return round(impact, 3)
    
    def _categorize_impact_level(self, impact_score: float) -> str:
        """영향도 수준 분류"""
        if impact_score >= 0.8:
            return "매우 높음"
        elif impact_score >= 0.6:
            return "높음"
        elif impact_score >= 0.4:
            return "보통"
        elif impact_score >= 0.2:
            return "낮음"
        else:
            return "매우 낮음"
    
    def _generate_award_recommendations(self, info: AwardShowInfo, impact_score: float) -> List[str]:
        """시상식 관련 추천사항 생성"""
        recommendations = []
        
        if impact_score > 0.7:
            recommendations.append("핵심 프로모션 대상으로 집중 투자 권장")
            
        if info.international_reach > 0.8:
            recommendations.append("글로벌 마케팅 전략과 연계 필요")
            
        if info.typical_duration_days > 1:
            recommendations.append("다일간 행사이므로 충분한 준비 기간 확보 필요")
            
        if len(info.award_categories) > 5:
            recommendations.append("다양한 부문 수상 가능성 검토")
            
        return recommendations
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        total_results = sum(len(results) for results in self.award_results.values())
        
        return {
            "award_shows_registered": len(self.award_shows_info),
            "total_award_results": total_results,
            "years_covered": len(set(key.split('_')[0] for key in self.award_results.keys())),
            "award_shows_with_data": len(set(key.split('_')[1] for key in self.award_results.keys())),
            "most_covered_show": max(
                (show.value for show in AwardShow),
                key=lambda s: sum(1 for key in self.award_results.keys() if key.endswith(s)),
                default="none"
            ) if self.award_results else "none",
            "system_initialized": datetime.now().isoformat()
        }    
    def get_all_award_shows(self) -> Dict[str, AwardShowInfo]:
        """
        모든 등록된 시상식 정보를 반환
        
        Returns:
            Dict[str, AwardShowInfo]: 시상식명을 키로 하는 시상식 정보 딕셔너리
        """
        try:
            result = {}
            
            for award_show, award_info in self.award_shows_info.items():
                # AwardShow enum을 문자열로 변환 (한국어명 사용)
                show_name = self._get_korean_name(award_show)
                result[show_name] = award_info
                
            self.logger.info(f"Retrieved {len(result)} award shows")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get all award shows: {e}")
            return {}
    
    def _get_korean_name(self, award_show: AwardShow) -> str:
        """AwardShow enum을 한국어 이름으로 변환"""
        korean_names = {
            AwardShow.MAMA: "Mnet 아시안 뮤직 어워즈 (MAMA)",
            AwardShow.GOLDEN_DISC: "골든 디스크 어워즈",
            AwardShow.SEOUL_MUSIC_AWARDS: "서울뮤직어워드",
            AwardShow.GAON_CHART: "가온 차트 뮤직 어워즈 (써클차트)",
            AwardShow.KBS_GAYO_DAECHUKJE: "KBS 가요대축제",
            AwardShow.MBC_GAYO_DAEJEJEON: "MBC 가요대제전",
            AwardShow.SBS_GAYO_DAEJEON: "SBS 가요대전",
            AwardShow.KOREAN_MUSIC_AWARDS: "한국대중음악상",
            AwardShow.MUSIC_BANK_YEAR_END: "뮤직뱅크 연말결산",
            AwardShow.INKIGAYO_YEAR_END: "인기가요 연말결산",
            AwardShow.KOREAN_HIP_HOP_AWARDS: "한국힙합어워드"
        }
        return korean_names.get(award_show, award_show.value.replace('_', ' ').title())
    
    def get_award_shows_by_month(self, month: int) -> Dict[str, AwardShowInfo]:
        """
        특정 월에 열리는 시상식들을 반환
        
        Args:
            month (int): 조회할 월 (1-12)
            
        Returns:
            Dict[str, AwardShowInfo]: 해당 월 시상식 정보
        """
        try:
            result = {}
            current_year = datetime.now().year
            
            for award_show, award_info in self.award_shows_info.items():
                estimated_date = self._estimate_award_date(current_year, award_info)
                
                if estimated_date.month == month:
                    show_name = self._get_korean_name(award_show)
                    result[show_name] = award_info
                    
            self.logger.info(f"Retrieved {len(result)} award shows for month {month}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get award shows for month {month}: {e}")
            return {}
    
    def get_upcoming_award_shows(self, days_ahead: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        앞으로 개최될 시상식들을 반환
        
        Args:
            days_ahead (int): 조회할 기간 (일 단위, 기본 30일)
            
        Returns:
            Dict[str, Dict[str, Any]]: 예정된 시상식 정보 (날짜 포함)
        """
        try:
            result = {}
            current_year = datetime.now().year
            today = date.today()
            
            for award_show, award_info in self.award_shows_info.items():
                estimated_date = self._estimate_award_date(current_year, award_info)
                
                # 앞으로 days_ahead일 내에 열리는지 확인
                days_until = (estimated_date - today).days
                if 0 <= days_until <= days_ahead:
                    show_name = self._get_korean_name(award_show)
                    result[show_name] = {
                        'info': award_info,
                        'estimated_date': estimated_date,
                        'days_until': days_until,
                        'is_major': award_info.importance == EventImportance.VERY_HIGH
                    }
                    
            # 날짜순으로 정렬
            result = dict(sorted(result.items(), key=lambda x: x[1]['estimated_date']))
                    
            self.logger.info(f"Retrieved {len(result)} upcoming award shows")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get upcoming award shows: {e}")
            return {}
