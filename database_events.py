"""
K-Pop Event System Database Access Layer
이벤트 관리 시스템을 위한 PostgreSQL 데이터베이스 인터페이스
"""

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor, Json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import asdict
import json

from .analytics.kpop_event_calendar import KPopEvent, EventCategory, EventImportance
from .analytics.award_shows_data import AwardShowInfo, AwardResult
from .analytics.comeback_season_analyzer import ComebackAnalysis
from .analytics.event_impact_analyzer import EventImpactAnalysis, ImpactMeasurement

logger = logging.getLogger(__name__)

class EventDatabaseManager:
    """이벤트 관리 시스템 데이터베이스 매니저"""
    
    def __init__(self, connection_pool):
        """
        데이터베이스 연결 풀을 사용하여 초기화
        
        Args:
            connection_pool: psycopg2 connection pool
        """
        self.pool = connection_pool

    def get_connection(self):
        """데이터베이스 연결 반환"""
        return self.pool.getconn()

    def return_connection(self, conn):
        """데이터베이스 연결 반환"""
        self.pool.putconn(conn)

    # ==================== 이벤트 카테고리 관리 ====================
    
    def get_event_categories(self) -> List[Dict[str, Any]]:
        """모든 이벤트 카테고리 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM event_categories 
                    ORDER BY importance_weight DESC, name
                """)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"이벤트 카테고리 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    def create_event_category(self, name: str, description: str, importance_weight: float = 1.0) -> Optional[int]:
        """새 이벤트 카테고리 생성"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO event_categories (name, description, importance_weight)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (name, description, importance_weight))
                category_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"이벤트 카테고리 생성 완료: {name} (ID: {category_id})")
                return category_id
        except Exception as e:
            conn.rollback()
            logger.error(f"이벤트 카테고리 생성 실패: {e}")
            return None
        finally:
            self.return_connection(conn)

    # ==================== K-Pop 이벤트 관리 ====================
    
    def create_kpop_event(self, event: KPopEvent) -> Optional[int]:
        """K-Pop 이벤트 생성"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 이벤트 카테고리 ID 조회
                category_id = self._get_category_id_by_name(cur, event.category.value)
                
                cur.execute("""
                    INSERT INTO kpop_events 
                    (name, event_type, category_id, date, end_date, venue, description, 
                     importance_level, global_impact_score, is_annual, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    event.name,
                    event.event_type,
                    category_id,
                    event.date,
                    event.end_date,
                    event.venue,
                    event.description,
                    event.importance.value,
                    event.global_impact_score,
                    event.is_annual,
                    Json(event.metadata)
                ))
                event_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"K-Pop 이벤트 생성 완료: {event.name} (ID: {event_id})")
                return event_id
        except Exception as e:
            conn.rollback()
            logger.error(f"K-Pop 이벤트 생성 실패: {e}")
            return None
        finally:
            self.return_connection(conn)

    def get_kpop_events(self, start_date: Optional[date] = None, end_date: Optional[date] = None,
                       event_type: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """K-Pop 이벤트 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM v_events_full_details WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND date >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date <= %s"
                    params.append(end_date)
                
                if event_type:
                    query += " AND event_type = %s"
                    params.append(event_type)
                
                if category:
                    query += " AND category_name = %s"
                    params.append(category)
                
                query += " ORDER BY date DESC"
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"K-Pop 이벤트 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    def get_upcoming_events(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """다가오는 이벤트 조회"""
        end_date = date.today() + timedelta(days=days_ahead)
        return self.get_kpop_events(start_date=date.today(), end_date=end_date)

    def get_events_by_artist(self, artist_id: int, start_date: Optional[date] = None,
                           end_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """특정 아티스트의 이벤트 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT e.*, ce.comeback_type, ce.promotion_period, ce.competition_level
                    FROM v_events_full_details e
                    LEFT JOIN comeback_events ce ON e.id = ce.event_id
                    WHERE ce.artist_id = %s
                """
                params = [artist_id]
                
                if start_date:
                    query += " AND e.date >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND e.date <= %s"
                    params.append(end_date)
                
                query += " ORDER BY e.date DESC"
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"아티스트 이벤트 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    # ==================== 시상식 관리 ====================
    
    def create_award_show(self, event_id: int, award_info: AwardShowInfo) -> Optional[int]:
        """시상식 상세 정보 생성"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO award_shows 
                    (event_id, organizer, venue_capacity, broadcast_channels, 
                     voting_start_date, voting_end_date, nomination_announcement_date,
                     award_categories, historical_winners, impact_metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    event_id,
                    award_info.organizer,
                    award_info.venue_capacity,
                    award_info.broadcast_channels,
                    award_info.voting_start_date,
                    award_info.voting_end_date,
                    award_info.nomination_announcement_date,
                    Json(award_info.award_categories),
                    Json(award_info.historical_winners),
                    Json(award_info.impact_metrics)
                ))
                award_show_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"시상식 정보 생성 완료: {award_info.organizer} (ID: {award_show_id})")
                return award_show_id
        except Exception as e:
            conn.rollback()
            logger.error(f"시상식 정보 생성 실패: {e}")
            return None
        finally:
            self.return_connection(conn)

    def get_award_shows(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """시상식 정보 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT e.*, aws.*
                    FROM kpop_events e
                    JOIN award_shows aws ON e.id = aws.event_id
                    WHERE e.event_type = 'award_ceremony'
                """
                params = []
                
                if year:
                    query += " AND EXTRACT(YEAR FROM e.date) = %s"
                    params.append(year)
                
                query += " ORDER BY e.date"
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"시상식 정보 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    # ==================== 컴백 이벤트 관리 ====================
    
    def create_comeback_event(self, event_id: int, artist_id: Optional[int] = None,
                            group_id: Optional[int] = None, album_id: Optional[int] = None,
                            comeback_type: str = 'single', promotion_period: int = 30,
                            competition_level: int = 3, expected_impact_score: float = 0.0) -> Optional[int]:
        """컴백 이벤트 상세 정보 생성"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO comeback_events 
                    (event_id, artist_id, group_id, album_id, comeback_type, 
                     promotion_period, competition_level, expected_impact_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    event_id, artist_id, group_id, album_id, comeback_type,
                    promotion_period, competition_level, expected_impact_score
                ))
                comeback_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"컴백 이벤트 생성 완료: (ID: {comeback_id})")
                return comeback_id
        except Exception as e:
            conn.rollback()
            logger.error(f"컴백 이벤트 생성 실패: {e}")
            return None
        finally:
            self.return_connection(conn)

    def get_comeback_events(self, start_date: Optional[date] = None, end_date: Optional[date] = None,
                          artist_id: Optional[int] = None, group_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """컴백 이벤트 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM v_comeback_events_details WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND comeback_date >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND comeback_date <= %s"
                    params.append(end_date)
                
                if artist_id:
                    query += " AND artist_id = %s"
                    params.append(artist_id)
                
                if group_id:
                    query += " AND group_id = %s"
                    params.append(group_id)
                
                query += " ORDER BY comeback_date DESC"
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"컴백 이벤트 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    # ==================== 이벤트 영향도 측정 ====================
    
    def record_event_impact(self, event_id: int, artist_id: int, metric_type: str,
                          platform: str, before_value: int, after_value: int,
                          measurement_period: int = 7, statistical_significance: float = 0.0,
                          confidence_level: float = 0.95) -> Optional[int]:
        """이벤트 영향도 측정 결과 기록"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                impact_percentage = ((after_value - before_value) / before_value * 100) if before_value > 0 else 0
                
                cur.execute("""
                    INSERT INTO event_impact_measurements 
                    (event_id, artist_id, metric_type, platform, before_value, after_value,
                     impact_percentage, measurement_period, statistical_significance, confidence_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    event_id, artist_id, metric_type, platform, before_value, after_value,
                    impact_percentage, measurement_period, statistical_significance, confidence_level
                ))
                measurement_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"이벤트 영향도 측정 기록 완료: (ID: {measurement_id})")
                return measurement_id
        except Exception as e:
            conn.rollback()
            logger.error(f"이벤트 영향도 측정 기록 실패: {e}")
            return None
        finally:
            self.return_connection(conn)

    def get_event_impact_analysis(self, event_id: int) -> List[Dict[str, Any]]:
        """이벤트 영향도 분석 결과 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM v_event_impact_analysis 
                    WHERE event_id = %s
                    ORDER BY impact_percentage DESC
                """, (event_id,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"이벤트 영향도 분석 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    def get_artist_event_impacts(self, artist_id: int, start_date: Optional[date] = None,
                               end_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """특정 아티스트의 이벤트 영향도 기록 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT * FROM v_event_impact_analysis 
                    WHERE artist_id = %s
                """
                params = [artist_id]
                
                if start_date:
                    query += " AND event_date >= %s"
                    params.append(start_date)
                
                if end_date:
                    query += " AND event_date <= %s"
                    params.append(end_date)
                
                query += " ORDER BY event_date DESC, impact_percentage DESC"
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"아티스트 이벤트 영향도 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    # ==================== 계절별 패턴 분석 ====================
    
    def save_seasonal_pattern(self, season: str, month: int, event_category_id: int,
                            pattern_type: str, intensity_score: float,
                            historical_data: Dict[str, Any], trend_direction: str,
                            analysis_year: int) -> Optional[int]:
        """계절별 패턴 분석 결과 저장"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO seasonal_patterns 
                    (season, month, event_category_id, pattern_type, intensity_score,
                     historical_data, trend_direction, analysis_year)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    season, month, event_category_id, pattern_type, intensity_score,
                    Json(historical_data), trend_direction, analysis_year
                ))
                pattern_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"계절별 패턴 저장 완료: {season}-{pattern_type} (ID: {pattern_id})")
                return pattern_id
        except Exception as e:
            conn.rollback()
            logger.error(f"계절별 패턴 저장 실패: {e}")
            return None
        finally:
            self.return_connection(conn)

    def get_seasonal_patterns(self, season: Optional[str] = None, pattern_type: Optional[str] = None,
                            analysis_year: Optional[int] = None) -> List[Dict[str, Any]]:
        """계절별 패턴 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT sp.*, ec.name as category_name, ec.description as category_description
                    FROM seasonal_patterns sp
                    LEFT JOIN event_categories ec ON sp.event_category_id = ec.id
                    WHERE 1=1
                """
                params = []
                
                if season:
                    query += " AND sp.season = %s"
                    params.append(season)
                
                if pattern_type:
                    query += " AND sp.pattern_type = %s"
                    params.append(pattern_type)
                
                if analysis_year:
                    query += " AND sp.analysis_year = %s"
                    params.append(analysis_year)
                
                query += " ORDER BY sp.analysis_year DESC, sp.month, sp.intensity_score DESC"
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"계절별 패턴 조회 실패: {e}")
            return []
        finally:
            self.return_connection(conn)

    # ==================== 예측 모델 결과 ====================
    
    def save_event_prediction(self, event_id: int, prediction_type: str, predicted_value: float,
                            confidence_interval_lower: float, confidence_interval_upper: float,
                            model_version: str = "1.0") -> Optional[int]:
        """이벤트 예측 결과 저장"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO event_predictions 
                    (event_id, prediction_type, predicted_value, confidence_interval_lower,
                     confidence_interval_upper, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    event_id, prediction_type, predicted_value,
                    confidence_interval_lower, confidence_interval_upper, model_version
                ))
                prediction_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"이벤트 예측 저장 완료: {prediction_type} (ID: {prediction_id})")
                return prediction_id
        except Exception as e:
            conn.rollback()
            logger.error(f"이벤트 예측 저장 실패: {e}")
            return None
        finally:
            self.return_connection(conn)

    def update_prediction_accuracy(self, prediction_id: int, actual_value: float) -> bool:
        """예측 정확도 업데이트 (실제 값 발생 후)"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 예측 정확도 계산
                cur.execute("""
                    UPDATE event_predictions 
                    SET actual_value = %s,
                        prediction_accuracy = 1 - ABS(predicted_value - %s) / GREATEST(predicted_value, %s)
                    WHERE id = %s
                """, (actual_value, actual_value, actual_value, prediction_id))
                conn.commit()
                logger.info(f"예측 정확도 업데이트 완료: (ID: {prediction_id})")
                return True
        except Exception as e:
            conn.rollback()
            logger.error(f"예측 정확도 업데이트 실패: {e}")
            return False
        finally:
            self.return_connection(conn)

    # ==================== 헬퍼 메서드 ====================
    
    def _get_category_id_by_name(self, cursor, category_name: str) -> Optional[int]:
        """카테고리 이름으로 ID 조회"""
        cursor.execute("SELECT id FROM event_categories WHERE name = %s", (category_name,))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_event_statistics(self) -> Dict[str, Any]:
        """이벤트 통계 정보 조회"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 총 이벤트 수
                cur.execute("SELECT COUNT(*) as total_events FROM kpop_events WHERE is_active = true")
                total_events = cur.fetchone()['total_events']
                
                # 이번 달 이벤트 수
                cur.execute("""
                    SELECT COUNT(*) as monthly_events 
                    FROM kpop_events 
                    WHERE is_active = true 
                    AND EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE)
                    AND EXTRACT(MONTH FROM date) = EXTRACT(MONTH FROM CURRENT_DATE)
                """)
                monthly_events = cur.fetchone()['monthly_events']
                
                # 카테고리별 이벤트 수
                cur.execute("""
                    SELECT ec.name, COUNT(e.id) as event_count
                    FROM event_categories ec
                    LEFT JOIN kpop_events e ON ec.id = e.category_id AND e.is_active = true
                    GROUP BY ec.id, ec.name
                    ORDER BY event_count DESC
                """)
                category_stats = [dict(row) for row in cur.fetchall()]
                
                # 다가오는 이벤트 수 (30일 이내)
                cur.execute("""
                    SELECT COUNT(*) as upcoming_events
                    FROM kpop_events 
                    WHERE is_active = true 
                    AND date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
                """)
                upcoming_events = cur.fetchone()['upcoming_events']
                
                return {
                    'total_events': total_events,
                    'monthly_events': monthly_events,
                    'upcoming_events': upcoming_events,
                    'category_stats': category_stats
                }
        except Exception as e:
            logger.error(f"이벤트 통계 조회 실패: {e}")
            return {}
        finally:
            self.return_connection(conn)