"""
이벤트 시스템 통합 모듈
기존 K-Pop 대시보드와 새로운 이벤트 관리 시스템을 연결하는 통합 레이어
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import asdict
import pandas as pd

from ..database_postgresql import DatabaseManager
from ..database_events import EventDatabaseManager
from ..analytics.kpop_event_calendar import (
    KPopEventCalendar, KPopEvent, EventCategory, EventImportance
)
from ..analytics.award_shows_data import AwardShowDataManager, AwardShowInfo
from ..analytics.comeback_season_analyzer import ComebackSeasonAnalyzer, ComebackSeason
from ..analytics.event_impact_analyzer import EventImpactAnalyzer, ImpactType

logger = logging.getLogger(__name__)

class EventSystemIntegration:
    """이벤트 시스템과 기존 시스템 통합 관리자"""
    
    def __init__(self, db_manager: DatabaseManager, event_db_manager: EventDatabaseManager):
        """
        통합 시스템 초기화
        
        Args:
            db_manager: 기존 데이터베이스 매니저
            event_db_manager: 이벤트 데이터베이스 매니저
        """
        self.db_manager = db_manager
        self.event_db_manager = event_db_manager
        
        # 분석 시스템 초기화
        self.event_calendar = KPopEventCalendar()
        self.award_manager = AwardShowDataManager()
        self.comeback_analyzer = ComebackSeasonAnalyzer()
        self.impact_analyzer = EventImpactAnalyzer()
        
        logger.info("이벤트 시스템 통합 모듈 초기화 완료")

    # ==================== 데이터 동기화 ====================
    
    def sync_artists_with_events(self) -> Dict[str, Any]:
        """아티스트 데이터와 이벤트 데이터 동기화"""
        try:
            # 기존 아티스트 데이터 조회
            artists = self.db_manager.get_all_artists()
            
            sync_results = {
                "synced_artists": 0,
                "created_events": 0,
                "updated_events": 0,
                "errors": []
            }
            
            for artist in artists:
                try:
                    # 아티스트별 이벤트 생성/업데이트
                    artist_events = self._create_artist_events(artist)
                    sync_results["synced_artists"] += 1
                    sync_results["created_events"] += len(artist_events)
                    
                except Exception as e:
                    error_msg = f"아티스트 {artist.get('name', 'Unknown')} 동기화 실패: {e}"
                    logger.error(error_msg)
                    sync_results["errors"].append(error_msg)
            
            logger.info(f"아티스트-이벤트 동기화 완료: {sync_results}")
            return sync_results
            
        except Exception as e:
            logger.error(f"아티스트-이벤트 동기화 실패: {e}")
            return {"error": str(e)}

    def sync_platform_metrics_with_impact(self, start_date: Optional[date] = None,
                                         end_date: Optional[date] = None) -> Dict[str, Any]:
        """플랫폼 메트릭스와 이벤트 영향도 동기화"""
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=30)
            if not end_date:
                end_date = date.today()
            
            # 해당 기간의 이벤트 조회
            events = self.event_db_manager.get_kpop_events(start_date, end_date)
            
            sync_results = {
                "processed_events": 0,
                "impact_measurements": 0,
                "errors": []
            }
            
            for event in events:
                try:
                    # 이벤트 영향도 측정 및 기록
                    measurements = self._measure_event_impact(event, start_date, end_date)
                    sync_results["impact_measurements"] += len(measurements)
                    sync_results["processed_events"] += 1
                    
                except Exception as e:
                    error_msg = f"이벤트 {event.get('name', 'Unknown')} 영향도 측정 실패: {e}"
                    logger.error(error_msg)
                    sync_results["errors"].append(error_msg)
            
            logger.info(f"메트릭스-영향도 동기화 완료: {sync_results}")
            return sync_results
            
        except Exception as e:
            logger.error(f"메트릭스-영향도 동기화 실패: {e}")
            return {"error": str(e)}

    # ==================== 통합 분석 기능 ====================
    
    def analyze_artist_event_performance(self, artist_id: int, 
                                       analysis_period: int = 90) -> Dict[str, Any]:
        """아티스트의 이벤트 성과 종합 분석"""
        try:
            # 분석 기간 설정
            end_date = date.today()
            start_date = end_date - timedelta(days=analysis_period)
            
            # 기본 아티스트 정보 조회
            artist_info = self.db_manager.get_artist_by_id(artist_id)
            if not artist_info:
                return {"error": "아티스트를 찾을 수 없습니다."}
            
            # 아티스트 이벤트 조회
            artist_events = self.event_db_manager.get_events_by_artist(
                artist_id, start_date, end_date
            )
            
            # 영향도 데이터 조회
            impact_data = self.event_db_manager.get_artist_event_impacts(
                artist_id, start_date, end_date
            )
            
            # 플랫폼 메트릭스 조회
            platform_metrics = self.db_manager.get_platform_metrics_by_artist(
                artist_id, start_date, end_date
            )
            
            # 종합 분석 수행
            analysis_result = {
                "artist_info": artist_info,
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": analysis_period
                },
                "event_summary": self._analyze_event_summary(artist_events),
                "impact_analysis": self._analyze_impact_performance(impact_data),
                "platform_performance": self._analyze_platform_performance(platform_metrics),
                "recommendations": self._generate_artist_recommendations(
                    artist_events, impact_data, platform_metrics
                )
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"아티스트 이벤트 성과 분석 실패: {e}")
            return {"error": str(e)}

    def get_market_competition_analysis(self, target_date: date,
                                      analysis_window: int = 14) -> Dict[str, Any]:
        """특정 날짜 주변의 시장 경쟁 상황 분석"""
        try:
            # 분석 기간 설정
            start_date = target_date - timedelta(days=analysis_window)
            end_date = target_date + timedelta(days=analysis_window)
            
            # 해당 기간의 모든 이벤트 조회
            all_events = self.event_db_manager.get_kpop_events(start_date, end_date)
            
            # 컴백 분석기를 사용한 경쟁 분석
            competition_analysis = self.comeback_analyzer.analyze_competition_level(
                target_date, all_events
            )
            
            # 카테고리별 이벤트 분포
            category_distribution = self._analyze_category_distribution(all_events)
            
            # 아티스트 등급별 분포
            tier_distribution = self._analyze_artist_tier_distribution(all_events)
            
            # 예상 시장 임팩트
            market_impact = self._calculate_market_impact(all_events, target_date)
            
            return {
                "target_date": target_date.isoformat(),
                "analysis_window": analysis_window,
                "competition_level": competition_analysis,
                "category_distribution": category_distribution,
                "tier_distribution": tier_distribution,
                "market_impact": market_impact,
                "total_events": len(all_events),
                "recommendations": self._generate_competition_recommendations(
                    competition_analysis, category_distribution
                )
            }
            
        except Exception as e:
            logger.error(f"시장 경쟁 분석 실패: {e}")
            return {"error": str(e)}

    def predict_event_success(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """이벤트 성공 가능성 예측"""
        try:
            # 이벤트 정보 파싱
            event_date = datetime.strptime(event_data["date"], "%Y-%m-%d").date()
            event_category = EventCategory(event_data["category"])
            artist_id = event_data.get("artist_id")
            
            # 기존 유사 이벤트 데이터 조회
            similar_events = self._find_similar_events(event_data)
            
            # 계절성 분석
            seasonal_analysis = self.comeback_analyzer.analyze_seasonal_trends(
                event_date.month, event_category.value
            )
            
            # 아티스트 과거 성과 분석
            artist_performance = None
            if artist_id:
                artist_performance = self._analyze_artist_historical_performance(artist_id)
            
            # 경쟁 상황 분석
            competition_analysis = self.get_market_competition_analysis(event_date)
            
            # 예측 모델 실행
            prediction_result = self._run_prediction_model(
                event_data, similar_events, seasonal_analysis, 
                artist_performance, competition_analysis
            )
            
            return {
                "event_data": event_data,
                "prediction": prediction_result,
                "supporting_analysis": {
                    "similar_events_count": len(similar_events),
                    "seasonal_factor": seasonal_analysis,
                    "artist_factor": artist_performance,
                    "competition_factor": competition_analysis.get("competition_level", {})
                },
                "confidence_level": prediction_result.get("confidence", 0.0),
                "recommendations": self._generate_prediction_recommendations(prediction_result)
            }
            
        except Exception as e:
            logger.error(f"이벤트 성공 예측 실패: {e}")
            return {"error": str(e)}

    # ==================== 헬퍼 메서드 ====================
    
    def _create_artist_events(self, artist: Dict[str, Any]) -> List[int]:
        """아티스트별 이벤트 생성"""
        created_events = []
        
        # 컴백 이벤트 생성 (앨범 정보 기반)
        albums = self.db_manager.get_albums_by_artist(artist["id"])
        for album in albums:
            if album.get("release_date"):
                event = KPopEvent(
                    name=f"{artist['name']} - {album['title']} 컴백",
                    event_type="comeback",
                    category=EventCategory.COMEBACK,
                    date=album["release_date"],
                    venue="Online",
                    description=f"{artist['name']}의 {album['title']} 발매",
                    importance=EventImportance.SIGNIFICANT,
                    metadata={
                        "artist_id": artist["id"],
                        "album_id": album["id"],
                        "album_type": album.get("type", "unknown")
                    }
                )
                
                event_id = self.event_db_manager.create_kpop_event(event)
                if event_id:
                    created_events.append(event_id)
        
        return created_events

    def _measure_event_impact(self, event: Dict[str, Any], 
                            start_date: date, end_date: date) -> List[int]:
        """이벤트 영향도 측정"""
        measurements = []
        
        # 이벤트와 연관된 아티스트 찾기
        artist_id = event.get("metadata", {}).get("artist_id")
        if not artist_id:
            return measurements
        
        # 이벤트 전후 플랫폼 메트릭스 비교
        event_date = event["date"]
        before_date = event_date - timedelta(days=7)
        after_date = event_date + timedelta(days=7)
        
        # 각 플랫폼별 영향도 측정
        platforms = ["youtube", "spotify", "instagram", "twitter"]
        for platform in platforms:
            try:
                before_metrics = self.db_manager.get_platform_metrics_by_date(
                    artist_id, platform, before_date
                )
                after_metrics = self.db_manager.get_platform_metrics_by_date(
                    artist_id, platform, after_date
                )
                
                if before_metrics and after_metrics:
                    for metric_type in ["subscribers", "followers", "views"]:
                        before_value = before_metrics.get(metric_type, 0)
                        after_value = after_metrics.get(metric_type, 0)
                        
                        if before_value > 0:
                            measurement_id = self.event_db_manager.record_event_impact(
                                event_id=event["id"],
                                artist_id=artist_id,
                                metric_type=metric_type,
                                platform=platform,
                                before_value=before_value,
                                after_value=after_value
                            )
                            if measurement_id:
                                measurements.append(measurement_id)
            
            except Exception as e:
                logger.error(f"플랫폼 {platform} 영향도 측정 실패: {e}")
        
        return measurements

    def _analyze_event_summary(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """이벤트 요약 분석"""
        if not events:
            return {"total_events": 0}
        
        total_events = len(events)
        category_counts = {}
        importance_distribution = {}
        
        for event in events:
            # 카테고리별 집계
            category = event.get("category_name", "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # 중요도별 집계
            importance = event.get("importance_level", 0)
            importance_distribution[importance] = importance_distribution.get(importance, 0) + 1
        
        return {
            "total_events": total_events,
            "category_distribution": category_counts,
            "importance_distribution": importance_distribution,
            "avg_importance": sum(e.get("importance_level", 0) for e in events) / total_events
        }

    def _analyze_impact_performance(self, impact_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """영향도 성과 분석"""
        if not impact_data:
            return {"total_measurements": 0}
        
        total_measurements = len(impact_data)
        platform_performance = {}
        metric_performance = {}
        
        total_impact = 0
        positive_impacts = 0
        
        for measurement in impact_data:
            impact_pct = measurement.get("impact_percentage", 0)
            platform = measurement.get("platform", "Unknown")
            metric_type = measurement.get("metric_type", "Unknown")
            
            # 플랫폼별 성과
            if platform not in platform_performance:
                platform_performance[platform] = {"total": 0, "count": 0, "avg": 0}
            platform_performance[platform]["total"] += impact_pct
            platform_performance[platform]["count"] += 1
            
            # 지표별 성과
            if metric_type not in metric_performance:
                metric_performance[metric_type] = {"total": 0, "count": 0, "avg": 0}
            metric_performance[metric_type]["total"] += impact_pct
            metric_performance[metric_type]["count"] += 1
            
            total_impact += impact_pct
            if impact_pct > 0:
                positive_impacts += 1
        
        # 평균 계산
        for platform_data in platform_performance.values():
            if platform_data["count"] > 0:
                platform_data["avg"] = platform_data["total"] / platform_data["count"]
        
        for metric_data in metric_performance.values():
            if metric_data["count"] > 0:
                metric_data["avg"] = metric_data["total"] / metric_data["count"]
        
        return {
            "total_measurements": total_measurements,
            "avg_impact": total_impact / total_measurements if total_measurements > 0 else 0,
            "positive_impact_rate": positive_impacts / total_measurements if total_measurements > 0 else 0,
            "platform_performance": platform_performance,
            "metric_performance": metric_performance
        }

    def _analyze_platform_performance(self, platform_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """플랫폼 성과 분석"""
        if not platform_metrics:
            return {"total_records": 0}
        
        platform_summary = {}
        
        for metric in platform_metrics:
            platform = metric.get("platform", "Unknown")
            metric_type = metric.get("metric_type", "Unknown")
            value = metric.get("value", 0)
            
            if platform not in platform_summary:
                platform_summary[platform] = {}
            
            if metric_type not in platform_summary[platform]:
                platform_summary[platform][metric_type] = []
            
            platform_summary[platform][metric_type].append(value)
        
        # 성장률 계산
        for platform, metrics in platform_summary.items():
            for metric_type, values in metrics.items():
                if len(values) >= 2:
                    growth_rate = ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
                    platform_summary[platform][f"{metric_type}_growth"] = growth_rate
        
        return {
            "total_records": len(platform_metrics),
            "platform_summary": platform_summary
        }

    def _generate_artist_recommendations(self, events: List[Dict[str, Any]],
                                       impact_data: List[Dict[str, Any]],
                                       platform_metrics: List[Dict[str, Any]]) -> List[str]:
        """아티스트 맞춤 추천사항 생성"""
        recommendations = []
        
        # 이벤트 빈도 기반 추천
        if len(events) < 3:
            recommendations.append("더 많은 이벤트 활동을 통해 팬 참여도를 높이세요.")
        
        # 영향도 기반 추천
        if impact_data:
            avg_impact = sum(d.get("impact_percentage", 0) for d in impact_data) / len(impact_data)
            if avg_impact < 10:
                recommendations.append("이벤트 프로모션 전략을 강화하여 더 큰 영향을 만들어보세요.")
            elif avg_impact > 30:
                recommendations.append("훌륭한 성과입니다! 이 전략을 다른 이벤트에도 적용해보세요.")
        
        # 플랫폼 기반 추천
        if platform_metrics:
            # 가장 성과가 좋은 플랫폼 찾기
            platform_performance = {}
            for metric in platform_metrics:
                platform = metric.get("platform", "Unknown")
                if platform not in platform_performance:
                    platform_performance[platform] = 0
                platform_performance[platform] += metric.get("value", 0)
            
            if platform_performance:
                best_platform = max(platform_performance, key=platform_performance.get)
                recommendations.append(f"{best_platform}에서 가장 좋은 성과를 보이고 있습니다. 이 플랫폼에 더 집중해보세요.")
        
        return recommendations

    def _analyze_category_distribution(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """카테고리별 이벤트 분포 분석"""
        distribution = {}
        for event in events:
            category = event.get("category_name", "Unknown")
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def _analyze_artist_tier_distribution(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """아티스트 등급별 분포 분석"""
        # 실제 구현에서는 아티스트 등급 정보를 데이터베이스에서 조회
        # 여기서는 샘플 구현
        return {
            "A급 (Top Tier)": len([e for e in events if e.get("importance_level", 0) >= 4]),
            "B급 (Mid Tier)": len([e for e in events if e.get("importance_level", 0) == 3]),
            "C급 (Rising)": len([e for e in events if e.get("importance_level", 0) <= 2])
        }

    def _calculate_market_impact(self, events: List[Dict[str, Any]], target_date: date) -> Dict[str, Any]:
        """시장 전체 임팩트 계산"""
        total_impact = sum(e.get("global_impact_score", 0) for e in events)
        avg_impact = total_impact / len(events) if events else 0
        
        return {
            "total_impact_score": total_impact,
            "average_impact_score": avg_impact,
            "high_impact_events": len([e for e in events if e.get("global_impact_score", 0) > 5.0])
        }

    def _generate_competition_recommendations(self, competition_analysis: Dict[str, Any],
                                            category_distribution: Dict[str, int]) -> List[str]:
        """경쟁 상황 기반 추천사항 생성"""
        recommendations = []
        
        competition_level = competition_analysis.get("competition_level", 0)
        
        if competition_level >= 4:
            recommendations.append("⚠️ 높은 경쟁 상황입니다. 차별화된 컨텐츠나 특별한 프로모션을 고려하세요.")
        elif competition_level <= 2:
            recommendations.append("✅ 상대적으로 경쟁이 적은 시기입니다. 적극적인 마케팅 기회를 활용하세요.")
        
        # 가장 많은 카테고리 기반 추천
        if category_distribution:
            dominant_category = max(category_distribution, key=category_distribution.get)
            recommendations.append(f"📊 {dominant_category} 이벤트가 많습니다. 다른 카테고리로 차별화를 고려해보세요.")
        
        return recommendations

    def _find_similar_events(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """유사한 이벤트 찾기"""
        # 실제 구현에서는 ML 기법을 사용한 유사도 분석
        # 여기서는 카테고리와 중요도 기반 단순 매칭
        category = event_data.get("category")
        importance = event_data.get("importance_level", 3)
        
        # 최근 1년간 유사 이벤트 조회
        start_date = date.today() - timedelta(days=365)
        similar_events = self.event_db_manager.get_kpop_events(
            start_date=start_date,
            category=category
        )
        
        # 중요도가 비슷한 이벤트 필터링
        filtered_events = [
            e for e in similar_events 
            if abs(e.get("importance_level", 3) - importance) <= 1
        ]
        
        return filtered_events[:10]  # 최대 10개 반환

    def _analyze_artist_historical_performance(self, artist_id: int) -> Dict[str, Any]:
        """아티스트 과거 성과 분석"""
        # 최근 1년간 성과 조회
        start_date = date.today() - timedelta(days=365)
        
        artist_impacts = self.event_db_manager.get_artist_event_impacts(
            artist_id, start_date
        )
        
        if not artist_impacts:
            return {"avg_impact": 0, "event_count": 0, "performance_trend": "insufficient_data"}
        
        avg_impact = sum(i.get("impact_percentage", 0) for i in artist_impacts) / len(artist_impacts)
        event_count = len(set(i.get("event_id") for i in artist_impacts))
        
        # 성과 트렌드 분석 (최근 3개월 vs 이전 9개월)
        recent_date = date.today() - timedelta(days=90)
        recent_impacts = [i for i in artist_impacts if i.get("event_date", date.min) >= recent_date]
        older_impacts = [i for i in artist_impacts if i.get("event_date", date.min) < recent_date]
        
        trend = "stable"
        if recent_impacts and older_impacts:
            recent_avg = sum(i.get("impact_percentage", 0) for i in recent_impacts) / len(recent_impacts)
            older_avg = sum(i.get("impact_percentage", 0) for i in older_impacts) / len(older_impacts)
            
            if recent_avg > older_avg * 1.2:
                trend = "improving"
            elif recent_avg < older_avg * 0.8:
                trend = "declining"
        
        return {
            "avg_impact": avg_impact,
            "event_count": event_count,
            "performance_trend": trend,
            "total_measurements": len(artist_impacts)
        }

    def _run_prediction_model(self, event_data: Dict[str, Any], 
                            similar_events: List[Dict[str, Any]],
                            seasonal_analysis: Dict[str, Any],
                            artist_performance: Optional[Dict[str, Any]],
                            competition_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """예측 모델 실행"""
        # 간단한 점수 기반 예측 모델
        base_score = 50.0  # 기본 점수
        
        # 계절성 요인 (±15점)
        seasonal_factor = seasonal_analysis.get("intensity_score", 3.0)
        seasonal_adjustment = (seasonal_factor - 3.0) * 5.0
        
        # 아티스트 성과 요인 (±20점)
        artist_adjustment = 0.0
        if artist_performance:
            avg_impact = artist_performance.get("avg_impact", 0)
            trend = artist_performance.get("performance_trend", "stable")
            
            if avg_impact > 20:
                artist_adjustment += 15
            elif avg_impact > 10:
                artist_adjustment += 5
            elif avg_impact < 5:
                artist_adjustment -= 10
            
            if trend == "improving":
                artist_adjustment += 5
            elif trend == "declining":
                artist_adjustment -= 5
        
        # 경쟁 요인 (±10점)
        competition_level = competition_analysis.get("competition_level", {}).get("competition_level", 3)
        competition_adjustment = (3 - competition_level) * 2.5
        
        # 유사 이벤트 평균 성과 (±10점)
        similar_adjustment = 0.0
        if similar_events:
            # 실제 구현에서는 유사 이벤트의 실제 성과 데이터 사용
            similar_adjustment = 5.0  # 샘플 값
        
        # 최종 예측 점수 계산
        predicted_score = base_score + seasonal_adjustment + artist_adjustment + competition_adjustment + similar_adjustment
        predicted_score = max(0, min(100, predicted_score))  # 0-100 범위로 제한
        
        # 신뢰도 계산
        confidence = 0.7  # 기본 신뢰도
        if artist_performance and len(similar_events) > 5:
            confidence = 0.85
        elif not artist_performance or len(similar_events) < 2:
            confidence = 0.55
        
        return {
            "success_probability": predicted_score,
            "confidence": confidence,
            "factors": {
                "seasonal": seasonal_adjustment,
                "artist_performance": artist_adjustment,
                "competition": competition_adjustment,
                "similar_events": similar_adjustment
            },
            "risk_level": "low" if predicted_score > 70 else "medium" if predicted_score > 40 else "high"
        }

    def _generate_prediction_recommendations(self, prediction_result: Dict[str, Any]) -> List[str]:
        """예측 결과 기반 추천사항 생성"""
        recommendations = []
        
        success_prob = prediction_result.get("success_probability", 50)
        risk_level = prediction_result.get("risk_level", "medium")
        factors = prediction_result.get("factors", {})
        
        if success_prob > 80:
            recommendations.append("🌟 높은 성공 가능성! 적극적인 프로모션을 진행하세요.")
        elif success_prob < 30:
            recommendations.append("⚠️ 성공 가능성이 낮습니다. 전략 재검토를 권장합니다.")
        
        # 요인별 추천
        if factors.get("seasonal", 0) < -5:
            recommendations.append("📅 계절적 요인이 불리합니다. 다른 시기를 고려해보세요.")
        
        if factors.get("competition", 0) < -5:
            recommendations.append("🔥 경쟁이 치열합니다. 차별화 전략이 필요합니다.")
        
        if factors.get("artist_performance", 0) < -10:
            recommendations.append("📈 아티스트 최근 성과 개선이 필요합니다.")
        
        return recommendations

    # ==================== 유틸리티 메서드 ====================
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 시스템 상태 확인"""
        try:
            # 기존 시스템 상태
            main_db_status = self.db_manager.test_connection()
            
            # 이벤트 시스템 상태  
            event_db_status = True  # 실제 구현에서는 연결 테스트
            
            # 분석 시스템 상태
            analysis_systems = {
                "event_calendar": bool(self.event_calendar),
                "award_manager": bool(self.award_manager),
                "comeback_analyzer": bool(self.comeback_analyzer),
                "impact_analyzer": bool(self.impact_analyzer)
            }
            
            return {
                "status": "healthy" if all([main_db_status, event_db_status] + list(analysis_systems.values())) else "unhealthy",
                "main_database": "connected" if main_db_status else "disconnected",
                "event_database": "connected" if event_db_status else "disconnected",
                "analysis_systems": analysis_systems,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"통합 시스템 상태 확인 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }