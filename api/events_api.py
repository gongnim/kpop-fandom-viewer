"""
K-Pop Event System API Endpoints
이벤트 관리 시스템을 위한 RESTful API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import date, datetime
from pydantic import BaseModel, Field
import logging

from ..database_events import EventDatabaseManager
from ..analytics.kpop_event_calendar import KPopEvent, EventCategory, EventImportance
from ..analytics.award_shows_data import AwardShowInfo
from ..analytics.comeback_season_analyzer import ComebackSeasonAnalyzer
from ..analytics.event_impact_analyzer import EventImpactAnalyzer

logger = logging.getLogger(__name__)

# API 라우터 생성
router = APIRouter(prefix="/api/events", tags=["events"])

# ==================== Pydantic 모델 정의 ====================

class EventCreateRequest(BaseModel):
    """이벤트 생성 요청 모델"""
    name: str = Field(..., description="이벤트 이름")
    event_type: str = Field(..., description="이벤트 타입")
    category: str = Field(..., description="이벤트 카테고리")
    date: date = Field(..., description="이벤트 날짜")
    end_date: Optional[date] = Field(None, description="종료 날짜")
    venue: Optional[str] = Field(None, description="개최 장소")
    description: Optional[str] = Field(None, description="이벤트 설명")
    importance_level: int = Field(3, ge=1, le=5, description="중요도 (1-5)")
    global_impact_score: float = Field(0.0, ge=0.0, description="글로벌 영향도 점수")
    is_annual: bool = Field(False, description="연례 이벤트 여부")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")

class ComebackEventRequest(BaseModel):
    """컴백 이벤트 생성 요청 모델"""
    event_id: int = Field(..., description="이벤트 ID")
    artist_id: Optional[int] = Field(None, description="아티스트 ID")
    group_id: Optional[int] = Field(None, description="그룹 ID")
    album_id: Optional[int] = Field(None, description="앨범 ID")
    comeback_type: str = Field("single", description="컴백 타입")
    promotion_period: int = Field(30, ge=1, description="프로모션 기간 (일)")
    competition_level: int = Field(3, ge=1, le=5, description="경쟁 수준 (1-5)")
    expected_impact_score: float = Field(0.0, ge=0.0, description="예상 영향도 점수")

class ImpactMeasurementRequest(BaseModel):
    """영향도 측정 기록 요청 모델"""
    event_id: int = Field(..., description="이벤트 ID")
    artist_id: int = Field(..., description="아티스트 ID")
    metric_type: str = Field(..., description="측정 지표 타입")
    platform: str = Field(..., description="플랫폼")
    before_value: int = Field(..., ge=0, description="이벤트 전 값")
    after_value: int = Field(..., ge=0, description="이벤트 후 값")
    measurement_period: int = Field(7, ge=1, description="측정 기간 (일)")
    statistical_significance: float = Field(0.0, ge=0.0, le=1.0, description="통계적 유의성")
    confidence_level: float = Field(0.95, ge=0.0, le=1.0, description="신뢰 수준")

class PredictionRequest(BaseModel):
    """예측 저장 요청 모델"""
    event_id: int = Field(..., description="이벤트 ID")
    prediction_type: str = Field(..., description="예측 타입")
    predicted_value: float = Field(..., description="예측 값")
    confidence_interval_lower: float = Field(..., description="신뢰구간 하한")
    confidence_interval_upper: float = Field(..., description="신뢰구간 상한")
    model_version: str = Field("1.0", description="모델 버전")

class EventResponse(BaseModel):
    """이벤트 응답 모델"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ==================== 의존성 주입 ====================

def get_event_db_manager() -> EventDatabaseManager:
    """이벤트 데이터베이스 매니저 의존성 주입"""
    # 실제 구현에서는 데이터베이스 연결 풀을 주입받아야 함
    # 여기서는 예시를 위한 placeholder
    pass

# ==================== 이벤트 기본 CRUD API ====================

@router.get("/categories", response_model=Dict[str, Any])
async def get_event_categories(db_manager: EventDatabaseManager = Depends(get_event_db_manager)):
    """모든 이벤트 카테고리 조회"""
    try:
        categories = db_manager.get_event_categories()
        return {
            "success": True,
            "message": "이벤트 카테고리 조회 성공",
            "data": {"categories": categories}
        }
    except Exception as e:
        logger.error(f"이벤트 카테고리 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="이벤트 카테고리 조회 중 오류가 발생했습니다.")

@router.post("/", response_model=EventResponse)
async def create_event(request: EventCreateRequest, 
                      db_manager: EventDatabaseManager = Depends(get_event_db_manager)):
    """새 이벤트 생성"""
    try:
        # Pydantic 모델을 KPopEvent 객체로 변환
        event = KPopEvent(
            name=request.name,
            event_type=request.event_type,
            category=EventCategory(request.category),
            date=request.date,
            end_date=request.end_date,
            venue=request.venue,
            description=request.description,
            importance=EventImportance(request.importance_level),
            global_impact_score=request.global_impact_score,
            is_annual=request.is_annual,
            metadata=request.metadata
        )
        
        event_id = db_manager.create_kpop_event(event)
        
        if event_id:
            return EventResponse(
                success=True,
                message="이벤트 생성 성공",
                data={"event_id": event_id}
            )
        else:
            raise HTTPException(status_code=400, detail="이벤트 생성에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"이벤트 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="이벤트 생성 중 오류가 발생했습니다.")

@router.get("/", response_model=Dict[str, Any])
async def get_events(
    start_date: Optional[date] = Query(None, description="시작 날짜"),
    end_date: Optional[date] = Query(None, description="종료 날짜"),
    event_type: Optional[str] = Query(None, description="이벤트 타입"),
    category: Optional[str] = Query(None, description="이벤트 카테고리"),
    limit: int = Query(100, ge=1, le=1000, description="결과 개수 제한"),
    offset: int = Query(0, ge=0, description="결과 시작 위치"),
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """이벤트 목록 조회"""
    try:
        events = db_manager.get_kpop_events(start_date, end_date, event_type, category)
        
        # 페이지네이션 적용
        paginated_events = events[offset:offset + limit]
        
        return {
            "success": True,
            "message": "이벤트 조회 성공",
            "data": {
                "events": paginated_events,
                "total_count": len(events),
                "returned_count": len(paginated_events),
                "offset": offset,
                "limit": limit
            }
        }
    except Exception as e:
        logger.error(f"이벤트 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="이벤트 조회 중 오류가 발생했습니다.")

@router.get("/upcoming", response_model=Dict[str, Any])
async def get_upcoming_events(
    days_ahead: int = Query(30, ge=1, le=365, description="앞으로 며칠간의 이벤트"),
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """다가오는 이벤트 조회"""
    try:
        events = db_manager.get_upcoming_events(days_ahead)
        
        return {
            "success": True,
            "message": "다가오는 이벤트 조회 성공",
            "data": {
                "events": events,
                "days_ahead": days_ahead,
                "count": len(events)
            }
        }
    except Exception as e:
        logger.error(f"다가오는 이벤트 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="다가오는 이벤트 조회 중 오류가 발생했습니다.")

@router.get("/artist/{artist_id}", response_model=Dict[str, Any])
async def get_artist_events(
    artist_id: int,
    start_date: Optional[date] = Query(None, description="시작 날짜"),
    end_date: Optional[date] = Query(None, description="종료 날짜"),
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """특정 아티스트의 이벤트 조회"""
    try:
        events = db_manager.get_events_by_artist(artist_id, start_date, end_date)
        
        return {
            "success": True,
            "message": "아티스트 이벤트 조회 성공",
            "data": {
                "artist_id": artist_id,
                "events": events,
                "count": len(events)
            }
        }
    except Exception as e:
        logger.error(f"아티스트 이벤트 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="아티스트 이벤트 조회 중 오류가 발생했습니다.")

# ==================== 컴백 이벤트 API ====================

@router.post("/comeback", response_model=EventResponse)
async def create_comeback_event(
    request: ComebackEventRequest,
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """컴백 이벤트 상세 정보 생성"""
    try:
        comeback_id = db_manager.create_comeback_event(
            event_id=request.event_id,
            artist_id=request.artist_id,
            group_id=request.group_id,
            album_id=request.album_id,
            comeback_type=request.comeback_type,
            promotion_period=request.promotion_period,
            competition_level=request.competition_level,
            expected_impact_score=request.expected_impact_score
        )
        
        if comeback_id:
            return EventResponse(
                success=True,
                message="컴백 이벤트 생성 성공",
                data={"comeback_id": comeback_id}
            )
        else:
            raise HTTPException(status_code=400, detail="컴백 이벤트 생성에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"컴백 이벤트 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="컴백 이벤트 생성 중 오류가 발생했습니다.")

@router.get("/comeback", response_model=Dict[str, Any])
async def get_comeback_events(
    start_date: Optional[date] = Query(None, description="시작 날짜"),
    end_date: Optional[date] = Query(None, description="종료 날짜"),
    artist_id: Optional[int] = Query(None, description="아티스트 ID"),
    group_id: Optional[int] = Query(None, description="그룹 ID"),
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """컴백 이벤트 조회"""
    try:
        comebacks = db_manager.get_comeback_events(start_date, end_date, artist_id, group_id)
        
        return {
            "success": True,
            "message": "컴백 이벤트 조회 성공",
            "data": {
                "comebacks": comebacks,
                "count": len(comebacks)
            }
        }
    except Exception as e:
        logger.error(f"컴백 이벤트 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="컴백 이벤트 조회 중 오류가 발생했습니다.")

# ==================== 이벤트 영향도 분석 API ====================

@router.post("/impact", response_model=EventResponse)
async def record_event_impact(
    request: ImpactMeasurementRequest,
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """이벤트 영향도 측정 결과 기록"""
    try:
        measurement_id = db_manager.record_event_impact(
            event_id=request.event_id,
            artist_id=request.artist_id,
            metric_type=request.metric_type,
            platform=request.platform,
            before_value=request.before_value,
            after_value=request.after_value,
            measurement_period=request.measurement_period,
            statistical_significance=request.statistical_significance,
            confidence_level=request.confidence_level
        )
        
        if measurement_id:
            return EventResponse(
                success=True,
                message="이벤트 영향도 측정 기록 성공",
                data={"measurement_id": measurement_id}
            )
        else:
            raise HTTPException(status_code=400, detail="이벤트 영향도 측정 기록에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"이벤트 영향도 측정 기록 실패: {e}")
        raise HTTPException(status_code=500, detail="이벤트 영향도 측정 기록 중 오류가 발생했습니다.")

@router.get("/impact/{event_id}", response_model=Dict[str, Any])
async def get_event_impact_analysis(
    event_id: int,
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """특정 이벤트의 영향도 분석 결과 조회"""
    try:
        impact_data = db_manager.get_event_impact_analysis(event_id)
        
        return {
            "success": True,
            "message": "이벤트 영향도 분석 조회 성공",
            "data": {
                "event_id": event_id,
                "impact_measurements": impact_data,
                "count": len(impact_data)
            }
        }
    except Exception as e:
        logger.error(f"이벤트 영향도 분석 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="이벤트 영향도 분석 조회 중 오류가 발생했습니다.")

@router.get("/impact/artist/{artist_id}", response_model=Dict[str, Any])
async def get_artist_event_impacts(
    artist_id: int,
    start_date: Optional[date] = Query(None, description="시작 날짜"),
    end_date: Optional[date] = Query(None, description="종료 날짜"),
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """특정 아티스트의 이벤트 영향도 기록 조회"""
    try:
        impact_data = db_manager.get_artist_event_impacts(artist_id, start_date, end_date)
        
        return {
            "success": True,
            "message": "아티스트 이벤트 영향도 조회 성공",
            "data": {
                "artist_id": artist_id,
                "impact_records": impact_data,
                "count": len(impact_data)
            }
        }
    except Exception as e:
        logger.error(f"아티스트 이벤트 영향도 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="아티스트 이벤트 영향도 조회 중 오류가 발생했습니다.")

# ==================== 계절별 패턴 분석 API ====================

@router.get("/patterns/seasonal", response_model=Dict[str, Any])
async def get_seasonal_patterns(
    season: Optional[str] = Query(None, description="계절 (spring, summer, autumn, winter)"),
    pattern_type: Optional[str] = Query(None, description="패턴 타입"),
    analysis_year: Optional[int] = Query(None, description="분석 연도"),
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """계절별 패턴 분석 결과 조회"""
    try:
        patterns = db_manager.get_seasonal_patterns(season, pattern_type, analysis_year)
        
        return {
            "success": True,
            "message": "계절별 패턴 조회 성공",
            "data": {
                "patterns": patterns,
                "filters": {
                    "season": season,
                    "pattern_type": pattern_type,
                    "analysis_year": analysis_year
                },
                "count": len(patterns)
            }
        }
    except Exception as e:
        logger.error(f"계절별 패턴 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="계절별 패턴 조회 중 오류가 발생했습니다.")

# ==================== 예측 모델 API ====================

@router.post("/predictions", response_model=EventResponse)
async def save_event_prediction(
    request: PredictionRequest,
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """이벤트 예측 결과 저장"""
    try:
        prediction_id = db_manager.save_event_prediction(
            event_id=request.event_id,
            prediction_type=request.prediction_type,
            predicted_value=request.predicted_value,
            confidence_interval_lower=request.confidence_interval_lower,
            confidence_interval_upper=request.confidence_interval_upper,
            model_version=request.model_version
        )
        
        if prediction_id:
            return EventResponse(
                success=True,
                message="이벤트 예측 저장 성공",
                data={"prediction_id": prediction_id}
            )
        else:
            raise HTTPException(status_code=400, detail="이벤트 예측 저장에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"이벤트 예측 저장 실패: {e}")
        raise HTTPException(status_code=500, detail="이벤트 예측 저장 중 오류가 발생했습니다.")

@router.put("/predictions/{prediction_id}/accuracy", response_model=EventResponse)
async def update_prediction_accuracy(
    prediction_id: int,
    actual_value: float,
    db_manager: EventDatabaseManager = Depends(get_event_db_manager)
):
    """예측 정확도 업데이트 (실제 값 발생 후)"""
    try:
        success = db_manager.update_prediction_accuracy(prediction_id, actual_value)
        
        if success:
            return EventResponse(
                success=True,
                message="예측 정확도 업데이트 성공",
                data={"prediction_id": prediction_id, "actual_value": actual_value}
            )
        else:
            raise HTTPException(status_code=400, detail="예측 정확도 업데이트에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"예측 정확도 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail="예측 정확도 업데이트 중 오류가 발생했습니다.")

# ==================== 통계 및 대시보드 API ====================

@router.get("/statistics", response_model=Dict[str, Any])
async def get_event_statistics(db_manager: EventDatabaseManager = Depends(get_event_db_manager)):
    """이벤트 통계 정보 조회"""
    try:
        stats = db_manager.get_event_statistics()
        
        return {
            "success": True,
            "message": "이벤트 통계 조회 성공",
            "data": stats
        }
    except Exception as e:
        logger.error(f"이벤트 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="이벤트 통계 조회 중 오류가 발생했습니다.")

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """API 상태 확인"""
    return {
        "status": "healthy",
        "service": "K-Pop Event System API",
        "timestamp": datetime.now().isoformat()
    }