"""
K-Pop 이벤트 시스템 API 클라이언트
Streamlit 앱에서 백엔드 API를 호출하기 위한 클라이언트
"""

import requests
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import date, datetime
import logging
import json

logger = logging.getLogger(__name__)

class EventAPIClient:
    """이벤트 API 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        API 클라이언트 초기화
        
        Args:
            base_url: API 서버 기본 URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """API 응답 처리"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {e}")
            raise Exception(f"API 요청 실패: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            raise Exception(f"응답 파싱 실패: {str(e)}")
    
    # ==================== 이벤트 기본 CRUD ====================
    
    def get_events(self, start_date: Optional[date] = None, 
                   end_date: Optional[date] = None,
                   event_type: Optional[str] = None,
                   category: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """이벤트 목록 조회"""
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if event_type:
            params["event_type"] = event_type
        if category:
            params["category"] = category
        
        response = self.session.get(f"{self.base_url}/api/events/", params=params)
        return self._handle_response(response)
    
    def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """새 이벤트 생성"""
        response = self.session.post(f"{self.base_url}/api/events/", json=event_data)
        return self._handle_response(response)
    
    def get_upcoming_events(self, days_ahead: int = 30) -> Dict[str, Any]:
        """다가오는 이벤트 조회"""
        params = {"days_ahead": days_ahead}
        response = self.session.get(f"{self.base_url}/api/events/upcoming", params=params)
        return self._handle_response(response)
    
    def get_artist_events(self, artist_id: int, 
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> Dict[str, Any]:
        """특정 아티스트의 이벤트 조회"""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        response = self.session.get(f"{self.base_url}/api/events/artist/{artist_id}", params=params)
        return self._handle_response(response)
    
    # ==================== 컴백 이벤트 ====================
    
    def create_comeback_event(self, comeback_data: Dict[str, Any]) -> Dict[str, Any]:
        """컴백 이벤트 생성"""
        response = self.session.post(f"{self.base_url}/api/events/comeback", json=comeback_data)
        return self._handle_response(response)
    
    def get_comeback_events(self, start_date: Optional[date] = None,
                           end_date: Optional[date] = None,
                           artist_id: Optional[int] = None,
                           group_id: Optional[int] = None) -> Dict[str, Any]:
        """컴백 이벤트 조회"""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if artist_id:
            params["artist_id"] = artist_id
        if group_id:
            params["group_id"] = group_id
        
        response = self.session.get(f"{self.base_url}/api/events/comeback", params=params)
        return self._handle_response(response)
    
    # ==================== 영향도 분석 ====================
    
    def record_event_impact(self, impact_data: Dict[str, Any]) -> Dict[str, Any]:
        """이벤트 영향도 기록"""
        response = self.session.post(f"{self.base_url}/api/events/impact", json=impact_data)
        return self._handle_response(response)
    
    def get_event_impact_analysis(self, event_id: int) -> Dict[str, Any]:
        """특정 이벤트의 영향도 분석 조회"""
        response = self.session.get(f"{self.base_url}/api/events/impact/{event_id}")
        return self._handle_response(response)
    
    def get_artist_event_impacts(self, artist_id: int,
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None) -> Dict[str, Any]:
        """특정 아티스트의 이벤트 영향도 기록 조회"""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        response = self.session.get(f"{self.base_url}/api/events/impact/artist/{artist_id}", params=params)
        return self._handle_response(response)
    
    # ==================== 통계 및 분석 ====================
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """이벤트 통계 정보 조회"""
        response = self.session.get(f"{self.base_url}/api/events/statistics")
        return self._handle_response(response)
    
    def get_seasonal_patterns(self, season: Optional[str] = None,
                             pattern_type: Optional[str] = None,
                             analysis_year: Optional[int] = None) -> Dict[str, Any]:
        """계절별 패턴 조회"""
        params = {}
        if season:
            params["season"] = season
        if pattern_type:
            params["pattern_type"] = pattern_type
        if analysis_year:
            params["analysis_year"] = analysis_year
        
        response = self.session.get(f"{self.base_url}/api/events/patterns/seasonal", params=params)
        return self._handle_response(response)
    
    # ==================== 예측 모델 ====================
    
    def save_event_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """이벤트 예측 결과 저장"""
        response = self.session.post(f"{self.base_url}/api/events/predictions", json=prediction_data)
        return self._handle_response(response)
    
    def update_prediction_accuracy(self, prediction_id: int, actual_value: float) -> Dict[str, Any]:
        """예측 정확도 업데이트"""
        data = {"actual_value": actual_value}
        response = self.session.put(f"{self.base_url}/api/events/predictions/{prediction_id}/accuracy", json=data)
        return self._handle_response(response)
    
    # ==================== 헬스 체크 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """API 서버 상태 확인"""
        response = self.session.get(f"{self.base_url}/health")
        return self._handle_response(response)
    
    def api_info(self) -> Dict[str, Any]:
        """API 정보 조회"""
        response = self.session.get(f"{self.base_url}/")
        return self._handle_response(response)

# Streamlit 캐시를 사용한 API 클라이언트 싱글톤
@st.cache_resource
def get_api_client(base_url: str = "http://localhost:8000") -> EventAPIClient:
    """API 클라이언트 싱글톤 (Streamlit 캐시 사용)"""
    return EventAPIClient(base_url)

# API 호출 헬퍼 함수들
def safe_api_call(func, *args, **kwargs) -> Optional[Dict[str, Any]]:
    """안전한 API 호출 (에러 핸들링 포함)"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"API 호출 실패: {str(e)}")
        logger.error(f"API 호출 실패: {e}")
        return None

def display_api_error(error_message: str):
    """API 에러 메시지 표시"""
    st.error(f"⚠️ {error_message}")
    st.info("💡 API 서버가 실행 중인지 확인하세요. (http://localhost:8000)")

# API 서버 연결 상태 확인
def check_api_connection(client: EventAPIClient) -> bool:
    """API 서버 연결 상태 확인"""
    try:
        response = client.health_check()
        return response.get("status") == "healthy"
    except:
        return False