"""
K-Pop 이벤트 시스템 FastAPI 서버
이벤트 관리 API를 실행하는 메인 서버 애플리케이션
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

# 로컬 모듈 import
from .api.events_api import router as events_router
from .database_postgresql import DatabaseManager
from .database_events import EventDatabaseManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
db_manager = None
event_db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global db_manager, event_db_manager
    
    # 시작 시 초기화
    logger.info("🚀 K-Pop 이벤트 API 서버 시작")
    
    try:
        # 데이터베이스 연결 초기화
        db_manager = DatabaseManager()
        # event_db_manager = EventDatabaseManager(db_manager.pool)
        
        logger.info("✅ 데이터베이스 연결 초기화 완료")
        
    except Exception as e:
        logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
        raise
    
    yield  # 서버 실행
    
    # 종료 시 정리
    logger.info("🛑 K-Pop 이벤트 API 서버 종료")
    if db_manager:
        # 연결 정리 (실제 구현에서는 connection pool 정리)
        pass

# FastAPI 앱 생성
app = FastAPI(
    title="K-Pop 이벤트 관리 API",
    description="K-Pop 이벤트 달력, 시상식, 컴백 분석을 위한 RESTful API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(events_router)

# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """API 서버 상태 확인"""
    return {
        "status": "healthy",
        "service": "K-Pop Event Management API",
        "version": "1.0.0",
        "timestamp": "2024-09-09T10:45:00Z"
    }

# 메인 정보 엔드포인트
@app.get("/")
async def root():
    """API 메인 정보"""
    return {
        "message": "K-Pop 이벤트 관리 API에 오신 것을 환영합니다!",
        "documentation": "/docs",
        "health_check": "/health",
        "api_endpoints": "/api/events"
    }

# 의존성 주입을 위한 함수들
def get_db_manager():
    """데이터베이스 매니저 의존성 주입"""
    if db_manager is None:
        raise HTTPException(status_code=500, detail="데이터베이스 연결이 초기화되지 않았습니다.")
    return db_manager

def get_event_db_manager():
    """이벤트 데이터베이스 매니저 의존성 주입"""
    if event_db_manager is None:
        raise HTTPException(status_code=500, detail="이벤트 데이터베이스 연결이 초기화되지 않았습니다.")
    return event_db_manager

# 개발 서버 실행 함수
def run_development_server():
    """개발 서버 실행"""
    uvicorn.run(
        "kpop_dashboard.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_development_server()