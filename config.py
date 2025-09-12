import os
import logging
from typing import Any
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# .env 파일의 절대 경로를 찾아서 로드 (로컬 개발 환경용)
# 이 스크립트 파일의 위치를 기준으로 .env 파일을 찾습니다.
try:
    from dotenv import load_dotenv
    # config.py 파일의 현재 디렉토리를 기준으로 .env 파일 경로를 설정
    env_path = Path(os.path.dirname(__file__)) / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.info(f".env file not found at {env_path}. Relying on Streamlit secrets or system environment variables.")
except ImportError:
    logger.info("dotenv not installed, skipping .env file load. Relying on Streamlit secrets or system environment variables.")

def _get_secret(key: str, default: Any = None, section: str = None) -> Any:
    """
    Streamlit secrets를 먼저 확인하고, 없으면 환경 변수에서 값을 가져옵니다.
    Streamlit 앱과 백그라운드 스크립트 양쪽에서 모두 동작합니다.
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            # 섹션별로 접근 시도
            if section and hasattr(st.secrets, section.lower()):
                section_secrets = getattr(st.secrets, section.lower())
                if key.lower() in section_secrets:
                    return section_secrets[key.lower()]
            
            # 루트 레벨에서 직접 접근
            if key.lower() in st.secrets:
                return st.secrets[key.lower()]
            if key.upper() in st.secrets:
                return st.secrets[key.upper()]
    except (ImportError, Exception) as e:
        logger.debug(f"Failed to access Streamlit secrets: {e}")
        # Streamlit 컨텍스트가 아닐 경우 (e.g., scheduler)
        pass
    
    # 환경 변수에서 가져오기 (키는 대문자로 가정)
    return os.getenv(key.upper(), default)

class Config:
    """
    애플리케이션 설정을 관리하는 클래스.
    Streamlit secrets과 환경 변수를 모두 지원하여 유연성을 높입니다.
    """
    # API Keys (Platform 섹션에서 우선 검색, 없으면 루트에서 검색)
    YOUTUBE_API_KEY = _get_secret('YOUTUBE_API_KEY', section='Platform') or _get_secret('YOUTUBE_API_KEY')
    SPOTIFY_CLIENT_ID = _get_secret('SPOTIFY_CLIENT_ID', section='Platform') or _get_secret('SPOTIFY_CLIENT_ID')
    SPOTIFY_CLIENT_SECRET = _get_secret('SPOTIFY_CLIENT_SECRET', section='Platform') or _get_secret('SPOTIFY_CLIENT_SECRET')
    TWITTER_BEARER_TOKEN = _get_secret('TWITTER_BEARER_TOKEN', section='Platform') or _get_secret('TWITTER_BEARER_TOKEN')

    # Database Configuration
    DB_CONFIG = {
        'host': _get_secret('POSTGRES_HOST', 'localhost', section='PostgreSQL'),
        'port': int(_get_secret('POSTGRES_PORT', 5432, section='PostgreSQL')),
        'database': _get_secret('POSTGRES_DB', 'kpop_dashboard_pg', section='PostgreSQL'),
        'user': _get_secret('POSTGRES_USER', 'kpop_dashboard_user', section='PostgreSQL'),
        'password': _get_secret('POSTGRES_PASSWORD', '', section='PostgreSQL'),
        'sslmode': _get_secret('POSTGRES_SSLMODE', 'prefer', section='PostgreSQL')
    }
    
    @classmethod
    def debug_config(cls):
        """디버그용: 현재 설정값들을 출력 (비밀번호 제외)"""
        logger.info(f"DB Host: {cls.DB_CONFIG['host']}")
        logger.info(f"DB Port: {cls.DB_CONFIG['port']}")
        logger.info(f"DB Database: {cls.DB_CONFIG['database']}")
        logger.info(f"DB User: {cls.DB_CONFIG['user']}")
        logger.info(f"DB Password exists: {bool(cls.DB_CONFIG['password'])}")
        logger.info(f"DB SSL Mode: {cls.DB_CONFIG['sslmode']}")
        return cls.DB_CONFIG

    @classmethod
    def validate_keys(cls):
        """필수 API 키들이 설정되었는지 확인"""
        if not cls.SPOTIFY_CLIENT_ID or not cls.SPOTIFY_CLIENT_SECRET:
            raise ValueError("Missing Spotify API keys: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        
        if not cls.YOUTUBE_API_KEY:
            raise ValueError("Missing API key: YOUTUBE_API_KEY")
            
        if not cls.TWITTER_BEARER_TOKEN:
            raise ValueError("Missing API key: TWITTER_BEARER_TOKEN")
