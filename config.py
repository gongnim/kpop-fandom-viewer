import streamlit as st
import logging
import os
from typing import Any

# Configure logger
logger = logging.getLogger(__name__)

# This new version directly and exclusively uses Streamlit secrets for deployment clarity.
# It avoids fallbacks to .env or os.getenv to prevent environment confusion on Streamlit Cloud.

DB_CONFIG = {}
YOUTUBE_API_KEY = None
SPOTIFY_CLIENT_ID = None
SPOTIFY_CLIENT_SECRET = None
TWITTER_BEARER_TOKEN = None

try:
    # Directly access secrets for PostgreSQL
    pg_secrets = st.secrets['PostgreSQL']
    DB_CONFIG = {
        'host': pg_secrets['POSTGRES_HOST'],
        'port': int(pg_secrets['POSTGRES_PORT']),
        'database': pg_secrets['POSTGRES_DB'],
        'user': pg_secrets['POSTGRES_USER'],
        'password': pg_secrets['POSTGRES_PASSWORD'],
        'sslmode': pg_secrets.get('POSTGRES_SSLMODE', 'prefer')
    }

    # Directly access secrets for Platform APIs
    platform_secrets = st.secrets['Platform']
    YOUTUBE_API_KEY = platform_secrets['YOUTUBE_API_KEY']
    SPOTIFY_CLIENT_ID = platform_secrets['SPOTIFY_CLIENT_ID']
    SPOTIFY_CLIENT_SECRET = platform_secrets['SPOTIFY_CLIENT_SECRET']
    TWITTER_BEARER_TOKEN = platform_secrets['TWITTER_BEARER_TOKEN']

except (AttributeError, KeyError) as e:
    logger.error(f"!!! CRITICAL: Failed to read secrets from Streamlit. Error: {e} !!!")
    logger.error("Please ensure secrets are correctly configured in Streamlit Cloud under [postgresql] and [platform] sections.")
    # Provide fallback for local execution if needed, but this will fail on deployment if secrets are missing.
    DB_CONFIG = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'kpop_dashboard_pg'),
        'user': os.getenv('POSTGRES_USER', 'kpop_dashboard_user'),
        'password': os.getenv('POSTGRES_PASSWORD', ''),
        'sslmode': os.getenv('POSTGRES_SSLMODE', 'prefer')
    }


class Config:
    """
    애플리케이션 설정을 관리하는 클래스.
    Streamlit secrets과 환경 변수를 모두 지원하여 유연성을 높입니다.
    """
    YOUTUBE_API_KEY = YOUTUBE_API_KEY
    SPOTIFY_CLIENT_ID = SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET = SPOTIFY_CLIENT_SECRET
    TWITTER_BEARER_TOKEN = TWITTER_BEARER_TOKEN
    DB_CONFIG = DB_CONFIG
    
    @classmethod
    def debug_config(cls):
        """디버그용: 현재 설정값들을 출력 (비밀번호 제외)"""
        logger.info(f"DB Host: {cls.DB_CONFIG.get('host')}")
        logger.info(f"DB Port: {cls.DB_CONFIG.get('port')}")
        logger.info(f"DB Database: {cls.DB_CONFIG.get('database')}")
        logger.info(f"DB User: {cls.DB_CONFIG.get('user')}")
        logger.info(f"DB Password exists: {bool(cls.DB_CONFIG.get('password'))}")
        logger.info(f"DB SSL Mode: {cls.DB_CONFIG.get('sslmode')}")
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