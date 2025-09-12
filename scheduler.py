import time
from apscheduler.schedulers.background import BackgroundScheduler
from database_postgresql import get_all_artist_accounts
from collectors.youtube_api import YouTubeCollector
from collectors.spotify_api import SpotifyCollector
from collectors.twitter_api import TwitterCollector
from logger_config import logger

def collect_all_data():
    """등록된 모든 아티스트의 모든 플랫폼 데이터를 수집합니다."""
    logger.info("Starting daily data collection job")
    accounts = get_all_artist_accounts()
    if not accounts:
        logger.warning("No artist accounts found in the database. Skipping collection job.")
        return

    # 각 컬렉터 초기화
    try:
        yt_collector = YouTubeCollector()
        sp_collector = SpotifyCollector()
        tw_collector = TwitterCollector()
    except ValueError as e:
        logger.error(f"Collector initialization failed: {e}")
        return

    logger.info(f"Found {len(accounts)} accounts to collect.")
    for account in accounts:
        account_id = account['account_id']
        platform = account['platform']
        identifier = account['account_identifier']
        
        logger.info(f"Collecting for Account ID: {account_id}, Platform: {platform}, Identifier: {identifier}")
        try:
            if platform == 'youtube':
                yt_collector.collect(account_id, identifier)
            elif platform == 'spotify':
                sp_collector.collect(account_id, identifier)
            elif platform == 'twitter':
                tw_collector.collect(account_id, identifier)
        except Exception as e:
            logger.error(f"Error collecting for {artist_id} on {platform}: {e}", exc_info=True)
        
        # API Rate Limiting을 위해 약간의 딜레이 추가
        time.sleep(2)
    
    logger.info("Daily data collection job finished.")

def start_scheduler():
    """스케줄러를 시작하고 작업을 등록합니다."""
    scheduler = BackgroundScheduler(daemon=True)
    
    # 매일 오전 3시에 데이터 수집 실행
    scheduler.add_job(
        func=collect_all_data,
        trigger="cron",
        hour=3,
        minute=0,
        misfire_grace_time=3600 # 1시간 내에 실행 못한 작업은 재시도
    )
    
    try:
        scheduler.start()
        logger.info("Scheduler started. Daily job scheduled for 03:00.")
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")
    
    return scheduler
