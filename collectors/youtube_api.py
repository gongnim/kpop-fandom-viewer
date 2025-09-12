import datetime
from config import Config
from collectors.base_collector import BaseCollector
from database_postgresql import get_api_usage, increment_api_usage, add_platform_metric
from logger_config import logger

YOUTUBE_QUOTA_LIMIT = 10000 # Daily quota limit for YouTube Data API

class YouTubeCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.api_key = Config.YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError("YouTube API key is not set in the environment variables.")
        self.base_url = "https://www.googleapis.com/youtube/v3/channels"

    def collect(self, account_id, channel_id):
        """YouTube 채널 ID를 사용하여 구독자 수와 총 조회수를 수집하고 DB에 저장합니다."""
        # Check YouTube API quota before making a request
        current_usage = get_api_usage('youtube')
        if current_usage >= YOUTUBE_QUOTA_LIMIT:
            logger.warning(f"[YouTube] Daily API quota limit ({YOUTUBE_QUOTA_LIMIT}) reached. Skipping collection for account {account_id}.")
            return False

        params = {
            'part': 'statistics',
            'id': channel_id,
            'key': self.api_key
        }
        
        data = self._safe_request(self.base_url, params=params)
        
        if data and 'items' in data and len(data['items']) > 0:
            increment_api_usage('youtube') 

            statistics = data['items'][0]['statistics']
            subscribers = int(statistics.get('subscriberCount', 0))
            total_views = int(statistics.get('viewCount', 0))
            
            logger.info(f"[YouTube] Account {account_id}: Subscribers={subscribers}, Views={total_views}")
            
            add_platform_metric(account_id, 'youtube', 'subscribers', subscribers)
            add_platform_metric(account_id, 'youtube', 'total_views', total_views)
            return True
        else:
            logger.warning(f"[YouTube] Account {account_id}: Could not retrieve data for channel {channel_id}. Response: {data}")
            return False

