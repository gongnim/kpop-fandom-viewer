import datetime
from config import Config
from collectors.base_collector import BaseCollector
from database_postgresql import add_platform_metric
from logger_config import logger

class TwitterCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.bearer_token = Config.TWITTER_BEARER_TOKEN
        if not self.bearer_token:
            raise ValueError("Twitter Bearer Token is not set.")
        self.base_url = "https://api.twitter.com/2/users/by/username/"

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def collect(self, account_id, username):
        """Twitter 유저네임을 사용하여 팔로워 수를 수집합니다."""
        headers = self._get_headers()
        params = {"user.fields": "public_metrics"}
        url = f"{self.base_url}{username}"
        
        data = self._safe_request(url, headers=headers, params=params)
        
        if data and 'data' in data:
            metrics = data['data']['public_metrics']
            followers = metrics.get('followers_count', 0)
            
            logger.info(f"[Twitter] Account {account_id}: Followers={followers}")
            add_platform_metric(account_id, 'twitter', 'followers', followers)
            return True
        else:
            logger.warning(f"[Twitter] Account {account_id}: Could not retrieve data for user {username}. Response: {data}")
            return False

