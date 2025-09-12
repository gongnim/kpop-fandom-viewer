import datetime
import base64
from config import Config
from collectors.base_collector import BaseCollector
from database_postgresql import add_platform_metric
from logger_config import logger

class SpotifyCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.client_id = Config.SPOTIFY_CLIENT_ID
        self.client_secret = Config.SPOTIFY_CLIENT_SECRET
        self.access_token = self._get_access_token()
        self.base_url = "https://api.spotify.com/v1/artists/"

    def _get_access_token(self):
        """Spotify API 접근을 위한 Access Token을 발급받습니다."""
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify Client ID/Secret is not set.")
        
        auth_str = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_str.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        response = self.session.post(
            "https://accounts.spotify.com/api/token",
            headers={'Authorization': f'Basic {auth_base64}'},
            data={'grant_type': 'client_credentials'}
        )
        
        if response.status_code == 200:
            logger.info("Successfully obtained Spotify access token.")
            return response.json()['access_token']
        else:
            logger.error(f"Failed to get Spotify access token. Status: {response.status_code}, Response: {response.text}")
            raise Exception("Failed to get Spotify access token.")

    def collect(self, account_id, spotify_artist_id):
        """Spotify 아티스트 ID를 사용하여 팔로워 수와 인기도를 수집합니다."""
        if not self.access_token:
            logger.warning("[Spotify] Access Token is not available. Skipping collection.")
            return False

        headers = {'Authorization': f'Bearer {self.access_token}'}
        url = f"{self.base_url}{spotify_artist_id}"
        
        data = self._safe_request(url, headers=headers)
        
        if data and 'followers' in data:
            followers = data['followers']['total']
            popularity = data.get('popularity', 0) # 인기도는 0-100 사이의 값

            logger.info(f"[Spotify] Account {account_id}: Followers={followers}, Popularity={popularity}")
            
            add_platform_metric(account_id, 'spotify', 'followers', followers)
            add_platform_metric(account_id, 'spotify', 'popularity', popularity)
            return True
        else:
            logger.warning(f"[Spotify] Account {account_id}: Could not retrieve data for artist {spotify_artist_id}. Response: {data}")
            return False


    
