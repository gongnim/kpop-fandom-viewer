import datetime
import re
from config import Config
from collectors.base_collector import BaseCollector
from database_postgresql import get_api_usage, increment_api_usage, add_platform_metric
from logger_config import logger

YOUTUBE_QUOTA_LIMIT = 10000  # Daily quota limit for YouTube Data API

class YouTubeCollectorEnhanced(BaseCollector):
    def __init__(self):
        super().__init__()
        self.api_key = Config.YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError("YouTube API key is not set in the environment variables.")
        self.channels_url = "https://www.googleapis.com/youtube/v3/channels"
        self.search_url = "https://www.googleapis.com/youtube/v3/search"

    def _detect_identifier_type(self, identifier):
        """
        Detect whether the identifier is a channel ID, handle, or username.
        
        Returns:
            tuple: (type, cleaned_identifier) where type is 'channel_id', 'handle', or 'username'
        """
        # Remove common URL prefixes
        identifier = identifier.strip()
        
        # Remove URL parts if present
        if identifier.startswith('https://www.youtube.com/'):
            if '/channel/' in identifier:
                identifier = identifier.split('/channel/')[-1].split('?')[0]
            elif '/@' in identifier:
                identifier = identifier.split('/@')[-1].split('?')[0]
            elif '/user/' in identifier:
                identifier = identifier.split('/user/')[-1].split('?')[0]
            elif '/c/' in identifier:
                identifier = identifier.split('/c/')[-1].split('?')[0]
        
        # Remove @ prefix if present
        if identifier.startswith('@'):
            identifier = identifier[1:]
        
        # Detect type based on pattern
        if re.match(r'^UC[a-zA-Z0-9_-]{22}$', identifier):
            return ('channel_id', identifier)
        elif len(identifier) < 30 and re.match(r'^[a-zA-Z0-9_.-]+$', identifier):
            # Could be handle or username - we'll try handle first as it's more common now
            return ('handle', identifier)
        else:
            # Default to username for other patterns
            return ('username', identifier)

    def _get_channel_by_id(self, channel_id):
        """Get channel statistics by channel ID."""
        params = {
            'part': 'statistics',
            'id': channel_id,
            'key': self.api_key
        }
        return self._safe_request(self.channels_url, params=params)

    def _get_channel_by_handle(self, handle):
        """Get channel statistics by handle (modern @username format)."""
        # Try to get channel ID by handle using search
        search_params = {
            'part': 'snippet',
            'q': f'@{handle}',
            'type': 'channel',
            'maxResults': 1,
            'key': self.api_key
        }
        
        search_data = self._safe_request(self.search_url, params=search_params)
        
        if search_data and 'items' in search_data and len(search_data['items']) > 0:
            channel_id = search_data['items'][0]['snippet']['channelId']
            logger.info(f"[YouTube] Found channel ID {channel_id} for handle @{handle}")
            
            # Now get the statistics using channel ID
            return self._get_channel_by_id(channel_id), channel_id
        
        return None, None

    def _get_channel_by_username(self, username):
        """Get channel statistics by legacy username."""
        params = {
            'part': 'statistics',
            'forUsername': username,
            'key': self.api_key
        }
        return self._safe_request(self.channels_url, params=params)

    def collect(self, account_id, identifier):
        """
        Enhanced YouTube data collection supporting channel IDs, handles, and usernames.
        
        Args:
            account_id: Database account ID
            identifier: YouTube channel identifier (ID, handle, or username)
            
        Returns:
            bool: Success status
        """
        # Check YouTube API quota before making requests
        current_usage = get_api_usage('youtube')
        if current_usage >= YOUTUBE_QUOTA_LIMIT:
            logger.warning(f"[YouTube] Daily API quota limit ({YOUTUBE_QUOTA_LIMIT}) reached. Skipping collection for account {account_id}.")
            return False

        # Detect identifier type and clean it
        identifier_type, cleaned_identifier = self._detect_identifier_type(identifier)
        
        logger.info(f"[YouTube] Account {account_id}: Detected {identifier_type} - '{cleaned_identifier}' from '{identifier}'")

        data = None
        discovered_channel_id = None
        
        try:
            if identifier_type == 'channel_id':
                data = self._get_channel_by_id(cleaned_identifier)
                increment_api_usage('youtube')
                
            elif identifier_type == 'handle':
                data, discovered_channel_id = self._get_channel_by_handle(cleaned_identifier)
                increment_api_usage('youtube', 2)  # Search + channels call
                
            elif identifier_type == 'username':
                data = self._get_channel_by_username(cleaned_identifier)
                increment_api_usage('youtube')
            
            if data and 'items' in data and len(data['items']) > 0:
                statistics = data['items'][0]['statistics']
                subscribers = int(statistics.get('subscriberCount', 0))
                total_views = int(statistics.get('viewCount', 0))
                
                logger.info(f"[YouTube] Account {account_id} ({identifier_type}): Subscribers={subscribers:,}, Views={total_views:,}")
                
                # Store metrics
                add_platform_metric(account_id, 'youtube', 'subscribers', subscribers)
                add_platform_metric(account_id, 'youtube', 'total_views', total_views)
                
                # If we discovered a channel ID from handle, log it for potential database update
                if discovered_channel_id and discovered_channel_id != identifier:
                    logger.info(f"[YouTube] Account {account_id}: Discovered channel ID '{discovered_channel_id}' for handle '{identifier}'. Consider updating database for better performance.")
                
                return True
            else:
                logger.warning(f"[YouTube] Account {account_id}: No data found for {identifier_type} '{cleaned_identifier}'. Response: {data}")
                
                # If handle failed, try as username fallback
                if identifier_type == 'handle':
                    logger.info(f"[YouTube] Account {account_id}: Trying '{cleaned_identifier}' as username fallback")
                    fallback_data = self._get_channel_by_username(cleaned_identifier)
                    increment_api_usage('youtube')
                    
                    if fallback_data and 'items' in fallback_data and len(fallback_data['items']) > 0:
                        statistics = fallback_data['items'][0]['statistics']
                        subscribers = int(statistics.get('subscriberCount', 0))
                        total_views = int(statistics.get('viewCount', 0))
                        
                        logger.info(f"[YouTube] Account {account_id} (username fallback): Subscribers={subscribers:,}, Views={total_views:,}")
                        
                        add_platform_metric(account_id, 'youtube', 'subscribers', subscribers)
                        add_platform_metric(account_id, 'youtube', 'total_views', total_views)
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"[YouTube] Account {account_id}: Error collecting data for '{identifier}': {str(e)}")
            return False

    def test_identifier(self, identifier):
        """
        Test method to check what type of identifier this is and if it works.
        For debugging purposes only.
        """
        identifier_type, cleaned_identifier = self._detect_identifier_type(identifier)
        print(f"Identifier: '{identifier}' -> Type: {identifier_type}, Cleaned: '{cleaned_identifier}'")
        return identifier_type, cleaned_identifier


# Legacy class for backward compatibility
class YouTubeCollector(YouTubeCollectorEnhanced):
    """Backward compatible wrapper for the enhanced collector."""
    
    def collect(self, account_id, channel_id):
        """Legacy interface - delegates to enhanced collector."""
        return super().collect(account_id, channel_id)