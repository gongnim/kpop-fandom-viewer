from abc import ABC, abstractmethod
import requests
import time
from config import Config

class BaseCollector(ABC):
    def __init__(self):
        self.session = requests.Session()
        # API마다 다른 rate limit을 적용하기 위해 자식 클래스에서 오버라이드 할 수 있도록 설정
        self.rate_limit_delay = 1  # 기본 1초
    
    @abstractmethod
    def collect(self, artist_account):
        """아티스트의 계정 정보를 바탕으로 데이터를 수집하는 메서드."""
        pass
    
    def _safe_request(self, url, headers=None, params=None):
        """Rate limiting과 에러 처리가 포함된 안전한 요청 메서드."""
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()  # 2xx 상태 코드가 아닐 경우 예외 발생
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패: {e}")
        return None
