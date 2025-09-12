import logging
import sys

def setup_logger():
    """프로젝트 전반에 사용될 로거를 설정합니다."""
    # 루트 로거를 가져오고, 핸들러가 이미 설정되어 있다면 중복 추가를 방지
    if logging.getLogger().hasHandlers():
        return logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        handlers=[
            # 로그 파일에 기록하는 핸들러
            logging.FileHandler("kpop_dashboard.log"),
            # 콘솔(터미널)에 출력하는 핸들러
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # apscheduler와 같은 외부 라이브러리의 로그 레벨을 WARNING으로 설정하여 너무 많은 로그 방지
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# 로거 인스턴스 생성
logger = setup_logger()
