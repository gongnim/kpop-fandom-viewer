# Python 3.9-slim 버전을 기반으로 이미지 생성
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt를 먼저 복사하여 라이브러리 설치 (레이어 캐싱 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 프로젝트 파일 전체 복사
COPY . .

# Streamlit이 사용하는 8501 포트 노출
EXPOSE 8501

# 컨테이너 실행 시 Streamlit 앱 실행
# --server.enableCORS=false 와 --server.enableXsrfProtection=false는 외부 환경 배포 시 필요할 수 있음
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
