# 🎵 K-Pop Entertainment Analysis Dashboard

이 프로젝트는 YouTube, Spotify, Twitter 등 공개된 API를 활용하여 K-Pop 아티스트 관련 데이터를 수집, 분석하고 시각화하는 Streamlit 기반의 대시보드입니다.

---

## ✨ 주요 기능

- **데이터 수집**: YouTube, Spotify, Twitter API를 통해 아티스트의 구독자, 팔로워, 인기도 등의 지표를 주기적으로 수집합니다.
- **기업별 분석**: 특정 엔터테인먼트사를 선택하여 소속 아티스트들의 주요 지표를 한눈에 비교합니다.
- **아티스트별 분석**: 개별 아티스트의 시간에 따른 플랫폼별 성장 추이를 시계열 차트로 확인합니다.
- **플랫폼 비교**: 여러 아티스트를 선택하여 플랫폼 간 영향력을 비교 분석합니다.
- **데이터 관리**: 대시보드에 필요한 회사, 그룹, 아티스트 정보를 UI를 통해 직접 관리할 수 있습니다.
- **자동화 및 인증**: 데이터 수집이 매일 자동으로 실행되며, 비밀번호를 통해 허가된 사용자만 접근할 수 있습니다.

---

## 🚀 실행 방법

### 1. 사전 준비

- Python 3.9 이상
- 필요한 API 키 발급:
  - YouTube Data API v3
  - Spotify Web API
  - Twitter API v2

### 2. 로컬 환경 설정

1.  **저장소 복제 및 이동**
    ```bash
    # 이 프로젝트 디렉토리로 이동합니다.
    cd kpop_dashboard
    ```

2.  **가상환경 생성 및 활성화**
    ```bash
    # 가상환경 생성
    python3 -m venv venv

    # macOS / Linux
    source venv/bin/activate

    # Windows
    # venv\Scripts\activate
    ```

3.  **필요 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **환경 변수 설정**
    `.env.example` 파일을 복사하여 `.env` 파일을 생성하고, 발급받은 API 키를 입력합니다.
    ```bash
    cp .env.example .env
    ```
    ```.env
    YOUTUBE_API_KEY=여러분의_YouTube_API_키
    SPOTIFY_CLIENT_ID=여러분의_Spotify_Client_ID
    SPOTIFY_CLIENT_SECRET=여러분의_Spotify_Client_Secret
    TWITTER_BEARER_TOKEN=여러분의_Twitter_Bearer_Token
    ```

5.  **데이터베이스 초기화**
    `database.py`를 직접 실행하여 `kpop_dashboard.db` 파일을 생성하고 테이블을 초기화합니다.
    ```bash
    python database.py
    ```

### 3. 애플리케이션 실행

아래 명령어를 실행하면 웹 브라우저에서 대시보드가 열립니다.

```bash
streamlit run app.py
```

최초 접속 시 `.streamlit/secrets.toml` 파일에 설정된 비밀번호를 입력해야 합니다. (예: `secure_password_123`)

---

## 🐳 배포

### Docker

`Dockerfile`이 포함되어 있어 Docker를 사용하여 쉽게 컨테이너화할 수 있습니다.

```bash
# Docker 이미지 빌드
docker build -t kpop-dashboard .

# Docker 컨테이너 실행
docker run -p 8501:8501 kpop-dashboard
```

### Railway

`railway.toml` 파일이 포함되어 있어 Railway.app에 저장소를 연결하면 자동으로 빌드 및 배포가 진행됩니다. 환경 변수만 Railway 프로젝트 설정에 추가해주면 됩니다.
