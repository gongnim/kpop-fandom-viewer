# fanpower_tracker.py
import requests, pandas as pd, os, time, streamlit as st
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Streamlit Secrets에서 API 키 불러오기 ---
# Streamlit Cloud 배포 시에는 앱 설정에서, 로컬에서는 .streamlit/secrets.toml 파일에서 키를 읽어옵니다.
try:
    YOUTUBE_API_KEY = st.secrets["youtube"]["api_key"]
except (FileNotFoundError, KeyError):
    st.error("YouTube API 키를 찾을 수 없습니다. .streamlit/secrets.toml 파일을 설정하거나 Streamlit Cloud의 Secrets에 키를 추가해주세요.")
    st.stop()

# 필요하면 ID 테이블을 늘려두세요
YT_CHANNEL = {
    "BTS": "UCLkAepWjdylmXSltofFvsYQ",
    "BLACKPINK": "UCOmHUn--16B90oW2L6FRR3A",
    "IVE": "UCcvqv1suY0rmjRBAZ0HjpVg",
    "NewJeans": "UC3SyT4_WLHzN7JmHQwKQZww",
    "ITZY": "UCaO6TYtlC8U5ttz62hTrZgg",
    "스트레이 키즈": "UC9rMiEjNaCSsebs31MRDCRA",
    "엔믹스": "UCJh7mYgS-a8pF6H4XLCtV2w",
    "TWICE": "UCzgxx_DM2Dcb9Y1spb9mUJA"
}
IG_ID = {
    "BTS": "bts.bighitofficial", "BLACKPINK": "blackpinkofficial",
    "Stray Kids": "realstraykids",  "NewJeans": "newjeans_official",
}
TW_ID = {
    "BTS": "bts_bighit", "BLACKPINK": "BLACKPINK",
    "Stray Kids": "Stray_Kids", "NewJeans": "NewJeans_ADOR",
}

# --- 안정성 개선: 재시도 로직 및 상세 예외 처리 ---

@st.cache_data(ttl=3600) # 1시간 동안 API 응답 캐싱
def requests_get_with_retry(url, retries=3, delay=5, timeout=10):
    """
    일시적인 네트워크 오류에 대응하기 위한 재시도 로직을 포함한 GET 요청 함수
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # 200 OK가 아니면 예외 발생
            return response
        except requests.exceptions.RequestException as e:
            print(f"  - 요청 실패 (시도 {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(delay)
    return None

@st.cache_data(ttl=3600) # 1시간 동안 API 응답 캐싱
def yt_stats(cid):
    """
    YouTube API를 사용하여 채널 통계 수집 (상세 예외 처리 추가)
    """
    try:
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        res = yt.channels().list(part="statistics", id=cid).execute()
        items = res.get("items", [])
        if not items:
            raise ValueError("채널 ID가 유효하지 않거나 비공개 채널입니다.")
        s = items[0]["statistics"]
        return int(s["subscriberCount"]), int(s["viewCount"]), int(s["videoCount"])
    except HttpError as e:
        print(f"[YouTube API Error] {cid}: {e.content.decode('utf-8')}")
    except Exception as e:
        print(f"[YouTube Error] {cid}: {e}")
    return "Error", "Error", "Error"

@st.cache_data(ttl=3600)
def insta_followers(username):
    """
    Instagram 팔로워 수 스크래핑 (재시도 로직 적용)
    참고: 인스타그램의 구조 변경 시 실패할 확률이 높습니다.
    """
    print(f"-> Instagram 팔로워 조회 시도: {username}")
    url = f"https://www.instagram.com/{username}/"
    r = requests_get_with_retry(url)

    if not r:
        print(f"[Instagram Error] {username}: 요청 실패")
        return "N/A"

    try:
        soup = BeautifulSoup(r.text, 'lxml')
        meta = soup.find('meta', property='og:description')
        
        if not meta or "Followers" not in meta['content']:
            raise ValueError("메타 태그를 찾을 수 없거나 형식이 다릅니다.")
            
        followers = meta['content'].split(" Followers")[0]
        return f"{followers} Followers"
    except Exception as e:
        print(f"[Instagram Error] {username}: {e}")
        return "N/A"

def twitter_followers(user):
    """
    Twitter(X) 팔로워 수 스크래핑.
    참고: 현재 X는 스크래핑이 거의 불가능하므로, 링크를 반환하는 것이 현실적입니다.
    """
    print(f"-> Twitter(X)는 현재 스크래핑을 지원하지 않으므로 링크를 반환합니다: {user}")
    return "Link Only"

@st.cache_data(ttl=3600)
def spotify_listeners(name):
    """
    Spotify 월간 청취자 수 스크래핑 (안정성 개선)
    """
    print(f"-> Spotify 월간 청취자 조회 시도: {name}")
    q = requests.utils.quote(name)
    url = f"https://open.spotify.com/search/{q}/artists"
    r = requests_get_with_retry(url)

    if not r:
        print(f"[Spotify Error] {name}: 요청 실패")
        return "N/A"

    try:
        soup = BeautifulSoup(r.text, "lxml")
        # "N명의 월별 리스너" 패턴을 포함하는 요소를 찾습니다.
        listener_element = soup.find('div', string=lambda text: text and 'monthly listeners' in text.lower())
        if listener_element:
            return listener_element.text.strip()
        
        # Fallback: 이전 방식 시도
        if "monthly listeners" in r.text:
            idx = r.text.lower().find("monthly listeners")
            snippet = r.text[idx-50:idx+50]
            return snippet.split(" monthly listeners")[0].split('>')[-1].strip()

        raise ValueError("월간 청취자 정보를 찾을 수 없습니다.")
    except Exception as e:
        print(f"[Spotify Error] {name}: {e}")
        return "N/A"

# --- 링크 생성 헬퍼 함수 ---
def twitter_link(username):
    return f"https://twitter.com/{username}" if username else "N/A"

def instagram_link(username):
    return f"https://instagram.com/{username}" if username else "N/A"

def spotify_link(artist_name):
    q = requests.utils.quote(artist_name)
    # 아티스트 검색 결과 페이지로 바로 연결
    return f"https://open.spotify.com/search/{q}/artists"

@st.cache_data(ttl=86400) # 24시간 동안 검색 결과 캐싱
def search_youtube_channel(query):
    """
    아티스트 이름으로 유튜브 채널 ID 검색
    """
    print(f"-> YouTube 채널 검색: {query}")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        res = youtube.search().list(
            part="snippet", type="channel", q=query, maxResults=1
        ).execute()
        if not res.get("items"):
            raise ValueError("검색 결과가 없습니다.")
        channel_title = res["items"][0]["snippet"]["title"]
        channel_id = res["items"][0]["snippet"]["channelId"]
        print(f"  - 채널 찾음: {channel_title} ({channel_id})")
        return channel_title, channel_id
    except Exception as e:
        print(f"⚠️ 채널 검색 실패: {query} → {e}")
        return None, None

def collect(artist_list):
    """
    아티스트 목록을 기반으로 모든 팬덤 파워 데이터를 수집합니다.
    """
    rows = []
    
    progress_bar = st.progress(0)
    total_artists = len(artist_list)

    for i, a in enumerate(artist_list):
        st.text(f"--- {a} 처리 중 ---")
        cid = YT_CHANNEL.get(a)
        if not cid:
            _, cid = search_youtube_channel(a)
            
        tid = TW_ID.get(a)
        iid = IG_ID.get(a)
        
        row = {"Artist": a}
        
        # YouTube
        if cid:
            sub, view, vid = yt_stats(cid)
            row.update({"YouTube Subs": sub, "YouTube Views": view, "YouTube Videos": vid})
        else:
            row.update({"YouTube Subs": "N/A", "YouTube Views": "N/A", "YouTube Videos": "N/A"})
            
        # Twitter (현재 스크래핑이 거의 불가능하여 링크만 제공)
        row["TWITTER"] = twitter_link(tid)
        
        # Instagram (팔로워 수 시도 -> 실패 시 링크 대체)
        if iid:
            insta_data = insta_followers(iid)
            if "N/A" in insta_data:
                row["Instagram"] = instagram_link(iid)
            else:
                row["Instagram"] = insta_data
        else:
            row["Instagram"] = "N/A"
        
        # Spotify (월간 청취자 수 시도 -> 실패 시 링크 대체)
        spotify_data = spotify_listeners(a)
        if "N/A" in spotify_data:
            row["Spotify"] = spotify_link(a)
        else:
            row["Spotify"] = spotify_data
            
        rows.append(row)
        progress_bar.progress((i + 1) / total_artists)

    st.success("데이터 수집 완료!")
    return pd.DataFrame(rows)

# Streamlit 앱 UI 구성
if __name__ == '__main__':
    st.title("아티스트 팬 파워 트래커")

    st.markdown("""
    이 앱은 지정된 아티스트의 소셜 미디어 및 스트리밍 플랫폼에서의 팬 파워 지표를 추적합니다.
    - **YouTube**: 구독자 수, 총 조회수, 동영상 수
    - **Instagram**: 팔로워 수 (실패 시 프로필 링크)
    - **Twitter(X)**: 프로필 링크
    - **Spotify**: 월간 청취자 수 (실패 시 검색 링크)
    """)

    # 기본 아티스트 목록 + 사용자가 추가할 수 있도록
    default_artists = ["BTS", "BLACKPINK", "NewJeans", "IVE", "LE SSERAFIM", "Stray Kids"]
    
    st.header("아티스트 선택")
    
    # 멀티셀렉트 박스
    selected_artists = st.multiselect(
        "조회할 아티스트를 선택하세요.",
        options=list(YT_CHANNEL.keys()) + ["LE SSERAFIM", "Stray Kids"], # 예시 목록 확장
        default=default_artists
    )

    # 텍스트 입력으로 새 아티스트 추가
    new_artist = st.text_input("새 아티스트 추가 (이름으로 YouTube 채널 검색):")
    if new_artist and new_artist not in selected_artists:
        selected_artists.append(new_artist)

    if st.button("데이터 조회 시작"):
        if not selected_artists:
            st.warning("조회할 아티스트를 한 명 이상 선택해주세요.")
        else:
            with st.spinner('데이터를 수집하는 중입니다... 잠시만 기다려주세요.'):
                fan_data = collect(selected_artists)
            
            st.header("조회 결과")
            st.dataframe(fan_data)
            
            # CSV 다운로드 버튼
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

            csv = convert_df_to_csv(fan_data)

            st.download_button(
                label="결과를 CSV 파일로 다운로드",
                data=csv,
                file_name='fanpower_data.csv',
                mime='text/csv',
            )

    st.sidebar.header("About")
    st.sidebar.info("이 앱은 Gemini를 사용하여 생성 및 개선되었습니다.")

