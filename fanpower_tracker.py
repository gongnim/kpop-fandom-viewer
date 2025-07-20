# fanpower_tracker.py
import requests, pandas as pd, os, time, streamlit as st, json
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Streamlit Secrets에서 API 키 불러오기 ---
try:
    YOUTUBE_API_KEY = st.secrets["youtube"]["api_key"]
except (FileNotFoundError, KeyError):
    st.error("YouTube API 키를 찾을 수 없습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
    st.stop()

# --- 아티스트 이름-ID 매핑 테이블 (한글/영문 통합) ---
ARTIST_MAP = {
    "BTS": {"yt": "UCLkAepWjdylmXSltofFvsYQ", "ig": "bts.bighitofficial", "tw": "bts_bighit"},
    "방탄소년단": {"yt": "UCLkAepWjdylmXSltofFvsYQ", "ig": "bts.bighitofficial", "tw": "bts_bighit"},
    "BLACKPINK": {"yt": "UCOmHUn--16B90oW2L6FRR3A", "ig": "blackpinkofficial", "tw": "BLACKPINK"},
    "블랙핑크": {"yt": "UCOmHUn--16B90oW2L6FRR3A", "ig": "blackpinkofficial", "tw": "BLACKPINK"},
    "IVE": {"yt": "UC-Fnix71vRP64WXeo0ikd0Q", "ig": "ivestarship", "tw": "IVEstarship"},
    "아이브": {"yt": "UC-Fnix71vRP64WXeo0ikd0Q", "ig": "ivestarship", "tw": "IVEstarship"},
    "NewJeans": {"yt": "UC3SyT4_WLHzN7JmHQwKQZww", "ig": "newjeans_official", "tw": "NewJeans_ADOR"},
    "뉴진스": {"yt": "UC3SyT4_WLHzN7JmHQwKQZww", "ig": "newjeans_official", "tw": "NewJeans_ADOR"},
    "ITZY": {"yt": "UCaO6TYtlC8U5ttz62hTrZgg", "ig": "itzy.all.in.us", "tw": "ITZYofficial"},
    "있지": {"yt": "UCaO6TYtlC8U5ttz62hTrZgg", "ig": "itzy.all.in.us", "tw": "ITZYofficial"},
    "Stray Kids": {"yt": "UC9rMiEjNaCSsebs31MRDCRA", "ig": "realstraykids", "tw": "Stray_Kids"},
    "스트레이 키즈": {"yt": "UC9rMiEjNaCSsebs31MRDCRA", "ig": "realstraykids", "tw": "Stray_Kids"},
    "(G)I-DLE": {"yt": "UCritG3Vvfk1N_w4gu2s9aYw", "ig": "official_g_i_dle", "tw": "G_I_DLE"},
    "(여자)아이들": {"yt": "UCritG3Vvfk1N_w4gu2s9aYw", "ig": "official_g_i_dle", "tw": "G_I_DLE"},
    "LE SSERAFIM": {"yt": "UCs-QBT4qkj_YiQw1ZntDO3g", "ig": "le_sserafim", "tw": "IM_LESSERAFIM"},
    "르세라핌": {"yt": "UCs-QBT4qkj_YiQw1ZntDO3g", "ig": "le_sserafim", "tw": "IM_LESSERAFIM"},
    "aespa": {"yt": "UC9GtSLeksfK4yuJ_g1lgQbg", "ig": "aespa_official", "tw": "aespa_official"},
    "에스파": {"yt": "UC9GtSLeksfK4yuJ_g1lgQbg", "ig": "aespa_official", "tw": "aespa_official"},
    "ENHYPEN": {"yt": "UCArLZtok93cO5R9RI4_Y5Jw", "ig": "enhypen", "tw": "ENHYPEN_members"},
    "엔하이픈": {"yt": "UCArLZtok93cO5R9RI4_Y5Jw", "ig": "enhypen", "tw": "ENHYPEN_members"},
    "TWICE": {"yt": "UCzgxx_DM2Dcb9Y1spb9mUJA", "ig": "twicetagram", "tw": "JYPETWICE"},
    "트와이스": {"yt": "UCzgxx_DM2Dcb9Y1spb9mUJA", "ig": "twicetagram", "tw": "JYPETWICE"},
}

# --- 데이터 수집 함수 ---

@st.cache_data(ttl=3600)
def yt_stats(cid):
    try:
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        res = yt.channels().list(part="statistics", id=cid).execute()
        s = res["items"][0]["statistics"]
        return int(s["subscriberCount"]), int(s["viewCount"]), int(s["videoCount"])
    except Exception as e:
        st.error(f"[YouTube Error] {cid}: {e}")
        return "Error", "Error", "Error"

@st.cache_data(ttl=3600)
def insta_followers(username):
    # 인스타그램은 로그인 없이는 팔로워 수 스크래핑이 매우 어려움
    return "N/A (수집 불가)"

@st.cache_data(ttl=3600, show_spinner=False)
def spotify_listeners(name):
    """
    Spotify 월간 청취자 수 (requests 기반, 실패 시 링크 반환)
    """
    #st.info(f"-> Spotify 월간 청취자 조회 시도: {name}")
    try:
        q = requests.utils.quote(name)
        url = f"https://open.spotify.com/search/{q}/artists"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, "lxml")
        # 월간 청취자 수를 포함하는 요소를 찾습니다. (이전의 JSON-LD 시도도 제거)
        listener_element = soup.find('div', string=lambda text: text and 'monthly listeners' in text.lower())
        if listener_element:
            return listener_element.text.strip()

        raise ValueError("월간 청취자 정보를 찾을 수 없습니다.")

    except Exception as e:
        #st.warning(f"[Spotify Error] {name}: {e}")
        return "N/A"

# --- 링크 생성 및 메인 수집 함수 ---
def get_artist_id(artist_name, platform):
    return ARTIST_MAP.get(artist_name, {}).get(platform)

def twitter_link(username): return f"https://twitter.com/{username}" if username else "N/A"
def instagram_link(username): return f"https://instagram.com/{username}" if username else "N/A"
def spotify_link(artist_name): return f"https://open.spotify.com/search/{artist_name.replace(' ', '%20')}/artists"

@st.cache_data(ttl=86400)
def search_youtube_channel(query):
    #st.info(f"-> YouTube 채널 자동 검색: {query}")
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        res = youtube.search().list(part="snippet", type="channel", q=query, maxResults=1).execute()
        return res["items"][0]["snippet"]["channelId"]
    except Exception as e:
        #st.error(f"⚠️ 채널 자동 검색 실패: {query} → {e}")
        return None

def collect(artist_list):
    rows = []
    progress_bar = st.progress(0, text="수집 대기 중...")
    for i, a in enumerate(artist_list):
        progress_bar.progress((i + 1) / len(artist_list), text=f"{a} 데이터 수집 중...")
        
        cid = get_artist_id(a, 'yt') or search_youtube_channel(a)
        row = {"Artist": a}
        if cid:
            sub, view, vid = yt_stats(cid)
            row.update({"YouTube Subs": sub, "YouTube Views": view, "YouTube Videos": vid})
        else:
            row.update({"YouTube Subs": "N/A", "YouTube Views": "N/A", "YouTube Videos": "N/A"})
            
        row["TWITTER"] = twitter_link(get_artist_id(a, 'tw'))
        row["Instagram"] = instagram_link(get_artist_id(a, 'ig'))
        spotify_data = spotify_listeners(a)
        row["Spotify"] = spotify_link(a) if "N/A" in spotify_data else spotify_data
        rows.append(row)
        
    progress_bar.empty()
    st.success("모든 아티스트 데이터 수집 완료!")
    return pd.DataFrame(rows)

# --- Streamlit 앱 UI 구성 ---
if __name__ == '__main__':
    st.set_page_config(page_title="아티스트 팬 파워 트래커", layout="wide")
    st.title("🎤 아티스트 팬 파워 트래커")
    st.markdown("아티스트의 유튜브, 스포티파이, 인스타그램, 트위터 지표를 추적합니다.")

    all_artists = sorted(list(ARTIST_MAP.keys()))
    default_artists = sorted(["BTS", "블랙핑크", "아이브", "뉴진스", "르세라핌", "Stray Kids"])

    selected_artists = st.multiselect("조회할 아티스트를 선택하세요.", options=all_artists, default=default_artists)
    
    if st.button("데이터 조회 시작", type="primary"):
        if selected_artists:
            fan_data = collect(selected_artists)
            st.header("📊 조회 결과")
            st.dataframe(fan_data)
            st.download_button("결과 다운로드 (CSV)", fan_data.to_csv(index=False).encode('utf-8-sig'), "fanpower_data.csv", "text/csv")
        else:
            st.warning("아티스트를 선택해주세요.")

    st.sidebar.header("⚠️ 주의사항")
    st.sidebar.info("""
    - **인스타그램/트위터:** 팔로워 수 자동 수집은 해당 플랫폼의 스크래핑 방지 정책으로 인해 불가능합니다. 현재는 프로필 링크를 제공합니다.
    - **스포티파이:** 월간 청취자 수는 비공식적인 웹 스크래핑 방식으로는 안정적인 수집이 매우 어렵습니다. 현재는 검색 결과 페이지 링크를 제공합니다.
    - **유튜브:** 공식 API를 사용하므로 가장 안정적으로 데이터를 수집합니다.
    """)


