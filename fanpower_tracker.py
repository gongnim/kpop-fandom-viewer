# fanpower_tracker.py
import requests, pandas as pd
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

YOUTUBE_API_KEY = "AIzaSyAUsjVCSRBSr1EIIwdqgIyyj4CstPvQZl4"

# 필요하면 ID 테이블을 늘려두세요
YT_CHANNEL = {
    "BTS": "UCLkAepWjdylmXSltofFvsYQ",  # BANGTANTV
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

def yt_stats(cid):
    try:
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        res = yt.channels().list(part="statistics", id=cid).execute()
        items = res.get("items", [])
        if not items:
            raise ValueError("Invalid channel ID or private channel")
        s = items[0]["statistics"]
        return int(s["subscriberCount"]), int(s["viewCount"]), int(s["videoCount"])
    except Exception as e:
        print(f"[YouTube Error] {cid}: {e}")
        return "Error", "Error", "Error"
        
def insta_followers(username):
    try:
        url = f"https://www.instagram.com/{username}/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)

        if r.status_code != 200:
            raise Exception(f"status {r.status_code}")

        soup = BeautifulSoup(r.text, 'lxml')
        meta = soup.find('meta', property='og:description')
        
        if not meta or "Followers" not in meta['content']:
            raise Exception("meta tag missing or unexpected")
            
        followers = meta['content'].split(" Followers")[0]
        return followers

    except Exception as e:
        print(f"[INSTAGRAM ERROR] {username}: {e}")
        return "N/A"
        
def twitter_followers(user):
    r = requests.get(f"https://twitter.com/{user}",
                     headers={"User-Agent": "Mozilla/5.0"})
    try:
        return BeautifulSoup(r.text, "lxml")\
                 .find("div", attrs={"data-testid": "UserProfileHeader_Items"}).text
    except Exception:
        return "N/A"

def spotify_listeners(name):
    q = name.replace(" ", "%20")
    r = requests.get(f"https://open.spotify.com/search/{q}",
                     headers={"User-Agent": "Mozilla/5.0"})
    if "Monthly listeners" in r.text:
        idx = r.text.find("Monthly listeners")
        snippet = r.text[idx-100:idx+100]
        return BeautifulSoup(snippet, "lxml").text.strip()
    return "N/A"
    
def twitter_link(username):
    return f"https://twitter.com/{username}" if username else "N/A"

def instagram_link(username):
    return f"https://instagram.com/{username}" if username else "N/A"

def spotify_link(artist_name):
    q = artist_name.replace(" ", "%20")
    return f"https://open.spotify.com/search/{q}"

# 기존 코드에 추가 또는 교체

def search_youtube_channel(query):
    """
    아티스트 이름으로 유튜브 채널 ID 검색
    """
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        res = youtube.search().list(
            part="snippet", type="channel", q=query, maxResults=1
        ).execute()
        channel_title = res["items"][0]["snippet"]["channelTitle"]
        channel_id = res["items"][0]["snippet"]["channelId"]
        return channel_title, channel_id
    except Exception as e:
        print(f"⚠️ 채널 검색 실패: {query} → {e}")
        return None, None
        
def collect(artist_list):
    rows = []
    for a in artist_list:
        # YouTube 채널 ID: 딕셔너리에 없으면 자동 검색
        cid = YT_CHANNEL.get(a)
        if not cid:
            _, cid = search_youtube_channel(a)
            
        tid = TW_ID.get(a)
        iid = IG_ID.get(a)
        
        row = {"Artist": a}
        
        if cid:
            try:
                sub, view, vid = yt_stats(cid)
                row.update({"YouTube Subs": sub, "YouTube Views": view, "YouTube Videos": vid})
            except:
                row.update({"YouTube Subs": "Error", "YouTube Views": "Error", "YouTube Videos": "Error"})
        else:
            row.update({"YouTube Subs": "N/A", "YouTube Views": "N/A", "YouTube Videos": "N/A"})
            
        row["TWITTER Followers"] = twitter_link(tid) # twitter_followers(tid) if tid else "N/A"
        
        # instagram : 팔로워 수 시도 -> 실패 시 링크 대체
        if iid:
            insta = insta_followers(iid)
            if insta == "N/A":
                insta = instagram_link(iid)
            row["Instagram"] = insta
        else:
            row["Instagram"] = "N/A"
        
        row["Spotify Listeners"] = spotify_link(a) # spotify_listeners(a)
        rows.append(row)
    return pd.DataFrame(rows)
