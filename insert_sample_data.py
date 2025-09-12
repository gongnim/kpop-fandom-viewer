#!/usr/bin/env python3
"""
샘플 데이터 삽입 스크립트
PostgreSQL 데이터베이스에 테스트용 샘플 데이터를 삽입합니다.
"""
import sys
import os
from datetime import datetime, timedelta
import random

# 경로를 추가하여 kpop_dashboard 모듈을 import할 수 있도록 함
sys.path.append('/Users/jamesgong/gemini/kpop/kpop_dashboard')

from database_postgresql import (
    init_connection_pool, add_company, add_group, add_artist, 
    add_artist_account, get_db_connection
)
import psycopg2

def insert_sample_data():
    """샘플 데이터를 데이터베이스에 삽입"""
    
    print("=== K-Pop 대시보드 샘플 데이터 삽입 시작 ===")
    
    # 연결 풀 초기화
    init_connection_pool()
    
    # 1. 회사 추가
    print("\n1. 회사 데이터 추가 중...")
    companies = [
        "SM Entertainment",
        "JYP Entertainment", 
        "HYBE",
        "YG Entertainment"
    ]
    
    company_ids = {}
    for company in companies:
        company_id = add_company(company, is_sample=True)
        if company_id:
            company_ids[company] = company_id
            print(f"  ✓ {company} 추가 완료 (샘플)")
    
    # 2. 그룹 추가
    print("\n2. 그룹 데이터 추가 중...")
    groups_data = [
        ("aespa", "SM Entertainment", "2020-11-17"),
        ("ITZY", "JYP Entertainment", "2019-02-12"),
        ("NewJeans", "HYBE", "2022-08-01"),
        ("BLACKPINK", "YG Entertainment", "2016-08-08")
    ]
    
    group_ids = {}
    for group_name, company_name, debut_date in groups_data:
        if company_name in company_ids:
            group_id = add_group(group_name, company_ids[company_name], debut_date, is_sample=True)
            if group_id:
                group_ids[group_name] = group_id
                print(f"  ✓ {group_name} 추가 완료 (샘플)")
    
    # 3. 아티스트 추가
    print("\n3. 아티스트 데이터 추가 중...")
    artists_data = [
        # (영문명, 한글명, 본명, 그룹명, 국적명, 국적코드)
        ("KARINA", "카리나", "유지민", "aespa", "대한민국", "KR"),
        ("WINTER", "윈터", "김민정", "aespa", "대한민국", "KR"),
        ("YEJI", "예지", "황예지", "ITZY", "대한민국", "KR"),
        ("RYUJIN", "류진", "신류진", "ITZY", "대한민국", "KR"),
        ("MINJI", "민지", "김민지", "NewJeans", "대한민국", "KR"),
        ("HANNI", "하니", "팜응옥한", "NewJeans", "오스트레일리아", "AU"),
        ("JENNIE", "제니", "김제니", "BLACKPINK", "대한민국", "KR"),
        ("LISA", "리사", "라리사 마노발", "BLACKPINK", "태국", "TH")
    ]
    
    artist_ids = {}
    for name, name_kr, fullname_kr, group_name, nationality_name, nationality_code in artists_data:
        if group_name in group_ids:
            artist_id = add_artist(name, name_kr, fullname_kr, group_ids[group_name], 
                                 nationality_name, nationality_code, is_sample=True)
            if artist_id:
                artist_ids[name] = artist_id
                print(f"  ✓ {name} 추가 완료 (샘플)")
    
    # 4. 그룹 계정 추가
    print("\n4. 그룹 계정 데이터 추가 중...")
    group_accounts_data = [
        # (그룹명, 플랫폼, 계정명, URL)
        ("aespa", "youtube", "@aespa_official", "https://youtube.com/@aespa_official"),
        ("aespa", "spotify", "aespa", "https://open.spotify.com/artist/aespa"),
        ("ITZY", "youtube", "@itzy", "https://youtube.com/@itzy"),
        ("ITZY", "spotify", "itzy", "https://open.spotify.com/artist/itzy"),
        ("NewJeans", "youtube", "@newjeans_official", "https://youtube.com/@newjeans_official"),
        ("NewJeans", "spotify", "newjeans", "https://open.spotify.com/artist/newjeans"),
        ("BLACKPINK", "youtube", "@blackpinkofficial", "https://youtube.com/@blackpinkofficial"),
        ("BLACKPINK", "spotify", "blackpink", "https://open.spotify.com/artist/blackpink"),
    ]
    
    # 5. 아티스트 계정 추가
    print("\n5. 아티스트 계정 데이터 추가 중...")
    artist_accounts_data = [
        # (아티스트명, 플랫폼, 계정명, URL)
        ("KARINA", "youtube", "@karina_official", "https://youtube.com/@karina_official"),
        ("KARINA", "spotify", "karina", "https://open.spotify.com/artist/karina"),
        ("WINTER", "youtube", "@winter_official", "https://youtube.com/@winter_official"),
        ("WINTER", "spotify", "winter", "https://open.spotify.com/artist/winter"),
        ("YEJI", "youtube", "@yeji_official", "https://youtube.com/@yeji_official"),
        ("JENNIE", "youtube", "@jennierubyjane", "https://youtube.com/@jennierubyjane"),
        ("LISA", "youtube", "@lalalalisa_m", "https://youtube.com/@lalalalisa_m"),
    ]
    
    account_ids = {}
    
    # 그룹 계정 추가
    for group_name, platform, account_name, url in group_accounts_data:
        if group_name in group_ids:
            account_id = add_artist_account(platform, account_name, group_id=group_ids[group_name], url=url, is_sample=True)
            if account_id:
                account_ids[f"{group_name}_{platform}"] = account_id
                print(f"  ✓ {group_name} {platform} 그룹 계정 추가 완료 (샘플)")
    
    # 아티스트 계정 추가  
    for artist_name, platform, account_name, url in artist_accounts_data:
        if artist_name in artist_ids:
            account_id = add_artist_account(platform, account_name, artist_id=artist_ids[artist_name], url=url, is_sample=True)
            if account_id:
                account_ids[f"{artist_name}_{platform}"] = account_id
                print(f"  ✓ {artist_name} {platform} 계정 추가 완료 (샘플)")
    
    # 6. 플랫폼 메트릭 샘플 데이터 추가
    print("\n6. 플랫폼 메트릭 샘플 데이터 추가 중...")
    insert_sample_metrics(account_ids)
    
    print("\n=== 샘플 데이터 삽입 완료 ===")
    print("이제 대시보드에서 데이터를 확인할 수 있습니다!")

def insert_sample_metrics(account_ids):
    """플랫폼 메트릭 샘플 데이터 삽입"""
    
    # YouTube 메트릭 타입들
    youtube_metrics = ["subscribers", "total_views", "video_count"]
    spotify_metrics = ["followers", "monthly_listeners"]
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 각 계정에 대해 지난 30일간의 샘플 데이터 생성
            for account_key, account_id in account_ids.items():
                entity_name, platform = account_key.split('_')
                
                # 그룹인지 아티스트인지 구분 (그룹은 더 큰 팬베이스)
                is_group = entity_name in ['aespa', 'ITZY', 'NewJeans', 'BLACKPINK']
                multiplier = 5.0 if is_group else 1.0  # 그룹은 5배 큰 값
                
                if platform == "youtube":
                    metrics = youtube_metrics
                    base_values = {
                        "subscribers": int(1000000 * multiplier), 
                        "total_views": int(50000000 * multiplier), 
                        "video_count": int(100 * (multiplier * 0.5))  # 비디오 수는 덜 차이남
                    }
                elif platform == "spotify":
                    metrics = spotify_metrics  
                    base_values = {
                        "followers": int(800000 * multiplier), 
                        "monthly_listeners": int(2000000 * multiplier)
                    }
                else:
                    continue
                
                # 지난 30일간 데이터 생성
                for days_ago in range(30, 0, -1):
                    collected_at = datetime.now() - timedelta(days=days_ago)
                    
                    for metric_type in metrics:
                        # 기본값에서 랜덤하게 변동
                        base_val = base_values[metric_type]
                        variation = random.uniform(0.95, 1.05)  # ±5% 변동
                        value = int(base_val * variation)
                        
                        cursor.execute("""
                            INSERT INTO platform_metrics (account_id, platform, metric_type, value, collected_at, is_sample)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (account_id, platform, metric_type, value, collected_at, True))
                
                entity_type = "그룹" if is_group else "아티스트"
                print(f"  ✓ {entity_name} {platform} {entity_type} 메트릭 데이터 추가 완료")
            
            conn.commit()
            
    except Exception as e:
        print(f"메트릭 데이터 삽입 중 오류 발생: {e}")

if __name__ == "__main__":
    insert_sample_data()