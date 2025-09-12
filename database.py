
import sqlite3
from logger_config import logger
from datetime import datetime # Added this line

DATABASE_NAME = 'kpop_dashboard.db'

def get_db_connection():
    """데이터베이스 연결을 생성하고 반환합니다."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """데이터베이스 테이블을 초기화합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Companies Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        company_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        parent_company_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (parent_company_id) REFERENCES companies (company_id)
    )
    """)
    
    # Groups Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS groups (
        group_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        company_id INTEGER,
        debut_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (company_id) REFERENCES companies (company_id)
    )
    """)

    # Artists Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS artists (
        artist_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        name_kr TEXT,
        fullname_kr TEXT,
        group_id INTEGER,
        nationality_name TEXT,
        nationality_code TEXT,
        birth_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (group_id) REFERENCES groups (group_id),
        UNIQUE(name, group_id)
    )
    """)

    # Artist Platform Accounts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS artist_accounts (
        account_id INTEGER PRIMARY KEY AUTOINCREMENT,
        artist_id INTEGER,
        platform TEXT NOT NULL,
        account_identifier TEXT NOT NULL,
        url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
        UNIQUE(artist_id, platform)
    )
    """)

    # Platform Metrics Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS platform_metrics (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id INTEGER,
        platform TEXT NOT NULL,
        metric_type TEXT NOT NULL, -- e.g., 'subscribers', 'followers', 'monthly_listeners'
        value INTEGER,
        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (account_id) REFERENCES artist_accounts (account_id)
    )
    """)

    # Albums Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS albums (
        album_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        artist_id INTEGER,
        release_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (artist_id) REFERENCES artists (artist_id)
    )
    """)

    # Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, -- 이벤트명 (예: 'BTS 생일', 'NewJeans 데뷔')
        event_type TEXT, -- 이벤트 종류 (예: '생일', '그룹 데뷔', '앨범 발매', '콘서트', '팬미팅')
        event_date DATE,
        artist_id INTEGER,
        group_id INTEGER,
        company_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
        FOREIGN KEY (group_id) REFERENCES groups (group_id),
        FOREIGN KEY (company_id) REFERENCES companies (company_id)
    )
    """)

    # API Quotas Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_quotas (
        platform TEXT PRIMARY KEY,
        date DATE NOT NULL,
        usage_count INTEGER DEFAULT 0
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully with new schema.")

# --- 데이터 추가/수정 함수 (이전과 동일) --- #
def add_company(name, parent_company_id=None):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO companies (name, parent_company_id) VALUES (?, ?)", (name, parent_company_id))
        conn.commit()
    except sqlite3.IntegrityError:
        logger.warning(f"Company '{name}' already exists.")
    finally:
        conn.close()

def add_group(name, company_id, debut_date=None):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO groups (name, company_id, debut_date) VALUES (?, ?, ?)", (name, company_id, debut_date))
        conn.commit()
    except sqlite3.IntegrityError:
        logger.warning(f"Group '{name}' already exists.")
    finally:
        conn.close()

def add_artist(name, name_kr, fullname_kr, group_id, nationality_name, nationality_code, birth_date=None):
    conn = get_db_connection()
    try:
        conn.execute("""INSERT INTO artists (name, name_kr, fullname_kr, group_id, nationality_name, nationality_code, birth_date) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                     (name, name_kr, fullname_kr, group_id, nationality_name, nationality_code, birth_date))
        conn.commit()
    except sqlite3.IntegrityError:
        logger.warning(f"Artist '{name}' might already exist in the same group.")
    finally:
        conn.close()

def add_artist_account(artist_id, platform, account_identifier, url=None):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO artist_accounts (artist_id, platform, account_identifier, url) VALUES (?, ?, ?, ?)", 
                     (artist_id, platform, account_identifier, url))
        conn.commit()
    except sqlite3.IntegrityError:
        logger.warning(f"Account for artist_id {artist_id} on {platform} already exists.")
    finally:
        conn.close()

def add_album(name, artist_id, release_date):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO albums (name, artist_id, release_date) VALUES (?, ?, ?)", (name, artist_id, release_date))
        conn.commit()
    except sqlite3.IntegrityError:
        logger.warning(f"Album '{name}' for artist_id {artist_id} already exists.")
    finally:
        conn.close()

def add_event(name, event_type, event_date, artist_id=None, group_id=None, company_id=None):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO events (name, event_type, event_date, artist_id, group_id, company_id) VALUES (?, ?, ?, ?, ?, ?)",
                       (name, event_type, event_date, artist_id, group_id, company_id))
        conn.commit()
    except sqlite3.IntegrityError:
        logger.warning(f"Event '{name}' on {event_date} already exists.")
    finally:
        conn.close()

# --- 데이터 조회 함수 (수정됨) --- #
def _rows_to_dicts(rows):
    """sqlite3.Row 객체 리스트를 딕셔너리 리스트로 변환합니다."""
    return [dict(row) for row in rows]

def get_companies():
    conn = get_db_connection()
    companies = conn.execute('SELECT * FROM companies ORDER BY name').fetchall()
    conn.close()
    return _rows_to_dicts(companies)

def get_listed_companies():
    """상장사 (HYBE, SM, JYP, YG, FNC)만 조회합니다."""
    listed_companies = ["HYBE", "SM Entertainment", "JYP Entertainment", "YG Entertainment", "FNC Entertainment"]
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = f"SELECT * FROM companies WHERE name IN ({','.join(['?' for _ in listed_companies])}) ORDER BY name"
        cursor.execute(query, listed_companies)
        return _rows_to_dicts(cursor.fetchall())

def get_company_by_name(company_name):
    """회사 이름으로 회사 정보를 조회합니다."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM companies WHERE name = ?", (company_name,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_subsidiaries(company_id):
    """주어진 회사 ID의 모든 자회사 (직접 및 간접)를 재귀적으로 조회합니다."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        subsidiaries = []
        
        # 직접 자회사 조회
        cursor.execute("SELECT company_id, name FROM companies WHERE parent_company_id = ?", (company_id,))
        direct_subs = _rows_to_dicts(cursor.fetchall())
        subsidiaries.extend(direct_subs)
        
        # 재귀적으로 간접 자회사 조회
        for sub in direct_subs:
            subsidiaries.extend(get_subsidiaries(sub['company_id']))
            
        return subsidiaries

def get_groups_by_company_id(company_id):
    """주어진 회사 ID에 속한 모든 그룹을 조회합니다."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT group_id, name FROM groups WHERE company_id = ?", (company_id,))
        return _rows_to_dicts(cursor.fetchall())

def get_artists_by_group_id(group_id):
    """주어진 그룹 ID에 속한 모든 아티스트를 조회합니다."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT artist_id, name FROM artists WHERE group_id = ?", (group_id,))
        return _rows_to_dicts(cursor.fetchall())

def get_groups_and_artists_in_hierarchy(company_id):
    """주어진 회사 및 그 자회사에 속한 모든 그룹과 아티스트를 계층적으로 조회합니다."""
    hierarchy = []
    
    # 현재 회사에 직접 속한 그룹 조회
    main_company_groups = get_groups_by_company_id(company_id)
    for group in main_company_groups:
        group_info = {
            'group_id': group['group_id'],
            'group_name': group['name'],
            'artists': get_artists_by_group_id(group['group_id'])
        }
        hierarchy.append(group_info)
        
    # 자회사 및 그 자회사에 속한 그룹과 아티스트 조회
    subsidiaries = get_subsidiaries(company_id)
    for sub_company in subsidiaries:
        sub_company_groups = get_groups_by_company_id(sub_company['company_id'])
        for group in sub_company_groups:
            group_info = {
                'group_id': group['group_id'],
                'group_name': f"{sub_company['name']} - {group['name']}", # 자회사명 포함
                'artists': get_artists_by_group_id(group['group_id'])
            }
            hierarchy.append(group_info)
            
    return hierarchy

def get_groups(company_id=None):
    conn = get_db_connection()
    if company_id:
        rows = conn.execute('SELECT * FROM groups WHERE company_id = ? ORDER BY name', (company_id,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM groups ORDER BY name').fetchall()
    conn.close()
    return _rows_to_dicts(rows)

def get_artists(group_id=None):
    conn = get_db_connection()
    if group_id:
        rows = conn.execute('SELECT * FROM artists WHERE group_id = ? ORDER BY name', (group_id,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM artists ORDER BY name').fetchall()
    conn.close()
    return _rows_to_dicts(rows)

def get_artists_by_company(company_id):
    conn = get_db_connection()
    rows = conn.execute("""SELECT a.artist_id, a.name, g.name as group_name
                        FROM artists a
                        JOIN groups g ON a.group_id = g.group_id
                        WHERE g.company_id = ?
                        ORDER BY g.name, a.name""", (company_id,)).fetchall()
    conn.close()
    return _rows_to_dicts(rows)

def get_all_artists_with_details():
    conn = get_db_connection()
    rows = conn.execute("""SELECT a.artist_id, a.name as artist_name, a.name_kr, g.name as group_name, c.name as company_name, a.nationality_name
                        FROM artists a
                        LEFT JOIN groups g ON a.group_id = g.group_id
                        LEFT JOIN companies c ON g.company_id = c.company_id
                        ORDER BY c.name, g.name, a.name""").fetchall()
    conn.close()
    return _rows_to_dicts(rows)

def get_all_artist_accounts():
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM artist_accounts').fetchall()
    conn.close()
    return _rows_to_dicts(rows)

def get_latest_metric(artist_id, platform):
    conn = get_db_connection()
    rows = conn.execute(""" 
        SELECT metric_type, value
        FROM platform_metrics
        WHERE account_id IN (SELECT account_id FROM artist_accounts WHERE artist_id = ? AND platform = ?)
        AND collected_at = (SELECT MAX(collected_at) FROM platform_metrics WHERE account_id IN (SELECT account_id FROM artist_accounts WHERE artist_id = ? AND platform = ?))
    """, (artist_id, platform, artist_id, platform)).fetchall()
    conn.close()
    return {row['metric_type']: row['value'] for row in rows}

def get_all_metrics_for_artist(artist_id):
    conn = get_db_connection()
    rows = conn.execute("""SELECT pm.platform, pm.metric_type, pm.value, pm.collected_at
                        FROM platform_metrics pm
                        JOIN artist_accounts aa ON pm.account_id = aa.account_id
                        WHERE aa.artist_id = ? ORDER BY pm.collected_at""", (artist_id,)).fetchall()
    conn.close()
    return _rows_to_dicts(rows)

def get_events_for_artist(artist_id):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Get group_id and company_id for the given artist
        cursor.execute("""
            SELECT 
                a.group_id,
                g.company_id
            FROM artists a
            LEFT JOIN groups g ON a.group_id = g.group_id
            WHERE a.artist_id = ?
        """, (artist_id,))
        artist_details = cursor.fetchone()
        
        if not artist_details:
            return []
            
        group_id = artist_details['group_id']
        company_id = artist_details['company_id']
        
        # Query events related to the artist, their group, or their company
        query = """
            SELECT 
                name, 
                event_type, 
                event_date
            FROM events
            WHERE artist_id = ? 
            OR (group_id = ? AND group_id IS NOT NULL) 
            OR (company_id = ? AND company_id IS NOT NULL)
            ORDER BY event_date
        """
        cursor.execute(query, (artist_id, group_id, company_id))
        
        return _rows_to_dicts(cursor.fetchall())
    finally:
        conn.close()

def get_api_usage(platform):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("SELECT usage_count, date FROM api_quotas WHERE platform = ?", (platform,))
        row = cursor.fetchone()
        
        if row:
            if row['date'] == today:
                return row['usage_count']
            else:
                # Reset for new day
                cursor.execute("UPDATE api_quotas SET usage_count = 0, date = ? WHERE platform = ?", (today, platform))
                conn.commit()
                return 0
        else:
            # First entry for platform
            cursor.execute("INSERT INTO api_quotas (platform, date, usage_count) VALUES (?, ?, 0)", (platform, today))
            conn.commit()
            return 0
    finally:
        conn.close()

def increment_api_usage(platform):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Ensure entry exists for today
        get_api_usage(platform) # This will create/reset if needed
        
        cursor.execute("UPDATE api_quotas SET usage_count = usage_count + 1 WHERE platform = ? AND date = ?", (platform, today))
        conn.commit()
    finally:
        conn.close()
