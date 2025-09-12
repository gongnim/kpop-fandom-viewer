"""
PostgreSQL Database Module for K-Pop Dashboard
Enhanced version of database.py with PostgreSQL support and connection pooling.
"""

import psycopg2
import psycopg2.extras
from psycopg2 import pool
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
import json

from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration from Config object
DB_CONFIG = Config.DB_CONFIG

# Global connection pool
connection_pool = None

def init_connection_pool():
    """Initialize PostgreSQL connection pool."""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG
        )
        logger.info("PostgreSQL connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        raise

@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic cleanup."""
    if connection_pool is None:
        init_connection_pool()
    
    conn = None
    try:
        conn = connection_pool.getconn()
        conn.autocommit = False
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

def init_db():
    """Initialize database tables (schema should already exist from migration)."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(tables)} tables in database: {', '.join(tables)}")
            
            # Verify essential tables exist
            essential_tables = ['companies', 'groups', 'artists', 'artist_accounts', 'platform_metrics']
            missing_tables = [table for table in essential_tables if table not in tables]
            
            if missing_tables:
                logger.warning(f"Missing essential tables: {', '.join(missing_tables)}")
                logger.warning("Please run the PostgreSQL schema migration first!")
            else:
                logger.info("Database initialization verified - all essential tables present")
                
            conn.commit()
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# --- Data insertion functions (Enhanced for PostgreSQL) --- #

def add_company(name: str, parent_company_id: Optional[int] = None) -> Optional[int]:
    """Add a company and return the company_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO companies (name, parent_company_id) VALUES (%s, %s) RETURNING company_id",
                (name, parent_company_id)
            )
            company_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added company: {name}")
            return company_id
    except psycopg2.IntegrityError:
        logger.warning(f"Company '{name}' already exists.")
        return None
    except Exception as e:
        logger.error(f"Failed to add company '{name}': {e}")
        return None

def add_group(name: str, company_id: int, debut_date: Optional[str] = None) -> Optional[int]:
    """Add a group and return the group_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO groups (name, company_id, debut_date) VALUES (%s, %s, %s) RETURNING group_id",
                (name, company_id, debut_date)
            )
            group_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added group: {name}")
            return group_id
    except psycopg2.IntegrityError:
        logger.warning(f"Group '{name}' already exists.")
        return None
    except Exception as e:
        logger.error(f"Failed to add group '{name}': {e}")
        return None

def add_artist(name: str, name_kr: Optional[str], fullname_kr: Optional[str], 
               group_id: int, nationality_name: Optional[str], 
               nationality_code: Optional[str], birth_date: Optional[str] = None) -> Optional[int]:
    """Add an artist and return the artist_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO artists (name, name_kr, fullname_kr, group_id, nationality_name, nationality_code, birth_date) 
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING artist_id
            """, (name, name_kr, fullname_kr, group_id, nationality_name, nationality_code, birth_date))
            artist_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added artist: {name}")
            return artist_id
    except psycopg2.IntegrityError:
        logger.warning(f"Artist '{name}' might already exist in the same group.")
        return None
    except Exception as e:
        logger.error(f"Failed to add artist '{name}': {e}")
        return None

def add_artist_account(platform: str, account_identifier: str, artist_id: Optional[int] = None, group_id: Optional[int] = None, 
                      url: Optional[str] = None, is_active: bool = True) -> Optional[int]:
    """Add an artist or group account and return the account_id."""
    if (artist_id is None and group_id is None) or (artist_id is not None and group_id is not None):
        logger.error("Account must have either an artist_id or a group_id, but not both.")
        return None

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO artist_accounts (artist_id, group_id, platform, account_identifier, url, is_active) 
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING account_id
            """, (artist_id, group_id, platform, account_identifier, url, is_active))
            account_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added account for artist_id {artist_id}/group_id {group_id} on {platform}")
            return account_id
    except psycopg2.IntegrityError:
        logger.warning(f"Account for artist_id {artist_id}/group_id {group_id} on {platform} already exists.")
        return None
    except Exception as e:
        logger.error(f"Failed to add account for artist_id {artist_id}/group_id {group_id} on {platform}: {e}")
        return None

def add_platform_metric(account_id: int, platform: str, metric_type: str, 
                       value: int, collected_at: Optional[datetime] = None) -> bool:
    """Add a platform metric."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if collected_at is None:
                collected_at = datetime.now()
            cursor.execute("""
                INSERT INTO platform_metrics (account_id, platform, metric_type, value, collected_at) 
                VALUES (%s, %s, %s, %s, %s)
            """, (account_id, platform, metric_type, value, collected_at))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to add metric for account_id {account_id}: {e}")
        return False

def add_album(name: str, artist_id: Optional[int] = None, group_id: Optional[int] = None,
              release_date: Optional[str] = None, album_type: str = 'album') -> Optional[int]:
    """Add an album and return the album_id."""
    # Enforce the check constraint logic in the function
    if (artist_id is None and group_id is None) or (artist_id is not None and group_id is not None):
        logger.error("Album must have either an artist_id or a group_id, but not both.")
        return None

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO albums (name, artist_id, group_id, release_date, album_type) 
                VALUES (%s, %s, %s, %s, %s) RETURNING album_id
            """, (name, artist_id, group_id, release_date, album_type))
            album_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added album: {name}")
            return album_id
    except psycopg2.IntegrityError as e:
        logger.warning(f"Album '{name}' likely already exists for this artist/group. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to add album '{name}': {e}")
        return None

def add_event(name: str, event_type: str, event_date: str, 
              artist_id: Optional[int] = None, group_id: Optional[int] = None, 
              company_id: Optional[int] = None, description: Optional[str] = None) -> Optional[int]:
    """Add an event and return the event_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (name, event_type, event_date, artist_id, group_id, company_id, description) 
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING event_id
            """, (name, event_type, event_date, artist_id, group_id, company_id, description))
            event_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added event: {name}")
            return event_id
    except Exception as e:
        logger.error(f"Failed to add event '{name}': {e}")
        return None

# --- Data retrieval functions (Enhanced with PostgreSQL features) --- #

def get_companies() -> List[Dict[str, Any]]:
    """Get all companies."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('SELECT * FROM companies ORDER BY name')
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get companies: {e}")
        return []

def get_listed_companies() -> List[Dict[str, Any]]:
    """Get main listed companies (parent companies only)."""
    # To-Do: 이 목록에 회사 이름을 추가하면 메인 페이지 필터에 나타납니다.
    listed_companies = ["HYBE", "JYP", "YG", "FNC", "SM", "CJ_ENM", "KAKAO"]
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            placeholders = ', '.join(['%s'] * len(listed_companies))
            cursor.execute(f"""
                SELECT * FROM companies 
                WHERE name IN ({placeholders}) 
                AND parent_company_id IS NULL
                ORDER BY name
            """, listed_companies)
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get listed companies: {e}")
        return []

def get_company_by_name(company_name: str) -> Optional[Dict[str, Any]]:
    """Get company by name."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM companies WHERE name = %s", (company_name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Failed to get company '{company_name}': {e}")
        return None

def get_subsidiaries(company_id: int) -> List[Dict[str, Any]]:
    """Get all subsidiaries of a company using recursive CTE."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                WITH RECURSIVE company_tree AS (
                    -- Direct subsidiaries
                    SELECT company_id, name, parent_company_id, 1 as level
                    FROM companies
                    WHERE parent_company_id = %s
                    
                    UNION ALL
                    
                    -- Recursive subsidiaries
                    SELECT c.company_id, c.name, c.parent_company_id, ct.level + 1
                    FROM companies c
                    JOIN company_tree ct ON c.parent_company_id = ct.company_id
                )
                SELECT * FROM company_tree ORDER BY level, name
            """, (company_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get subsidiaries for company_id {company_id}: {e}")
        return []

def get_groups_by_company_id(company_id: int) -> List[Dict[str, Any]]:
    """Get groups by company ID."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT group_id, name FROM groups WHERE company_id = %s ORDER BY name", (company_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get groups for company_id {company_id}: {e}")
        return []

def get_artists_by_group_id(group_id: int) -> List[Dict[str, Any]]:
    """Get artists by group ID."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT artist_id, name FROM artists WHERE group_id = %s ORDER BY name", (group_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get artists for group_id {group_id}: {e}")
        return []

def get_groups_and_artists_in_hierarchy(company_id: int) -> List[Dict[str, Any]]:
    """Get hierarchical view of groups and artists for a company and its subsidiaries."""
    hierarchy = []
    
    try:
        # Main company groups
        main_groups = get_groups_by_company_id(company_id)
        for group in main_groups:
            group_info = {
                'group_id': group['group_id'],
                'group_name': group['name'],
                'artists': get_artists_by_group_id(group['group_id'])
            }
            hierarchy.append(group_info)
        
        # Subsidiary company groups
        subsidiaries = get_subsidiaries(company_id)
        for sub_company in subsidiaries:
            sub_groups = get_groups_by_company_id(sub_company['company_id'])
            for group in sub_groups:
                group_info = {
                    'group_id': group['group_id'],
                    'group_name': f"{sub_company['name']} - {group['name']}",
                    'artists': get_artists_by_group_id(group['group_id'])
                }
                hierarchy.append(group_info)
                
        return hierarchy
    except Exception as e:
        logger.error(f"Failed to get hierarchy for company_id {company_id}: {e}")
        return []

def get_groups(company_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get groups, optionally filtered by company."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if company_id:
                cursor.execute('SELECT * FROM groups WHERE company_id = %s ORDER BY name', (company_id,))
            else:
                cursor.execute('SELECT * FROM groups ORDER BY name')
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get groups: {e}")
        return []

def get_artists(group_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get artists, optionally filtered by group."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if group_id:
                cursor.execute('SELECT * FROM artists WHERE group_id = %s ORDER BY name', (group_id,))
            else:
                cursor.execute('SELECT * FROM artists ORDER BY name')
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get artists: {e}")
        return []

def get_artists_by_company(company_id: int) -> List[Dict[str, Any]]:
    """Get artists by company."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT a.artist_id, a.name, g.name as group_name
                FROM artists a
                JOIN groups g ON a.group_id = g.group_id
                WHERE g.company_id = %s
                ORDER BY g.name, a.name
            """, (company_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get artists for company_id {company_id}: {e}")
        return []

def get_all_artists_with_details() -> List[Dict[str, Any]]:
    """Get all artists with full details using the PostgreSQL view."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM v_artists_full_details ORDER BY company_name, group_name, artist_name")
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get all artists with details: {e}")
        return []

def get_all_artist_accounts() -> List[Dict[str, Any]]:
    """Get all artist accounts."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('SELECT * FROM artist_accounts WHERE is_active = true')
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get artist accounts: {e}")
        return []

def get_latest_metric(artist_id: int, platform: str) -> Dict[str, Any]:
    """Get latest metrics for an artist on a specific platform."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT pm.metric_type, pm.value
                FROM platform_metrics pm
                JOIN artist_accounts aa ON pm.account_id = aa.account_id
                WHERE aa.artist_id = %s AND pm.platform = %s
                ORDER BY pm.collected_at DESC
                LIMIT 10
            """, (artist_id, platform))
            rows = cursor.fetchall()
            return {row['metric_type']: row['value'] for row in rows}
    except Exception as e:
        logger.error(f"Failed to get latest metric for artist_id {artist_id}, platform {platform}: {e}")
        return {}

def get_latest_metric_for_group(group_id: int, platform: str) -> Dict[str, Any]:
    """Get latest metrics for a group on a specific platform."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT pm.metric_type, pm.value
                FROM platform_metrics pm
                JOIN artist_accounts aa ON pm.account_id = aa.account_id
                WHERE aa.group_id = %s AND pm.platform = %s
                ORDER BY pm.collected_at DESC
                LIMIT 10
            """, (group_id, platform))
            rows = cursor.fetchall()
            return {row['metric_type']: row['value'] for row in rows}
    except Exception as e:
        logger.error(f"Failed to get latest metric for group_id {group_id}, platform {platform}: {e}")
        return {}

def get_all_groups_with_details() -> List[Dict[str, Any]]:
    """Get all groups with company details."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT 
                    g.group_id,
                    g.name AS group_name,
                    g.debut_date,
                    c.name AS company_name,
                    c.company_id
                FROM groups g
                JOIN companies c ON g.company_id = c.company_id
                ORDER BY g.name
            """)
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get groups with details: {e}")
        return []

def get_all_metrics_for_artist(artist_id: int) -> List[Dict[str, Any]]:
    """Get all metrics for an artist across all platforms."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT pm.platform, pm.metric_type, pm.value, pm.collected_at
                FROM platform_metrics pm
                JOIN artist_accounts aa ON pm.account_id = aa.account_id
                WHERE aa.artist_id = %s 
                ORDER BY pm.collected_at DESC
            """, (artist_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get metrics for artist_id {artist_id}: {e}")
        return []

def get_all_metrics_for_group(group_id: int) -> List[Dict[str, Any]]:
    """Get all metrics for a group across all platforms."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT pm.platform, pm.metric_type, pm.value, pm.collected_at
                FROM platform_metrics pm
                JOIN artist_accounts aa ON pm.account_id = aa.account_id
                WHERE aa.group_id = %s
                ORDER BY pm.collected_at DESC
            """, (group_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get metrics for group_id {group_id}: {e}")
        return []

def get_events_for_artist(artist_id: int) -> List[Dict[str, Any]]:
    """Get events for an artist, their group, and their company."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get artist details first
            cursor.execute("""
                SELECT a.group_id, g.company_id
                FROM artists a
                LEFT JOIN groups g ON a.group_id = g.group_id
                WHERE a.artist_id = %s
            """, (artist_id,))
            
            artist_details = cursor.fetchone()
            if not artist_details:
                return []
            
            group_id = artist_details['group_id']
            company_id = artist_details['company_id']
            
            # Query events
            cursor.execute("""
                SELECT name, event_type, event_date, description
                FROM events
                WHERE artist_id = %s 
                OR (group_id = %s AND group_id IS NOT NULL) 
                OR (company_id = %s AND company_id IS NOT NULL)
                ORDER BY event_date DESC
            """, (artist_id, group_id, company_id))
            
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get events for artist_id {artist_id}: {e}")
        return []

def get_events_for_group(group_id: int) -> List[Dict[str, Any]]:
    """Get events for a specific group."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT name, event_type, event_date, description
                FROM events
                WHERE group_id = %s
                ORDER BY event_date DESC
            """, (group_id,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get events for group_id {group_id}: {e}")
        return []

def get_events_by_date_range(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Get events within a specific date range."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT * FROM events
                WHERE event_date BETWEEN %s AND %s
                ORDER BY event_date DESC
            """, (start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get events for date range {start_date} to {end_date}: {e}")
        return []

# --- API Quota Management (Enhanced) --- #

def get_api_usage(platform: str) -> int:
    """Get current API usage for platform."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute("SELECT usage_count FROM api_quotas WHERE platform = %s AND date = %s", (platform, today))
            row = cursor.fetchone()
            
            if row:
                return row[0]
            else:
                # Create entry for today
                cursor.execute("""
                    INSERT INTO api_quotas (platform, date, usage_count) VALUES (%s, %s, 0)
                    ON CONFLICT (platform, date) DO NOTHING
                """, (platform, today))
                conn.commit()
                return 0
    except Exception as e:
        logger.error(f"Failed to get API usage for {platform}: {e}")
        return 0

def increment_api_usage(platform: str) -> bool:
    """Increment API usage for platform."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute("""
                INSERT INTO api_quotas (platform, date, usage_count) VALUES (%s, %s, 1)
                ON CONFLICT (platform, date) DO UPDATE SET usage_count = api_quotas.usage_count + 1
            """, (platform, today))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to increment API usage for {platform}: {e}")
        return False

def check_api_limit(platform: str, daily_limit: int = 10000) -> bool:
    """Check if API usage is within limit."""
    current_usage = get_api_usage(platform)
    return current_usage < daily_limit

# --- Utility Functions --- #

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            stats = {}
            
            # Table row counts
            tables = ['companies', 'groups', 'artists', 'artist_accounts', 'platform_metrics', 'events', 'albums']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()['count']
            
            # Latest metric collection date
            cursor.execute("SELECT MAX(collected_at) as latest_collection FROM platform_metrics")
            row = cursor.fetchone()
            stats['latest_collection'] = row['latest_collection']
            
            return stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}

# --- Analytics Views Functions (New) --- #

def get_growth_summary(company_filter: Optional[str] = None, 
                      growth_category: Optional[str] = None,
                      influence_tier: Optional[str] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get growth summary data with optional filters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build dynamic query with filters
            where_conditions = []
            params = []
            
            if company_filter:
                where_conditions.append("company_name = %s")
                params.append(company_filter)
            
            if growth_category:
                where_conditions.append("growth_category = %s")
                params.append(growth_category)
                
            if influence_tier:
                where_conditions.append("influence_tier = %s")
                params.append(influence_tier)
            
            where_clause = " AND ".join(where_conditions)
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
                SELECT * FROM v_growth_summary 
                {where_clause}
                ORDER BY total_influence_score DESC
                {limit_clause}
            """
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get growth summary: {e}")
        return []

def get_top_performers(company_filter: Optional[str] = None,
                      performance_badge: Optional[str] = None,
                      momentum_status: Optional[str] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
    """Get top performers with optional filters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = []
            params = []
            
            if company_filter:
                where_conditions.append("company_name = %s")
                params.append(company_filter)
                
            if performance_badge:
                where_conditions.append("performance_badge = %s")
                params.append(performance_badge)
                
            if momentum_status:
                where_conditions.append("momentum_status = %s")
                params.append(momentum_status)
            
            where_clause = " AND ".join(where_conditions)
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            query = f"""
                SELECT * FROM v_top_performers 
                {where_clause}
                ORDER BY overall_rank ASC
                LIMIT %s
            """
            
            params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get top performers: {e}")
        return []

def get_growth_alerts(severity_filter: Optional[str] = None,
                     alert_category: Optional[str] = None,
                     company_filter: Optional[str] = None,
                     hours_limit: int = 168) -> List[Dict[str, Any]]:
    """Get active growth alerts with optional filters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = ["hours_since_detected <= %s"]
            params = [hours_limit]
            
            if severity_filter:
                where_conditions.append("severity_level = %s")
                params.append(severity_filter)
                
            if alert_category:
                where_conditions.append("alert_category = %s") 
                params.append(alert_category)
                
            if company_filter:
                where_conditions.append("company_name = %s")
                params.append(company_filter)
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT * FROM v_growth_alerts_active 
                WHERE {where_clause}
                ORDER BY priority_score DESC, alert_timestamp DESC
                LIMIT 100
            """
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get growth alerts: {e}")
        return []

def get_analytics_dashboard_summary() -> Dict[str, Any]:
    """Get summary statistics for analytics dashboard."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get various summary statistics
            stats = {}
            
            # Growth summary stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_accounts,
                    COUNT(CASE WHEN avg_growth_rate > 0 THEN 1 END) as growing_accounts,
                    COUNT(CASE WHEN influence_tier IN ('S급_메가스타', 'A급_톱스타') THEN 1 END) as top_tier_accounts,
                    ROUND(AVG(avg_growth_rate), 2) as avg_growth_rate,
                    MAX(total_influence_score) as max_influence_score
                FROM v_growth_summary
            """)
            growth_stats = cursor.fetchone()
            stats.update(dict(growth_stats))
            
            # Top performers stats
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN performance_badge = 'TOP10' THEN 1 END) as top_10_performers,
                    COUNT(CASE WHEN momentum_status = '폭발적성장' THEN 1 END) as explosive_growth_count,
                    ROUND(AVG(performance_score), 1) as avg_performance_score
                FROM v_top_performers
            """)
            performer_stats = cursor.fetchone()
            stats.update(dict(performer_stats))
            
            # Active alerts stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_active_alerts,
                    COUNT(CASE WHEN severity_level = 'critical' THEN 1 END) as critical_alerts,
                    COUNT(CASE WHEN alert_category = '성장기회' THEN 1 END) as growth_opportunities,
                    COUNT(CASE WHEN alert_category = '성과우려' THEN 1 END) as performance_concerns
                FROM v_growth_alerts_active
                WHERE hours_since_detected <= 168
            """)
            alert_stats = cursor.fetchone()
            stats.update(dict(alert_stats))
            
            # Company performance distribution
            cursor.execute("""
                SELECT 
                    company_name,
                    COUNT(*) as artist_count,
                    ROUND(AVG(total_influence_score), 0) as avg_influence,
                    ROUND(AVG(avg_growth_rate), 2) as avg_growth
                FROM v_growth_summary
                WHERE company_name IS NOT NULL
                GROUP BY company_name
                ORDER BY avg_influence DESC
                LIMIT 10
            """)
            stats['company_rankings'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get analytics dashboard summary: {e}")
        return {}

def get_platform_performance_comparison() -> List[Dict[str, Any]]:
    """Get platform-wise performance comparison."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    'YouTube' as platform,
                    COUNT(CASE WHEN youtube_subscribers > 0 THEN 1 END) as active_accounts,
                    SUM(COALESCE(youtube_subscribers, 0)) as total_followers,
                    ROUND(AVG(CASE WHEN youtube_subs_growth IS NOT NULL THEN youtube_subs_growth END), 2) as avg_growth_rate,
                    MAX(youtube_subscribers) as max_followers
                FROM v_growth_summary
                
                UNION ALL
                
                SELECT 
                    'Spotify' as platform,
                    COUNT(CASE WHEN spotify_monthly_listeners > 0 THEN 1 END) as active_accounts,
                    SUM(COALESCE(spotify_monthly_listeners, 0)) as total_followers,
                    ROUND(AVG(CASE WHEN spotify_listeners_growth IS NOT NULL THEN spotify_listeners_growth END), 2) as avg_growth_rate,
                    MAX(spotify_monthly_listeners) as max_followers
                FROM v_growth_summary
                
                UNION ALL
                
                SELECT 
                    'Twitter' as platform,
                    COUNT(CASE WHEN twitter_followers > 0 THEN 1 END) as active_accounts,
                    SUM(COALESCE(twitter_followers, 0)) as total_followers,
                    ROUND(AVG(CASE WHEN twitter_followers_growth IS NOT NULL THEN twitter_followers_growth END), 2) as avg_growth_rate,
                    MAX(twitter_followers) as max_followers
                FROM v_growth_summary
                
                UNION ALL
                
                SELECT 
                    'Instagram' as platform,
                    COUNT(CASE WHEN instagram_followers > 0 THEN 1 END) as active_accounts,
                    SUM(COALESCE(instagram_followers, 0)) as total_followers,
                    ROUND(AVG(CASE WHEN instagram_followers_growth IS NOT NULL THEN instagram_followers_growth END), 2) as avg_growth_rate,
                    MAX(instagram_followers) as max_followers
                FROM v_growth_summary
                
                UNION ALL
                
                SELECT 
                    'TikTok' as platform,
                    COUNT(CASE WHEN tiktok_followers > 0 THEN 1 END) as active_accounts,
                    SUM(COALESCE(tiktok_followers, 0)) as total_followers,
                    ROUND(AVG(CASE WHEN tiktok_followers_growth IS NOT NULL THEN tiktok_followers_growth END), 2) as avg_growth_rate,
                    MAX(tiktok_followers) as max_followers
                FROM v_growth_summary
                
                ORDER BY total_followers DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get platform performance comparison: {e}")
        return []

def get_growth_trends_by_company(company_name: str, days_back: int = 30) -> Dict[str, Any]:
    """Get growth trends for a specific company."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get company artists and their recent growth
            cursor.execute("""
                SELECT 
                    artist_name,
                    group_name,
                    growth_category,
                    influence_tier,
                    avg_growth_rate,
                    total_influence_score,
                    platform_diversity_count,
                    last_updated
                FROM v_growth_summary 
                WHERE company_name = %s
                ORDER BY total_influence_score DESC
            """, (company_name,))
            
            artists_data = [dict(row) for row in cursor.fetchall()]
            
            # Get company alerts
            cursor.execute("""
                SELECT 
                    alert_type,
                    severity_level,
                    COUNT(*) as alert_count
                FROM v_growth_alerts_active
                WHERE company_name = %s
                  AND hours_since_detected <= %s
                GROUP BY alert_type, severity_level
                ORDER BY alert_count DESC
            """, (company_name, days_back * 24))
            
            alerts_summary = [dict(row) for row in cursor.fetchall()]
            
            # Company performance summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_artists,
                    ROUND(AVG(avg_growth_rate), 2) as company_avg_growth,
                    SUM(total_influence_score) as total_company_influence,
                    COUNT(CASE WHEN growth_category IN ('급성장', '고성장') THEN 1 END) as high_growth_artists,
                    COUNT(CASE WHEN influence_tier IN ('S급_메가스타', 'A급_톱스타') THEN 1 END) as top_tier_artists
                FROM v_growth_summary
                WHERE company_name = %s
            """, (company_name,))
            
            company_summary = dict(cursor.fetchone())
            
            return {
                'company_name': company_name,
                'artists': artists_data,
                'alerts_summary': alerts_summary,
                'company_summary': company_summary
            }
            
    except Exception as e:
        logger.error(f"Failed to get growth trends for company {company_name}: {e}")
        return {}

def acknowledge_alert(alert_id: str, acknowledged_by: str) -> bool:
    """Mark an alert as acknowledged (future enhancement)."""
    # This function is prepared for future implementation when we add 
    # alert management functionality to the growth_alerts table
    logger.info(f"Alert acknowledgment feature not yet implemented. Alert ID: {alert_id}, User: {acknowledged_by}")
    return False

def get_view_refresh_status() -> Dict[str, Any]:
    """Get information about when analytics views were last updated."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Check latest data timestamps from base tables
            cursor.execute("""
                SELECT 
                    'platform_metrics' as table_name,
                    COUNT(*) as record_count,
                    MAX(collected_at) as latest_data,
                    MIN(collected_at) as earliest_data
                FROM platform_metrics
                WHERE collected_at >= CURRENT_DATE - INTERVAL '7 days'
                
                UNION ALL
                
                SELECT 
                    'artist_accounts' as table_name,
                    COUNT(*) as record_count,
                    MAX(updated_at) as latest_data,
                    MIN(created_at) as earliest_data
                FROM artist_accounts
                WHERE is_active = TRUE
            """)
            
            base_data_status = [dict(row) for row in cursor.fetchall()]
            
            # Get view data freshness
            cursor.execute("""
                SELECT 
                    'v_growth_summary' as view_name,
                    COUNT(*) as record_count,
                    MAX(last_updated) as latest_data_timestamp
                FROM v_growth_summary
                
                UNION ALL
                
                SELECT 
                    'v_growth_alerts_active' as view_name,
                    COUNT(*) as record_count,
                    MAX(alert_timestamp) as latest_data_timestamp
                FROM v_growth_alerts_active
            """)
            
            view_status = [dict(row) for row in cursor.fetchall()]
            
            return {
                'base_tables': base_data_status,
                'views': view_status,
                'refresh_timestamp': datetime.now()
            }
            
    except Exception as e:
        logger.error(f"Failed to get view refresh status: {e}")
        return {}

# =============================================================================
# GROWTH ANALYTICS CRUD FUNCTIONS (Phase 2 Implementation)
# =============================================================================

# --- Growth Rates Table Functions --- #

def add_growth_rate(account_id: int, platform: str, metric_type: str, 
                   current_value: int, previous_value: Optional[int] = None,
                   daily_growth_rate: Optional[float] = None,
                   weekly_growth_rate: Optional[float] = None,
                   monthly_growth_rate: Optional[float] = None,
                   quarterly_growth_rate: Optional[float] = None,
                   yoy_growth_rate: Optional[float] = None,
                   calculation_method: str = 'rolling_average',
                   data_quality_score: float = 1.0,
                   calculated_at: Optional[datetime] = None) -> Optional[int]:
    """Add a growth rate record and return the growth_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if calculated_at is None:
                calculated_at = datetime.now()
                
            cursor.execute("""
                INSERT INTO growth_rates (
                    account_id, platform, metric_type, current_value, previous_value,
                    daily_growth_rate, weekly_growth_rate, monthly_growth_rate, 
                    quarterly_growth_rate, yoy_growth_rate, calculation_method, 
                    data_quality_score, calculated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING growth_id
            """, (account_id, platform, metric_type, current_value, previous_value,
                  daily_growth_rate, weekly_growth_rate, monthly_growth_rate,
                  quarterly_growth_rate, yoy_growth_rate, calculation_method,
                  data_quality_score, calculated_at))
            
            growth_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added growth rate record for account_id {account_id}")
            return growth_id
            
    except psycopg2.IntegrityError as e:
        logger.error(f"Integrity error adding growth rate: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to add growth rate: {e}")
        return None

def get_growth_rates_by_account(account_id: int, platform: Optional[str] = None,
                               metric_type: Optional[str] = None,
                               days_back: int = 30) -> List[Dict[str, Any]]:
    """Get growth rates for a specific account with optional filters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = ["account_id = %s", "calculated_at >= %s"]
            params = [account_id, datetime.now() - timedelta(days=days_back)]
            
            if platform:
                where_conditions.append("platform = %s")
                params.append(platform)
                
            if metric_type:
                where_conditions.append("metric_type = %s")
                params.append(metric_type)
            
            where_clause = " AND ".join(where_conditions)
            
            cursor.execute(f"""
                SELECT * FROM growth_rates 
                WHERE {where_clause}
                ORDER BY calculated_at DESC
            """, params)
            
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get growth rates for account {account_id}: {e}")
        return []

def update_growth_rate(growth_id: int, **kwargs) -> bool:
    """Update a growth rate record with provided fields."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query
            update_fields = []
            params = []
            
            allowed_fields = {
                'daily_growth_rate', 'weekly_growth_rate', 'monthly_growth_rate',
                'quarterly_growth_rate', 'yoy_growth_rate', 'current_value',
                'previous_value', 'calculation_method', 'data_quality_score'
            }
            
            for field, value in kwargs.items():
                if field in allowed_fields:
                    update_fields.append(f"{field} = %s")
                    params.append(value)
            
            if not update_fields:
                logger.warning("No valid fields provided for growth rate update")
                return False
            
            params.append(growth_id)
            update_clause = ", ".join(update_fields)
            
            cursor.execute(f"""
                UPDATE growth_rates 
                SET {update_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE growth_id = %s
            """, params)
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Updated growth rate {growth_id}")
                return True
            else:
                logger.warning(f"No growth rate found with ID {growth_id}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to update growth rate {growth_id}: {e}")
        return False

def delete_growth_rate(growth_id: int) -> bool:
    """Delete a growth rate record."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM growth_rates WHERE growth_id = %s", (growth_id,))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Deleted growth rate {growth_id}")
                return True
            else:
                logger.warning(f"No growth rate found with ID {growth_id}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to delete growth rate {growth_id}: {e}")
        return False

def bulk_insert_growth_rates(growth_data: List[Dict[str, Any]]) -> int:
    """Bulk insert growth rate records. Returns number of records inserted."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare bulk insert data
            insert_data = []
            for data in growth_data:
                insert_data.append((
                    data['account_id'], data['platform'], data['metric_type'],
                    data['current_value'], data.get('previous_value'),
                    data.get('daily_growth_rate'), data.get('weekly_growth_rate'),
                    data.get('monthly_growth_rate'), data.get('quarterly_growth_rate'),
                    data.get('yoy_growth_rate'), data.get('calculation_method', 'rolling_average'),
                    data.get('data_quality_score', 1.0), data.get('calculated_at', datetime.now())
                ))
            
            cursor.executemany("""
                INSERT INTO growth_rates (
                    account_id, platform, metric_type, current_value, previous_value,
                    daily_growth_rate, weekly_growth_rate, monthly_growth_rate,
                    quarterly_growth_rate, yoy_growth_rate, calculation_method,
                    data_quality_score, calculated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, insert_data)
            
            inserted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Bulk inserted {inserted_count} growth rate records")
            return inserted_count
            
    except Exception as e:
        logger.error(f"Failed to bulk insert growth rates: {e}")
        return 0

# --- Growth Predictions Table Functions --- #

def add_growth_prediction(account_id: int, platform: str, metric_type: str,
                         prediction_method: str, prediction_date: str, 
                         prediction_horizon: int, predicted_value: int,
                         confidence_lower: int, confidence_upper: int,
                         model_accuracy: Optional[float] = None,
                         mae: Optional[float] = None, rmse: Optional[float] = None,
                         mape: Optional[float] = None, training_data_points: Optional[int] = None,
                         training_period_days: Optional[int] = None,
                         model_parameters: Optional[Dict] = None) -> Optional[int]:
    """Add a growth prediction record and return the prediction_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Convert model_parameters to JSON
            model_params_json = json.dumps(model_parameters) if model_parameters else None
            
            cursor.execute("""
                INSERT INTO growth_predictions (
                    account_id, platform, metric_type, prediction_method, 
                    prediction_date, prediction_horizon, predicted_value,
                    confidence_lower, confidence_upper, model_accuracy, mae, rmse, mape,
                    training_data_points, training_period_days, model_parameters
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING prediction_id
            """, (account_id, platform, metric_type, prediction_method, prediction_date,
                  prediction_horizon, predicted_value, confidence_lower, confidence_upper,
                  model_accuracy, mae, rmse, mape, training_data_points, 
                  training_period_days, model_params_json))
            
            prediction_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added growth prediction for account_id {account_id}")
            return prediction_id
            
    except psycopg2.IntegrityError as e:
        if "uq_predictions_unique" in str(e):
            logger.warning(f"Duplicate prediction detected for account {account_id}")
        else:
            logger.error(f"Integrity error adding prediction: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to add growth prediction: {e}")
        return None

def get_predictions_by_account(account_id: int, platform: Optional[str] = None,
                              prediction_horizon: Optional[int] = None,
                              active_only: bool = True) -> List[Dict[str, Any]]:
    """Get growth predictions for a specific account with optional filters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = ["account_id = %s"]
            params = [account_id]
            
            if platform:
                where_conditions.append("platform = %s")
                params.append(platform)
                
            if prediction_horizon:
                where_conditions.append("prediction_horizon = %s")
                params.append(prediction_horizon)
                
            if active_only:
                where_conditions.append("is_active = TRUE")
                where_conditions.append("prediction_date >= CURRENT_DATE")
            
            where_clause = " AND ".join(where_conditions)
            
            cursor.execute(f"""
                SELECT * FROM growth_predictions 
                WHERE {where_clause}
                ORDER BY prediction_date ASC, created_at DESC
            """, params)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON model_parameters back to dict
            for result in results:
                if result.get('model_parameters'):
                    try:
                        result['model_parameters'] = json.loads(result['model_parameters'])
                    except json.JSONDecodeError:
                        result['model_parameters'] = {}
                        
            return results
            
    except Exception as e:
        logger.error(f"Failed to get predictions for account {account_id}: {e}")
        return []

def update_prediction_accuracy(prediction_id: int, actual_value: int) -> bool:
    """Update prediction accuracy after actual value is available."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get the prediction details
            cursor.execute("""
                SELECT predicted_value, confidence_lower, confidence_upper 
                FROM growth_predictions 
                WHERE prediction_id = %s
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Prediction {prediction_id} not found")
                return False
            
            predicted_value, conf_lower, conf_upper = result
            
            # Calculate accuracy metrics
            absolute_error = abs(actual_value - predicted_value)
            percentage_error = (absolute_error / max(actual_value, 1)) * 100
            
            # Check if actual value is within confidence interval
            within_confidence = conf_lower <= actual_value <= conf_upper
            
            # Update the prediction with accuracy information
            cursor.execute("""
                UPDATE growth_predictions 
                SET mae = %s, mape = %s, is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE prediction_id = %s
            """, (absolute_error, percentage_error, prediction_id))
            
            conn.commit()
            logger.info(f"Updated prediction accuracy for {prediction_id}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to update prediction accuracy: {e}")
        return False

def deactivate_old_predictions(days_old: int = 30) -> int:
    """Deactivate predictions that are older than specified days."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE growth_predictions 
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE prediction_date < CURRENT_DATE - INTERVAL '%s days'
                AND is_active = TRUE
            """, (days_old,))
            
            deactivated_count = cursor.rowcount
            conn.commit()
            logger.info(f"Deactivated {deactivated_count} old predictions")
            return deactivated_count
            
    except Exception as e:
        logger.error(f"Failed to deactivate old predictions: {e}")
        return 0

# --- Growth Alerts Table Functions --- #

def create_growth_alert(account_id: int, platform: str, metric_type: str,
                       alert_type: str, severity_level: str, alert_message: str,
                       trigger_condition: Optional[str] = None, trigger_value: Optional[float] = None,
                       threshold_value: Optional[float] = None, current_value: Optional[int] = None,
                       baseline_value: Optional[int] = None, alert_category: str = 'performance',
                       alert_summary: Optional[str] = None, related_event_id: Optional[int] = None,
                       confidence_score: float = 1.0, expires_at: Optional[datetime] = None) -> Optional[int]:
    """Create a new growth alert and return the alert_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO growth_alerts (
                    account_id, platform, metric_type, alert_type, severity_level,
                    alert_category, alert_message, alert_summary, trigger_condition,
                    trigger_value, threshold_value, current_value, baseline_value,
                    related_event_id, confidence_score, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING alert_id
            """, (account_id, platform, metric_type, alert_type, severity_level,
                  alert_category, alert_message, alert_summary, trigger_condition,
                  trigger_value, threshold_value, current_value, baseline_value,
                  related_event_id, confidence_score, expires_at))
            
            alert_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Created growth alert {alert_id} for account {account_id}")
            return alert_id
            
    except Exception as e:
        logger.error(f"Failed to create growth alert: {e}")
        return None

def get_active_alerts(account_id: Optional[int] = None, severity_level: Optional[str] = None,
                     alert_category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get active growth alerts with optional filters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = ["is_resolved = FALSE"]
            params = []
            
            # Add expiration filter
            where_conditions.append("(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)")
            
            if account_id:
                where_conditions.append("account_id = %s")
                params.append(account_id)
                
            if severity_level:
                where_conditions.append("severity_level = %s")
                params.append(severity_level)
                
            if alert_category:
                where_conditions.append("alert_category = %s")
                params.append(alert_category)
            
            where_clause = " AND ".join(where_conditions)
            params.append(limit)
            
            cursor.execute(f"""
                SELECT ga.*, aa.artist_id, a.name as artist_name, g.name as group_name, c.name as company_name
                FROM growth_alerts ga
                LEFT JOIN artist_accounts aa ON ga.account_id = aa.account_id
                LEFT JOIN artists a ON aa.artist_id = a.artist_id
                LEFT JOIN groups g ON aa.group_id = g.group_id OR a.group_id = g.group_id
                LEFT JOIN companies c ON g.company_id = c.company_id
                WHERE {where_clause}
                ORDER BY severity_level DESC, created_at DESC
                LIMIT %s
            """, params)
            
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        return []

def acknowledge_growth_alert(alert_id: int, acknowledged_by: str) -> bool:
    """Acknowledge a growth alert."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE growth_alerts 
                SET is_acknowledged = TRUE, acknowledged_by = %s, 
                    acknowledged_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE alert_id = %s AND is_resolved = FALSE
            """, (acknowledged_by, alert_id))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Acknowledged alert {alert_id} by {acknowledged_by}")
                return True
            else:
                logger.warning(f"Alert {alert_id} not found or already resolved")
                return False
                
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        return False

def resolve_growth_alert(alert_id: int, resolution_note: Optional[str] = None) -> bool:
    """Resolve a growth alert."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE growth_alerts 
                SET is_resolved = TRUE, resolved_at = CURRENT_TIMESTAMP, 
                    resolution_note = %s, updated_at = CURRENT_TIMESTAMP
                WHERE alert_id = %s AND is_resolved = FALSE
            """, (resolution_note, alert_id))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Resolved alert {alert_id}")
                return True
            else:
                logger.warning(f"Alert {alert_id} not found or already resolved")
                return False
                
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        return False

def get_alert_statistics(days_back: int = 30) -> Dict[str, Any]:
    """Get alert statistics for the specified time period."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            stats = {}
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Overall alert statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(CASE WHEN is_resolved = TRUE THEN 1 END) as resolved_alerts,
                    COUNT(CASE WHEN is_acknowledged = TRUE THEN 1 END) as acknowledged_alerts,
                    COUNT(CASE WHEN severity_level = 'red' THEN 1 END) as critical_alerts,
                    COUNT(CASE WHEN severity_level = 'yellow' THEN 1 END) as warning_alerts,
                    COUNT(CASE WHEN severity_level = 'green' THEN 1 END) as info_alerts,
                    ROUND(AVG(confidence_score), 3) as avg_confidence_score
                FROM growth_alerts 
                WHERE created_at >= %s
            """, (cutoff_date,))
            
            overall_stats = cursor.fetchone()
            stats.update(dict(overall_stats))
            
            # Alert type distribution
            cursor.execute("""
                SELECT alert_type, COUNT(*) as count
                FROM growth_alerts 
                WHERE created_at >= %s
                GROUP BY alert_type
                ORDER BY count DESC
            """, (cutoff_date,))
            
            stats['alert_type_distribution'] = [dict(row) for row in cursor.fetchall()]
            
            # Platform distribution
            cursor.execute("""
                SELECT platform, COUNT(*) as count
                FROM growth_alerts 
                WHERE created_at >= %s
                GROUP BY platform
                ORDER BY count DESC
            """, (cutoff_date,))
            
            stats['platform_distribution'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {e}")
        return {}

def cleanup_expired_alerts() -> int:
    """Remove expired alerts from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM growth_alerts 
                WHERE expires_at IS NOT NULL 
                AND expires_at < CURRENT_TIMESTAMP
                AND is_resolved = TRUE
            """, )
            
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted_count} expired alerts")
            return deleted_count
            
    except Exception as e:
        logger.error(f"Failed to cleanup expired alerts: {e}")
        return 0

# --- Bulk Operations and Analytics --- #

def calculate_growth_rates_for_account(account_id: int, platform: str, metric_type: str) -> bool:
    """Calculate and store growth rates for a specific account/platform/metric combination."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get the latest metrics for calculation
            cursor.execute("""
                SELECT value, collected_at 
                FROM platform_metrics pm
                JOIN artist_accounts aa ON pm.account_id = aa.account_id
                WHERE aa.account_id = %s AND pm.platform = %s AND pm.metric_type = %s
                ORDER BY collected_at DESC
                LIMIT 100
            """, (account_id, platform, metric_type))
            
            metrics = cursor.fetchall()
            
            if len(metrics) < 2:
                logger.warning(f"Insufficient data for growth calculation: account {account_id}")
                return False
            
            current = metrics[0]
            previous = metrics[1]
            
            # Calculate growth rates
            if previous['value'] > 0:
                daily_growth = ((current['value'] - previous['value']) / previous['value']) * 100
            else:
                daily_growth = None
            
            # Calculate weekly growth (7 days back)
            weekly_growth = None
            if len(metrics) >= 7:
                week_ago = metrics[6]
                if week_ago['value'] > 0:
                    weekly_growth = ((current['value'] - week_ago['value']) / week_ago['value']) * 100
            
            # Calculate monthly growth (30 days back)
            monthly_growth = None
            if len(metrics) >= 30:
                month_ago = metrics[29]
                if month_ago['value'] > 0:
                    monthly_growth = ((current['value'] - month_ago['value']) / month_ago['value']) * 100
            
            # Store the calculated growth rates
            add_growth_rate(
                account_id=account_id,
                platform=platform,
                metric_type=metric_type,
                current_value=current['value'],
                previous_value=previous['value'],
                daily_growth_rate=daily_growth,
                weekly_growth_rate=weekly_growth,
                monthly_growth_rate=monthly_growth,
                calculation_method='rolling_average',
                calculated_at=current['collected_at']
            )
            
            logger.info(f"Calculated growth rates for account {account_id}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to calculate growth rates: {e}")
        return False

def get_growth_analytics_summary() -> Dict[str, Any]:
    """Get comprehensive summary of growth analytics data."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            summary = {}
            
            # Growth rates summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_growth_records,
                    COUNT(DISTINCT account_id) as unique_accounts,
                    ROUND(AVG(daily_growth_rate), 2) as avg_daily_growth,
                    ROUND(AVG(monthly_growth_rate), 2) as avg_monthly_growth,
                    MAX(calculated_at) as latest_calculation
                FROM growth_rates
                WHERE calculated_at >= CURRENT_DATE - INTERVAL '7 days'
            """)
            growth_summary = cursor.fetchone()
            summary['growth_rates'] = dict(growth_summary)
            
            # Predictions summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT account_id) as accounts_with_predictions,
                    COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_predictions,
                    ROUND(AVG(model_accuracy), 2) as avg_model_accuracy
                FROM growth_predictions
            """)
            predictions_summary = cursor.fetchone()
            summary['predictions'] = dict(predictions_summary)
            
            # Alerts summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(CASE WHEN is_resolved = FALSE THEN 1 END) as active_alerts,
                    COUNT(CASE WHEN severity_level = 'red' THEN 1 END) as critical_alerts,
                    ROUND(AVG(confidence_score), 3) as avg_confidence
                FROM growth_alerts
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
            """)
            alerts_summary = cursor.fetchone()
            summary['alerts'] = dict(alerts_summary)
            
            return summary
            
    except Exception as e:
        logger.error(f"Failed to get growth analytics summary: {e}")
        return {}

# =============================================================================
# KPI SYSTEM DATABASE FUNCTIONS
# =============================================================================

# --- KPI Definitions Management --- #

def add_kpi_definition(kpi_name: str, kpi_formula: Dict[str, Any], kpi_description: str,
                      kpi_category: str = 'custom', aggregation_method: str = 'weighted_average',
                      weighting_scheme: str = 'platform_based', platform_weights: Optional[Dict] = None,
                      time_window_days: int = 30, target_value: Optional[float] = None,
                      warning_threshold: float = 80.0, critical_threshold: float = 50.0,
                      normalization_factor: Optional[float] = None, min_data_points: int = 1,
                      created_by: Optional[str] = None) -> Optional[int]:
    """Add a new KPI definition and return the kpi_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Convert formula and weights to JSON
            formula_json = json.dumps(kpi_formula)
            weights_json = json.dumps(platform_weights) if platform_weights else None
            
            cursor.execute("""
                INSERT INTO kpi_definitions (
                    kpi_name, kpi_formula, kpi_description, kpi_category,
                    aggregation_method, weighting_scheme, platform_weights,
                    time_window_days, target_value, warning_threshold, critical_threshold,
                    normalization_factor, min_data_points, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING kpi_id
            """, (kpi_name, formula_json, kpi_description, kpi_category,
                  aggregation_method, weighting_scheme, weights_json,
                  time_window_days, target_value, warning_threshold, critical_threshold,
                  normalization_factor, min_data_points, created_by))
            
            kpi_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added KPI definition: {kpi_name}")
            return kpi_id
            
    except psycopg2.IntegrityError:
        logger.warning(f"KPI definition '{kpi_name}' already exists")
        return None
    except Exception as e:
        logger.error(f"Failed to add KPI definition '{kpi_name}': {e}")
        return None

def get_kpi_definitions(active_only: bool = True) -> List[Dict[str, Any]]:
    """Get all KPI definitions."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_clause = "WHERE is_active = TRUE" if active_only else ""
            cursor.execute(f"""
                SELECT * FROM kpi_definitions 
                {where_clause}
                ORDER BY kpi_name
            """)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                if result.get('kpi_formula'):
                    try:
                        if isinstance(result['kpi_formula'], str):
                            result['kpi_formula'] = json.loads(result['kpi_formula'])
                    except json.JSONDecodeError:
                        result['kpi_formula'] = {}
                
                if result.get('platform_weights'):
                    try:
                        if isinstance(result['platform_weights'], str):
                            result['platform_weights'] = json.loads(result['platform_weights'])
                    except json.JSONDecodeError:
                        result['platform_weights'] = {}
            
            return results
            
    except Exception as e:
        logger.error(f"Failed to get KPI definitions: {e}")
        return []

def get_kpi_definition_by_name(kpi_name: str) -> Optional[Dict[str, Any]]:
    """Get a specific KPI definition by name."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM kpi_definitions WHERE kpi_name = %s", (kpi_name,))
            
            result = cursor.fetchone()
            if result:
                result = dict(result)
                
                # Parse JSON fields
                if result.get('kpi_formula'):
                    try:
                        result['kpi_formula'] = json.loads(result['kpi_formula'])
                    except json.JSONDecodeError:
                        result['kpi_formula'] = {}
                
                if result.get('platform_weights'):
                    try:
                        result['platform_weights'] = json.loads(result['platform_weights'])
                    except json.JSONDecodeError:
                        result['platform_weights'] = {}
                
                return result
            return None
            
    except Exception as e:
        logger.error(f"Failed to get KPI definition '{kpi_name}': {e}")
        return None

def update_kpi_definition(kpi_id: int, **kwargs) -> bool:
    """Update a KPI definition with provided fields."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query
            update_fields = []
            params = []
            
            allowed_fields = {
                'kpi_name', 'kpi_formula', 'kpi_description', 'kpi_category',
                'aggregation_method', 'weighting_scheme', 'platform_weights',
                'time_window_days', 'target_value', 'warning_threshold', 'critical_threshold',
                'normalization_factor', 'min_data_points', 'is_active'
            }
            
            for field, value in kwargs.items():
                if field in allowed_fields:
                    if field in ['kpi_formula', 'platform_weights'] and isinstance(value, dict):
                        value = json.dumps(value)
                    update_fields.append(f"{field} = %s")
                    params.append(value)
            
            if not update_fields:
                logger.warning("No valid fields provided for KPI definition update")
                return False
            
            params.append(kpi_id)
            update_clause = ", ".join(update_fields)
            
            cursor.execute(f"""
                UPDATE kpi_definitions 
                SET {update_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE kpi_id = %s
            """, params)
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Updated KPI definition {kpi_id}")
                return True
            else:
                logger.warning(f"No KPI definition found with ID {kpi_id}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to update KPI definition {kpi_id}: {e}")
        return False

def delete_kpi_definition(kpi_id: int) -> bool:
    """Delete a KPI definition (soft delete by setting is_active = FALSE)."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE kpi_definitions 
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE kpi_id = %s
            """, (kpi_id,))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Deactivated KPI definition {kpi_id}")
                return True
            else:
                logger.warning(f"No KPI definition found with ID {kpi_id}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to delete KPI definition {kpi_id}: {e}")
        return False

# --- KPI Calculations Management --- #

def add_kpi_calculation(kpi_id: int, entity_type: str, entity_id: int, entity_name: str,
                       calculated_value: float, calculation_date: Optional[datetime] = None,
                       normalized_value: Optional[float] = None, target_achievement_rate: Optional[float] = None,
                       status: str = 'normal', data_quality_score: float = 1.0,
                       platform_contributions: Optional[Dict] = None, metric_contributions: Optional[Dict] = None,
                       rank_position: Optional[int] = None, percentile: Optional[float] = None) -> Optional[int]:
    """Add a KPI calculation result and return the calculation_id."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if calculation_date is None:
                calculation_date = datetime.now().date()
            
            # Convert contributions to JSON
            platform_json = json.dumps(platform_contributions) if platform_contributions else None
            metric_json = json.dumps(metric_contributions) if metric_contributions else None
            
            cursor.execute("""
                INSERT INTO kpi_calculations (
                    kpi_id, entity_type, entity_id, entity_name, calculated_value,
                    calculation_date, normalized_value, target_achievement_rate, status,
                    data_quality_score, platform_contributions, metric_contributions,
                    rank_position, percentile
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING calculation_id
            """, (kpi_id, entity_type, entity_id, entity_name, calculated_value,
                  calculation_date, normalized_value, target_achievement_rate, status,
                  data_quality_score, platform_json, metric_json, rank_position, percentile))
            
            calculation_id = cursor.fetchone()[0]
            conn.commit()
            logger.debug(f"Added KPI calculation for {entity_type} {entity_id}")
            return calculation_id
            
    except psycopg2.IntegrityError as e:
        if "uq_kpi_calc_unique" in str(e):
            logger.debug(f"KPI calculation already exists for {entity_type} {entity_id} on {calculation_date}")
        else:
            logger.error(f"Integrity error adding KPI calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to add KPI calculation: {e}")
        return None

def get_kpi_calculations(kpi_id: Optional[int] = None, entity_type: Optional[str] = None,
                        entity_id: Optional[int] = None, calculation_date: Optional[datetime] = None,
                        days_back: int = 30, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get KPI calculations with optional filters."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = []
            params = []
            
            if kpi_id:
                where_conditions.append("kc.kpi_id = %s")
                params.append(kpi_id)
            
            if entity_type:
                where_conditions.append("kc.entity_type = %s")
                params.append(entity_type)
            
            if entity_id:
                where_conditions.append("kc.entity_id = %s")
                params.append(entity_id)
            
            if calculation_date:
                where_conditions.append("kc.calculation_date = %s")
                params.append(calculation_date)
            else:
                # Default to recent calculations
                cutoff_date = datetime.now().date() - timedelta(days=days_back)
                where_conditions.append("kc.calculation_date >= %s")
                params.append(cutoff_date)
            
            where_clause = " AND ".join(where_conditions)
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            cursor.execute(f"""
                SELECT kc.*, kd.kpi_name, kd.kpi_description, kd.kpi_category,
                       kd.target_value, kd.warning_threshold, kd.critical_threshold
                FROM kpi_calculations kc
                JOIN kpi_definitions kd ON kc.kpi_id = kd.kpi_id
                {where_clause}
                ORDER BY kc.calculation_date DESC, kc.calculated_value DESC
                {limit_clause}
            """, params)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for result in results:
                if result.get('platform_contributions'):
                    try:
                        result['platform_contributions'] = json.loads(result['platform_contributions'])
                    except json.JSONDecodeError:
                        result['platform_contributions'] = {}
                
                if result.get('metric_contributions'):
                    try:
                        result['metric_contributions'] = json.loads(result['metric_contributions'])
                    except json.JSONDecodeError:
                        result['metric_contributions'] = {}
            
            return results
            
    except Exception as e:
        logger.error(f"Failed to get KPI calculations: {e}")
        return []

def get_latest_kpi_calculations(entity_type: Optional[str] = None, entity_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get latest KPI calculations for entities."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = []
            params = []
            
            if entity_type:
                where_conditions.append("entity_type = %s")
                params.append(entity_type)
            
            if entity_id:
                where_conditions.append("entity_id = %s")
                params.append(entity_id)
            
            where_clause = " AND ".join(where_conditions)
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            cursor.execute(f"""
                SELECT * FROM v_kpi_latest
                {where_clause}
                ORDER BY kpi_name, entity_type, entity_name
            """, params)
            
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get latest KPI calculations: {e}")
        return []

def get_kpi_dashboard_summary() -> List[Dict[str, Any]]:
    """Get KPI dashboard summary using the view."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM v_kpi_dashboard ORDER BY kpi_name")
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get KPI dashboard summary: {e}")
        return []

def get_kpi_top_performers(kpi_name: Optional[str] = None, entity_type: Optional[str] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
    """Get top performers for KPIs."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_conditions = []
            params = []
            
            if kpi_name:
                where_conditions.append("kpi_name = %s")
                params.append(kpi_name)
            
            if entity_type:
                where_conditions.append("entity_type = %s")
                params.append(entity_type)
            
            where_clause = " AND ".join(where_conditions)
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            params.append(limit)
            
            cursor.execute(f"""
                SELECT * FROM v_kpi_top_performers
                {where_clause}
                ORDER BY kpi_name, rank_position ASC
                LIMIT %s
            """, params)
            
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        logger.error(f"Failed to get KPI top performers: {e}")
        return []

def update_kpi_calculation_rankings(kpi_id: int, calculation_date: Optional[datetime] = None) -> bool:
    """Update rank and percentile for all calculations of a specific KPI on a specific date."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if calculation_date is None:
                calculation_date = datetime.now().date()
            
            # Calculate rankings using window functions
            cursor.execute("""
                WITH ranked_calculations AS (
                    SELECT 
                        calculation_id,
                        calculated_value,
                        RANK() OVER (ORDER BY calculated_value DESC) as rank_pos,
                        PERCENT_RANK() OVER (ORDER BY calculated_value DESC) as percentile_pos
                    FROM kpi_calculations 
                    WHERE kpi_id = %s AND calculation_date = %s
                )
                UPDATE kpi_calculations 
                SET 
                    rank_position = rc.rank_pos,
                    percentile = ROUND((1 - rc.percentile_pos) * 100, 2),
                    updated_at = CURRENT_TIMESTAMP
                FROM ranked_calculations rc
                WHERE kpi_calculations.calculation_id = rc.calculation_id
            """, (kpi_id, calculation_date))
            
            updated_count = cursor.rowcount
            conn.commit()
            logger.info(f"Updated rankings for {updated_count} KPI calculations")
            return True
            
    except Exception as e:
        logger.error(f"Failed to update KPI calculation rankings: {e}")
        return False

def bulk_insert_kpi_calculations(calculations: List[Dict[str, Any]]) -> int:
    """Bulk insert KPI calculations. Returns number of records inserted."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare bulk insert data
            insert_data = []
            for calc in calculations:
                platform_json = json.dumps(calc.get('platform_contributions')) if calc.get('platform_contributions') else None
                metric_json = json.dumps(calc.get('metric_contributions')) if calc.get('metric_contributions') else None
                
                insert_data.append((
                    calc['kpi_id'], calc['entity_type'], calc['entity_id'], calc['entity_name'],
                    calc['calculated_value'], calc.get('calculation_date', datetime.now().date()),
                    calc.get('normalized_value'), calc.get('target_achievement_rate'),
                    calc.get('status', 'normal'), calc.get('data_quality_score', 1.0),
                    platform_json, metric_json, calc.get('rank_position'), calc.get('percentile')
                ))
            
            cursor.executemany("""
                INSERT INTO kpi_calculations (
                    kpi_id, entity_type, entity_id, entity_name, calculated_value,
                    calculation_date, normalized_value, target_achievement_rate, status,
                    data_quality_score, platform_contributions, metric_contributions,
                    rank_position, percentile
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (kpi_id, entity_type, entity_id, calculation_date) 
                DO UPDATE SET
                    calculated_value = EXCLUDED.calculated_value,
                    normalized_value = EXCLUDED.normalized_value,
                    target_achievement_rate = EXCLUDED.target_achievement_rate,
                    status = EXCLUDED.status,
                    data_quality_score = EXCLUDED.data_quality_score,
                    platform_contributions = EXCLUDED.platform_contributions,
                    metric_contributions = EXCLUDED.metric_contributions,
                    rank_position = EXCLUDED.rank_position,
                    percentile = EXCLUDED.percentile,
                    updated_at = CURRENT_TIMESTAMP
            """, insert_data)
            
            inserted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Bulk inserted/updated {inserted_count} KPI calculations")
            return inserted_count
            
    except Exception as e:
        logger.error(f"Failed to bulk insert KPI calculations: {e}")
        return 0

def cleanup_old_kpi_calculations(days_to_keep: int = 365) -> int:
    """Remove old KPI calculations beyond the retention period."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)
            cursor.execute("""
                DELETE FROM kpi_calculations 
                WHERE calculation_date < %s
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted_count} old KPI calculations")
            return deleted_count
            
    except Exception as e:
        logger.error(f"Failed to cleanup old KPI calculations: {e}")
        return 0

# Initialize connection pool when module is imported
try:
    init_connection_pool()
except Exception as e:
    logger.warning(f"Failed to initialize connection pool on import: {e}")
    logger.warning("Connection pool will be initialized on first use")

def get_platform_metrics_history(account_id: int, platform: str, metric_type: str, 
                                days_back: int = 30) -> List[Dict[str, Any]]:
    """
    Get historical platform metrics for an account
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT 
                pm.recorded_at,
                pm.metric_value,
                pm.metric_type,
                pm.platform,
                aa.account_identifier
            FROM platform_metrics pm
            JOIN artist_accounts aa ON pm.account_id = aa.account_id
            WHERE pm.account_id = %s 
                AND pm.platform = %s 
                AND pm.metric_type = %s
                AND pm.recorded_at >= NOW() - INTERVAL '%s days'
            ORDER BY pm.recorded_at DESC
            """
            
            cursor.execute(query, (account_id, platform, metric_type, days_back))
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
    except Exception as e:
        logger.error(f"Failed to get platform metrics history: {e}")
        return []

class DatabaseManager:
    """Legacy compatibility class for old code"""
    
    @staticmethod
    def get_connection():
        """Get database connection - legacy method"""
        return get_db_connection()
    
    @staticmethod 
    def get_artists():
        """Get all artists - legacy method"""
        return get_artists()
    
    @staticmethod
    def get_companies():
        """Get all companies - legacy method"""
        return get_companies()
    
    @staticmethod
    def get_groups():
        """Get all groups - legacy method"""
        return get_groups()
        
    @staticmethod
    def add_event(name: str, event_type: str, event_date: str, **kwargs):
        """Add event - legacy method"""
        return add_event(name, event_type, event_date, **kwargs)
    
    @staticmethod
    def get_events_for_artist(artist_id: int):
        """Get events for artist - legacy method"""
        return get_events_for_artist(artist_id)
    
    @staticmethod
    def get_events_for_group(group_id: int):
        """Get events for group - legacy method"""
        return get_events_for_group(group_id)


def get_main_dashboard_summary() -> Dict[str, int]:
    """Get summary data for the main dashboard."""
    summary = {
        'total_artists': 0, 
        'total_groups': 0,
        'total_subscribers': 0,
        'active_platforms': 0
    }
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get total active artists
            cursor.execute("SELECT COUNT(*) as total_artists FROM artists WHERE is_active = TRUE")
            artist_result = cursor.fetchone()
            if artist_result:
                summary['total_artists'] = artist_result['total_artists']

            # Get total groups
            cursor.execute("SELECT COUNT(*) as total_groups FROM groups")
            group_result = cursor.fetchone()
            if group_result:
                summary['total_groups'] = group_result['total_groups']

            # Get total subscribers from the pre-calculated view for efficiency
            cursor.execute("""
                SELECT SUM(
                    COALESCE(youtube_subscribers, 0) + 
                    COALESCE(spotify_monthly_listeners, 0) + 
                    COALESCE(twitter_followers, 0) + 
                    COALESCE(instagram_followers, 0) + 
                    COALESCE(tiktok_followers, 0)
                ) as total_subscribers
                FROM v_growth_summary
            """)
            subscriber_result = cursor.fetchone()
            if subscriber_result and subscriber_result['total_subscribers']:
                summary['total_subscribers'] = int(subscriber_result['total_subscribers'])
            
            # Get active platform count (platforms with recent data)
            cursor.execute("""
                SELECT COUNT(DISTINCT platform) as active_platforms
                FROM platform_metrics
                WHERE collected_at >= NOW() - INTERVAL '1 day'
            """)
            platform_result = cursor.fetchone()
            if platform_result:
                summary['active_platforms'] = platform_result['active_platforms']

            return summary
    except Exception as e:
        logger.error(f"Failed to get main dashboard summary: {e}")
        return summary


# =================================================================================
# KPI and Executive Dashboard Functions
# =================================================================================

def get_executive_kpi_summary():
    """경영진 KPI 대시보드를 위한 핵심 지표 요약 데이터 조회"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # 총 아티스트, 그룹, 회사 수
                cursor.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM artists) as total_artists,
                        (SELECT COUNT(*) FROM groups) as total_groups,
                        (SELECT COUNT(*) FROM companies) as total_companies,
                        (SELECT COUNT(*) FROM artist_accounts) as total_accounts
                """)
                counts = cursor.fetchone()
                
                # 최신 플랫폼 지표 합계
                cursor.execute("""
                    SELECT 
                        platform,
                        metric_type,
                        SUM(value) as total_value,
                        COUNT(*) as account_count
                    FROM (
                        SELECT DISTINCT ON (account_id, platform, metric_type)
                            account_id, platform, metric_type, value
                        FROM platform_metrics
                        WHERE collected_at >= NOW() - INTERVAL '7 days'
                        ORDER BY account_id, platform, metric_type, collected_at DESC
                    ) latest_metrics
                    GROUP BY platform, metric_type
                    ORDER BY platform, metric_type
                """)
                metrics = cursor.fetchall()
                
                return {
                    'counts': dict(counts) if counts else {},
                    'metrics': [dict(m) for m in metrics]
                }
                
    except Exception as e:
        logger.error(f"Failed to get executive KPI summary: {e}")
        return {'counts': {}, 'metrics': []}

def get_top_growing_groups(limit=5, days_back=30):
    """성장률 기준 상위 그룹들을 조회"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    WITH growth_calculation AS (
                        SELECT 
                            g.group_id,
                            g.name as group_name,
                            c.name as company_name,
                            pm.platform,
                            pm.metric_type,
                            FIRST_VALUE(pm.value) OVER (
                                PARTITION BY g.group_id, pm.platform, pm.metric_type 
                                ORDER BY pm.collected_at DESC
                            ) as latest_value,
                            FIRST_VALUE(pm.value) OVER (
                                PARTITION BY g.group_id, pm.platform, pm.metric_type 
                                ORDER BY pm.collected_at ASC
                            ) as earliest_value,
                            COUNT(*) OVER (
                                PARTITION BY g.group_id, pm.platform, pm.metric_type
                            ) as data_points
                        FROM groups g
                        JOIN artist_accounts aa ON g.group_id = aa.group_id  -- 그룹별 계정 직접 연결
                        JOIN platform_metrics pm ON aa.account_id = pm.account_id
                        JOIN companies c ON g.company_id = c.company_id
                        WHERE pm.collected_at >= NOW() - INTERVAL '%s days'
                            AND pm.value > 0
                            AND aa.is_active = true  -- 활성 계정만
                    ),
                    group_growth AS (
                        SELECT DISTINCT
                            group_id,
                            group_name,
                            company_name,
                            CASE 
                                WHEN earliest_value > 0 
                                THEN ((latest_value - earliest_value) * 100.0 / earliest_value)
                                ELSE 0 
                            END as growth_rate,
                            latest_value,
                            data_points
                        FROM growth_calculation
                        WHERE data_points >= 2
                    ),
                    aggregated_growth AS (
                        SELECT 
                            group_id,
                            group_name,
                            company_name,
                            MAX(growth_rate) as max_growth_rate, -- Changed from AVG to MAX
                            SUM(latest_value) as total_followers,
                            COUNT(*) as platform_count
                        FROM group_growth
                        GROUP BY group_id, group_name, company_name
                        HAVING SUM(latest_value) > 100000 AND COUNT(*) >= 2
                    )
                    SELECT 
                        group_name,
                        company_name,
                        ROUND(max_growth_rate, 2) as growth_rate, -- Changed to max_growth_rate
                        total_followers,
                        platform_count
                    FROM aggregated_growth
                    ORDER BY max_growth_rate DESC -- Changed to max_growth_rate
                    LIMIT %s
                """, (days_back, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
    except Exception as e:
        logger.error(f"Failed to get top growing groups: {e}")
        return []

def get_platform_summary_stats():
    """플랫폼별 요약 통계 조회"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    WITH latest_metrics AS (
                        SELECT DISTINCT ON (account_id, platform)
                            account_id, platform, metric_type, value, collected_at
                        FROM platform_metrics
                        WHERE collected_at >= NOW() - INTERVAL '7 days'
                        ORDER BY account_id, platform, collected_at DESC
                    ),
                    previous_metrics AS (
                        SELECT DISTINCT ON (account_id, platform)
                            account_id, platform, metric_type, value, collected_at
                        FROM platform_metrics
                        WHERE collected_at >= NOW() - INTERVAL '14 days'
                            AND collected_at < NOW() - INTERVAL '7 days'
                        ORDER BY account_id, platform, collected_at DESC
                    )
                    SELECT 
                        l.platform,
                        COUNT(DISTINCT l.account_id) as active_accounts,
                        SUM(l.value) as current_total,
                        COALESCE(SUM(p.value), 0) as previous_total,
                        CASE 
                            WHEN COALESCE(SUM(p.value), 0) > 0 
                            THEN ROUND(((SUM(l.value) - COALESCE(SUM(p.value), 0)) * 100.0 / COALESCE(SUM(p.value), 1)), 2)
                            ELSE 0 
                        END as growth_rate
                    FROM latest_metrics l
                    LEFT JOIN previous_metrics p ON l.account_id = p.account_id AND l.platform = p.platform
                    WHERE l.metric_type IN ('subscribers', 'followers')
                    GROUP BY l.platform
                    ORDER BY current_total DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
    except Exception as e:
        logger.error(f"Failed to get platform summary stats: {e}")
        return []

def format_number(num):
    """숫자를 K, M, B 단위로 포맷팅"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))

def calculate_growth_percentage(current, previous):
    """성장률 계산"""
    if previous == 0:
        return 0
    return round(((current - previous) * 100.0 / previous), 2)

def get_basic_growth_analysis(days_back=30):
    """기본 성장률 분석을 위한 데이터 조회"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    WITH artist_growth AS (
                        SELECT 
                            a.artist_name,
                            g.name as group_name,
                            c.name as company_name,
                            pm.platform,
                            pm.metric_type,
                            FIRST_VALUE(pm.value) OVER (
                                PARTITION BY a.artist_id, pm.platform, pm.metric_type 
                                ORDER BY pm.collected_at DESC
                            ) as latest_value,
                            FIRST_VALUE(pm.value) OVER (
                                PARTITION BY a.artist_id, pm.platform, pm.metric_type 
                                ORDER BY pm.collected_at ASC
                            ) as earliest_value,
                            COUNT(*) OVER (
                                PARTITION BY a.artist_id, pm.platform, pm.metric_type
                            ) as data_points
                        FROM artists a
                        LEFT JOIN groups g ON a.group_id = g.group_id
                        JOIN companies c ON COALESCE(g.company_id, a.company_id) = c.company_id
                        JOIN artist_accounts aa ON a.artist_id = aa.artist_id
                        JOIN platform_metrics pm ON aa.account_id = pm.account_id
                        WHERE pm.collected_at >= NOW() - INTERVAL '%s days'
                            AND pm.value > 0
                            AND pm.metric_type IN ('subscribers', 'followers')
                    ),
                    growth_calculated AS (
                        SELECT DISTINCT
                            artist_name,
                            COALESCE(group_name, '솔로') as group_name,
                            company_name,
                            platform,
                            metric_type,
                            latest_value,
                            earliest_value,
                            CASE 
                                WHEN earliest_value > 0 
                                THEN ROUND(((latest_value - earliest_value) * 100.0 / earliest_value), 2)
                                ELSE 0 
                            END as growth_rate,
                            data_points
                        FROM artist_growth
                        WHERE data_points >= 2
                    )
                    SELECT 
                        artist_name,
                        group_name,
                        company_name,
                        platform,
                        SUM(latest_value) as total_followers,
                        AVG(growth_rate) as avg_growth_rate,
                        COUNT(*) as platform_count
                    FROM growth_calculated
                    GROUP BY artist_name, group_name, company_name, platform
                    HAVING AVG(growth_rate) IS NOT NULL
                    ORDER BY avg_growth_rate DESC
                    LIMIT 50
                """, (days_back,))
                
                return [dict(row) for row in cursor.fetchall()]
                
    except Exception as e:
        logger.error(f"Failed to get basic growth analysis: {e}")
        return []

def get_group_growth_analysis(days_back=30):
    """그룹 기준 성장률 분석 - 그룹 멤버들의 지표를 통합하여 분석"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    WITH group_metrics_aggregated AS (
                        SELECT 
                            g.name as group_name,
                            g.name as display_name,
                            c.name as company_name,
                            pm.platform,
                            pm.metric_type,
                            pm.collected_at,
                            pm.value as aggregated_value  -- 그룹별 통합 계정 값
                        FROM groups g
                        JOIN companies c ON g.company_id = c.company_id
                        JOIN artist_accounts aa ON g.group_id = aa.group_id  -- 그룹별 계정 연결
                        JOIN platform_metrics pm ON aa.account_id = pm.account_id
                        WHERE pm.collected_at >= NOW() - INTERVAL '%s days'
                            AND pm.value > 0
                            AND pm.metric_type IN ('subscribers', 'followers')
                            AND aa.is_active = true  -- 활성 계정만
                    ),
                    group_growth_calculation AS (
                        SELECT 
                            group_name,
                            display_name,
                            company_name,
                            platform,
                            metric_type,
                            MIN(aggregated_value) as earliest_value,
                            MAX(aggregated_value) as latest_value,
                            COUNT(*) as data_points
                        FROM group_metrics_aggregated
                        GROUP BY group_name, display_name, company_name, platform, metric_type
                        HAVING COUNT(*) >= 2  -- 최소 2개 데이터 포인트 필요
                    ),
                    growth_with_rates AS (
                        SELECT 
                            group_name,
                            display_name,
                            company_name,
                            platform,
                            latest_value as total_followers,
                            CASE 
                                WHEN earliest_value > 0 
                                THEN ROUND(((latest_value - earliest_value) * 100.0 / earliest_value), 2)
                                ELSE 0 
                            END as growth_rate,
                            data_points
                        FROM group_growth_calculation
                    )
                    SELECT 
                        display_name as group_name,
                        company_name,
                        platform,
                        total_followers,
                        AVG(growth_rate) as avg_growth_rate,
                        COUNT(DISTINCT platform) as platform_count
                    FROM growth_with_rates
                    GROUP BY display_name, company_name, platform, total_followers
                    HAVING AVG(growth_rate) IS NOT NULL
                    ORDER BY avg_growth_rate DESC
                    LIMIT 50
                """, (days_back,))
                
                return [dict(row) for row in cursor.fetchall()]
                
    except Exception as e:
        logger.error(f"Failed to get group growth analysis: {e}")
        return []

def get_platform_growth_comparison_groups():
    """그룹 기준 플랫폼별 성장률 비교 데이터"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    WITH group_platform_growth AS (
                        SELECT 
                            group_name,
                            platform,
                            metric_type,
                            SUM(latest_value) as latest_total,
                            SUM(earliest_value) as earliest_total
                        FROM (
                            SELECT DISTINCT
                                g.name as group_name,
                                pm.platform,
                                pm.metric_type,
                                FIRST_VALUE(pm.value) OVER (
                                    PARTITION BY g.group_id, pm.platform, pm.metric_type 
                                    ORDER BY pm.collected_at DESC
                                ) as latest_value,
                                FIRST_VALUE(pm.value) OVER (
                                    PARTITION BY g.group_id, pm.platform, pm.metric_type 
                                    ORDER BY pm.collected_at ASC
                                ) as earliest_value,
                                g.group_id,
                                COUNT(*) OVER (
                                    PARTITION BY g.group_id, pm.platform, pm.metric_type
                                ) as data_points
                            FROM groups g
                            JOIN artist_accounts aa ON g.group_id = aa.group_id  -- 그룹별 계정 직접 연결
                            JOIN platform_metrics pm ON aa.account_id = pm.account_id
                            WHERE pm.collected_at >= NOW() - INTERVAL '30 days'
                                AND pm.value > 0
                                AND pm.metric_type IN ('subscribers', 'followers')
                                AND g.group_id IS NOT NULL  -- Only include artists with groups
                        ) artist_metrics
                        WHERE data_points >= 2
                        GROUP BY 
                            group_name,
                            platform,
                            metric_type
                    )
                    SELECT 
                        platform,
                        COUNT(DISTINCT group_name) as active_groups,
                        AVG(CASE 
                            WHEN earliest_total > 0 
                            THEN ((latest_total - earliest_total) * 100.0 / earliest_total)
                            ELSE 0 
                        END) as growth_rate
                    FROM group_platform_growth
                    GROUP BY platform
                    ORDER BY growth_rate DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
    except Exception as e:
        logger.error(f"Failed to get platform growth comparison for groups: {e}")
        return []

# Keep original function for backward compatibility
def get_platform_growth_comparison():
    """플랫폼별 성장률 비교 데이터 (레거시 - 개별 아티스트 기준)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    WITH platform_growth AS (
                        SELECT 
                            pm.platform,
                            COUNT(DISTINCT aa.account_id) as active_accounts,
                            AVG(CASE 
                                WHEN pm1.value > 0 
                                THEN ((pm.value - pm1.value) * 100.0 / pm1.value)
                                ELSE 0 
                            END) as avg_growth_rate
                        FROM platform_metrics pm
                        JOIN artist_accounts aa ON pm.account_id = aa.account_id
                        LEFT JOIN platform_metrics pm1 ON pm.account_id = pm1.account_id 
                            AND pm.platform = pm1.platform 
                            AND pm.metric_type = pm1.metric_type
                            AND pm1.collected_at >= NOW() - INTERVAL '14 days'
                            AND pm1.collected_at < NOW() - INTERVAL '7 days'
                        WHERE pm.collected_at >= NOW() - INTERVAL '7 days'
                            AND pm.metric_type IN ('subscribers', 'followers')
                            AND pm.value > 0
                        GROUP BY pm.platform
                    )
                    SELECT 
                        platform,
                        active_accounts,
                        ROUND(COALESCE(avg_growth_rate, 0), 2) as growth_rate
                    FROM platform_growth
                    ORDER BY growth_rate DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
    except Exception as e:
        logger.error(f"Failed to get platform growth comparison: {e}")
        return []
