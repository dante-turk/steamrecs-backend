import sqlite3
from datetime import datetime
from typing import Optional

DB_PATH = "steam_recs.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def upsert_user(
    steamid: str,
    depth: int,
    is_public: Optional[int]
):
    now = datetime.utcnow().isoformat()

    sql = """
    INSERT INTO users (steamid, depth, is_public, last_fetched_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(steamid) DO UPDATE SET
      depth = MIN(users.depth, excluded.depth),
      is_public = COALESCE(excluded.is_public, users.is_public),
      last_fetched_at = excluded.last_fetched_at;
    """

    with get_conn() as conn:
        conn.execute(sql, (steamid, depth, is_public, now))

def upsert_game(appid: int, name: Optional[str]):
    now = datetime.utcnow().isoformat()

    sql = """
    INSERT INTO games (appid, name, last_seen_at)
    VALUES (?, ?, ?)
    ON CONFLICT(appid) DO UPDATE SET
      name = COALESCE(excluded.name, games.name),
      last_seen_at = excluded.last_seen_at;
    """

    with get_conn() as conn:
        conn.execute(sql, (appid, name, now))
        
def upsert_user_game(
    steamid: str,
    appid: int,
    playtime_forever: int,
    playtime_2weeks: int
):
    now = datetime.utcnow().isoformat()

    sql = """
    INSERT INTO user_games (
        steamid,
        appid,
        playtime_forever_minutes,
        playtime_2weeks_minutes,
        last_updated_at
    )
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(steamid, appid) DO UPDATE SET
      playtime_forever_minutes = excluded.playtime_forever_minutes,
      playtime_2weeks_minutes = excluded.playtime_2weeks_minutes,
      last_updated_at = excluded.last_updated_at;
    """

    with get_conn() as conn:
        conn.execute(
            sql,
            (steamid, appid, playtime_forever, playtime_2weeks, now)
        )
