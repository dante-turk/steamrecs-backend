import random
import time
from collections import deque
from typing import Deque, Set, Tuple

from steam_fetch_simple import get_friend_ids, get_owned_games, SteamPrivacyError
from db import get_conn, upsert_user, upsert_game, upsert_user_game


MAX_DEPTH = 4
FRIENDS_SAMPLE_SIZE = 50
MIN_GAMES_REQUIRED = 5
MIN_PLAYTIME_MINUTES = 120  # 5 hours
SLEEP_SECONDS = 0.5


def already_fetched(steamid: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT 1 FROM users WHERE steamid = ? AND last_fetched_at IS NOT NULL",
            (steamid,),
        )
        return cur.fetchone() is not None


def total_users() -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM users")
        return cur.fetchone()[0]


def crawl_user_store_if_valid(steamid: str, depth: int) -> None:
    try:
        owned = get_owned_games(steamid)
    except SteamPrivacyError:
        upsert_user(steamid, depth, is_public=0)
        return

    games = owned.get("games", []) or []

    played_games = [
        g for g in games
        if (g.get("playtime_forever", 0) or 0) >= MIN_PLAYTIME_MINUTES
    ]

    upsert_user(steamid, depth, is_public=1)

    if len(played_games) < MIN_GAMES_REQUIRED:
        return

    for g in played_games:
        appid = g["appid"]
        name = g.get("name")
        upsert_game(appid, name)
        upsert_user_game(
            steamid,
            appid,
            g.get("playtime_forever", 0) or 0,
            g.get("playtime_2weeks", 0) or 0,
        )


def crawl_network(seed_steamid: str):
    queue: Deque[Tuple[str, int]] = deque()
    visited: Set[str] = set()

    queue.append((seed_steamid, 0))

    while queue:
        steamid, depth = queue.popleft()

        if steamid in visited:
            continue
        visited.add(steamid)

        if depth > MAX_DEPTH:
            continue

        if not already_fetched(steamid):
            crawl_user_store_if_valid(steamid, depth)
            if total_users() % 50 == 0:
                print(f"users stored: {total_users()}")
            time.sleep(SLEEP_SECONDS)

        if depth == MAX_DEPTH:
            continue

        try:
            friends = get_friend_ids(steamid)
        except SteamPrivacyError:
            continue

        if not friends:
            continue

        sampled = random.sample(friends, min(FRIENDS_SAMPLE_SIZE, len(friends)))

        from steam_fetch_simple import filter_public_profiles
        sampled = filter_public_profiles(sampled)

        for fid in sampled:
            if fid not in visited:
                queue.append((fid, depth + 1))


if __name__ == "__main__":
    SEED_STEAMID = "76561198121222250"  # replace with your SteamID64
    crawl_network(SEED_STEAMID)
