import os
import sys
import time
from math import log1p
import requests
from typing import List, Dict, Any

STEAM_KEY = "4E864183A05FBB3CC2614246F11DB88C"
BASE = "https://api.steampowered.com"

if not STEAM_KEY:
    raise SystemExit("Set STEAM_API_KEY environment variable")

session = requests.Session()

class SteamPrivacyError(Exception):
    pass

def get_player_summaries(steamids: List[str]) -> List[Dict[str, any]]:
        url = f"{BASE}/ISteamUser/GetPlayerSummaries/v0002/"
        out: List[Dict[str, Any]] = []

        for i in range(0, len(steamids), 100):
             chunk = steamids[i:i+100]
             r = session.get(url, params={
                  "key": STEAM_KEY,
                  "steamids": ",".join(chunk)
             }, timeout=20)

             r.raise_for_status()
             out.extend(r.json().get("response", {}).get("players", []))
            
        return out

def filter_public_profiles(steamids: List[str]) -> List[str]:
    summaries = get_player_summaries(steamids)
    publicids = []

    for p in summaries:
         if(int(p.get("communityvisibilitystate", 1)) == 3):
              publicids.append(p["steamid"])
        
    return publicids

def get_friend_ids(steamid64: str) -> List[str]:
    url = f"{BASE}/ISteamUser/GetFriendList/v0001/"
    r = session.get(url, params={
        "key": STEAM_KEY,
        "steamid": steamid64,
        "relationship": "friend"
    }, timeout=20)

    if r.status_code == 401:
        raise SteamPrivacyError("Friend list is private (401).")

    r.raise_for_status()
    friends = r.json().get("friendslist", {}).get("friends", [])
    return [f["steamid"] for f in friends]

def get_recently_played_games(steamid64: str, count: int = 20) -> Dict[str, Any]:
    url = f"{BASE}/IPlayerService/GetRecentlyPlayedGames/v0001/"
    r = session.get(url, params={
         "key": STEAM_KEY,
         "steamid": steamid64,
         "count": count
    }, timeout = 20)

    if r.status_code == 401:
        raise SteamPrivacyError("Recently played games private (401).")

    r.raise_for_status()
    return r.json().get("response", {})

def get_owned_games(steamid64: str) -> Dict[str, Any]:
    url = f"{BASE}/IPlayerService/GetOwnedGames/v0001/"
    r = session.get(url, params={
        "key": STEAM_KEY,
        "steamid": steamid64,
        "include_appinfo": 1,
        "include_played_free_games": 1
    }, timeout=30)

    if r.status_code == 401:
         raise SteamPrivacyError("Owned games private (401).")
    
    r.raise_for_status()

    return r.json().get("response", {})

if __name__ == "__main__":
    MY_ID = "76561198966453583"

    print("=== TEST: FRIEND LIST ===")
    friends = get_friend_ids(MY_ID)
    print(f"Friends found: {len(friends)}")
    print("First 5 friend IDs:", friends[:5])

    print("\n=== TEST: PUBLIC FRIEND FILTER ===")
    public_friends = filter_public_profiles(friends)
    print(f"Public profiles: {len(public_friends)}")

    print("\n=== TEST: RECENTLY PLAYED ===")
    recent = get_recently_played_games(MY_ID)
    print("Recently played count:", recent.get("total_count", 0))
    for g in recent.get("games", [])[:5]:
        print(f"- {g['name']} ({g['playtime_2weeks']} mins)")

    print("\n=== TEST: OWNED GAMES ===")
    owned = get_owned_games(MY_ID)
    print("Owned games:", owned.get("game_count", 0))
    for g in owned.get("games", [])[:60]:
        print(f"- {g['name']} | total hours={g['playtime_forever'] / 60:.1f}")