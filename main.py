from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import csr_matrix, load_npz

from usemodel import (
    get_game_names_from_db,
    load_artifacts,
    recommend_for_known_user,
    recommend_for_pseudo_user,
)
from steam_fetch_simple import get_owned_games
# Load env vars early (so STEAM_API_KEY is available to imported modules)
load_dotenv()

# --------------------------------------------------
# Paths (all files in same folder as main.py)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "steam_recs.db")
MODEL_PATH = os.path.join(BASE_DIR, "als_model.pkl")
MAPS_PATH = os.path.join(BASE_DIR, "mappings.pkl")
USER_ITEM_PATH = os.path.join(BASE_DIR, "user_item.npz")

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # CRA default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Global cache (loaded once at startup, used for /health)
# --------------------------------------------------
MODEL = None
MAPS: Optional[dict] = None
USER_ITEM: Optional[csr_matrix] = None


class Recommendation(BaseModel):
    appid: int
    name: str
    score: float


class RecommendResponse(BaseModel):
    steamid: str
    n: int
    source: str  # "known_user" or "pseudo_user"
    recommendations: List[Recommendation]
    warning: Optional[str] = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def recommend_any_user(steamid: str, n: int) -> dict:
    """
    Returns a payload dict:
      {
        "steamid": ...,
        "n": ...,
        "mode": "known_user" | "pseudo_user",
        "recommendations": [{"appid":..., "name":..., "score":...}, ...],
        "warning": optional str
      }
    """
    _model, maps, _user_item = load_artifacts()
    user_to_idx = maps["user_to_idx"]

    if steamid in user_to_idx:
        results = recommend_for_known_user(steamid, N=n)
        mode = "known_user"
    else:
        results = recommend_for_pseudo_user(steamid, N=n)
        mode = "pseudo_user"

    appids = [appid for appid, _score in results]
    names = get_game_names_from_db(appids)

    payload = {
        "steamid": steamid,
        "n": n,
        "mode": mode,
        "recommendations": [
            {"appid": appid, "name": names.get(appid, "Unknown"), "score": float(score)}
            for appid, score in results
        ],
    }

    if mode == "pseudo_user" and len(results) == 0:
        payload["warning"] = "No overlap between this user's library and the model's known games."

    return payload


def get_game_names(appids: List[int]) -> Dict[int, str]:
    """Only used in legacy code paths; kept for convenience."""
    if not appids:
        return {}
    conn = sqlite3.connect(DB_PATH)
    try:
        placeholders = ",".join(["?"] * len(appids))
        df = pd.read_sql_query(
            f"SELECT appid, name FROM games WHERE appid IN ({placeholders})",
            conn,
            params=tuple(appids),
        )
    finally:
        conn.close()
    return {int(r["appid"]): (r["name"] or "Unknown") for _, r in df.iterrows()}


# --------------------------------------------------
# Startup: load model + data ONCE (for /health)
# --------------------------------------------------
@app.on_event("startup")
def load_artifacts_once():
    global MODEL, MAPS, USER_ITEM

    missing = [p for p in [MODEL_PATH, MAPS_PATH, USER_ITEM_PATH, DB_PATH] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")

    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)

    with open(MAPS_PATH, "rb") as f:
        MAPS = pickle.load(f)

    USER_ITEM = load_npz(USER_ITEM_PATH).tocsr()


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Steam recommender API running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "model_loaded": MODEL is not None,
        "mappings_loaded": MAPS is not None,
        "user_item_loaded": USER_ITEM is not None,
        "db_exists": os.path.exists(DB_PATH),
        "steam_api_key_set": bool(os.environ.get("STEAM_API_KEY")),
    }


@app.get("/recommend/{steamid}", response_model=RecommendResponse)
async def recommend(steamid: str, n: int = Query(15, ge=1, le=50)):
    payload = recommend_any_user(steamid, n)

    # IMPORTANT: payload already contains names/scores. Don't re-wrap it.
    return {
        "steamid": payload["steamid"],
        "n": payload["n"],
        "source": payload["mode"],
        "recommendations": payload["recommendations"],
        "warning": payload.get("warning"),
    }


@app.get("/user/{steamid}")
async def user_info(steamid: str):
    in_model = MAPS is not None and steamid in MAPS.get("user_to_idx", {})
    db_game_count = 0

    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT COUNT(*) FROM user_games WHERE steamid = ?", (steamid,)).fetchone()
        db_game_count = int(row[0]) if row else 0
    finally:
        conn.close()

    return {
        "steamid": steamid,
        "in_model": in_model,
        "db_games_for_user": db_game_count,
    }
