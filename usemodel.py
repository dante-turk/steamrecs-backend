import os
import pickle
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

from steam_fetch_simple import get_owned_games, SteamPrivacyError


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "als_model.pkl")
MAPS_PATH = os.path.join(BASE_DIR, "mappings.pkl")
USER_ITEM_PATH = os.path.join(BASE_DIR, "user_item.npz")
DB_PATH = os.path.join(BASE_DIR, "steam_recs.db")

ALPHA = 20.0
USE_TWO_WEEKS_BOOST = True


# ----------------------------
# Artifact loading (cache)
# ----------------------------
_model_cache = None
_maps_cache = None
_user_item_cache = None

def load_artifacts():
    global _model_cache, _maps_cache, _user_item_cache
    if _model_cache is None:
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    if _maps_cache is None:
        with open(MAPS_PATH, "rb") as f:
            _maps_cache = pickle.load(f)
    if _user_item_cache is None:
        _user_item_cache = load_npz(USER_ITEM_PATH).tocsr()
    return _model_cache, _maps_cache, _user_item_cache


# ----------------------------
# Helpers
# ----------------------------
def get_game_names_from_db(appids: List[int]) -> Dict[int, str]:
    if not appids:
        return {}
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join(["?"] * len(appids))
    df = pd.read_sql_query(
        f"SELECT appid, name FROM games WHERE appid IN ({placeholders})",
        conn,
        params=tuple(appids),
    )
    conn.close()
    return {int(r["appid"]): (r["name"] or "Unknown") for _, r in df.iterrows()}


def build_weights_from_owned_games(owned_games: List[dict]) -> Dict[int, float]:
    """
    Returns {appid: weight} using the SAME weighting rule as training.
    Only uses games with playtime_forever > 0.
    """
    weights = {}
    for g in owned_games:
        appid = int(g["appid"])
        pf = float(g.get("playtime_forever", 0) or 0)
        if pf <= 0:
            continue

        hours = pf / 60.0
        base = np.log1p(hours)

        if USE_TWO_WEEKS_BOOST:
            p2w = float(g.get("playtime_2weeks", 0) or 0)
            recent_hours = p2w / 60.0
            recent = np.log1p(recent_hours)
            w = base + 0.5 * recent
        else:
            w = base

        weight = float(ALPHA * w)
        if weight > 0:
            weights[appid] = weight
    return weights


def build_pseudo_user_vector(steamid: str, item_to_idx: Dict[int, int], num_items: int) -> csr_matrix:
    owned = get_owned_games(steamid)
    games = owned.get("games", []) or []

    appid_to_weight = build_weights_from_owned_games(games)

    cols = []
    data = []

    for appid, weight in appid_to_weight.items():
        idx = item_to_idx.get(appid)
        if idx is None:
            continue
        if 0 <= idx < num_items:
            cols.append(idx)
            data.append(np.float32(weight))

    if not cols:
        return csr_matrix((1, num_items), dtype=np.float32)

    rows = np.zeros(len(cols), dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    return csr_matrix((data, (rows, cols)), shape=(1, num_items))


def recommend_for_pseudo_user(steamid: str, N: int = 15) -> List[Tuple[int, float]]:
    model, maps, _user_item = load_artifacts()
    item_to_idx = maps["item_to_idx"]
    idx_to_item = maps["idx_to_item"]

    num_items = model.item_factors.shape[0]  # ✅ the model’s truth
    user_vec = build_pseudo_user_vector(steamid, item_to_idx, num_items)

    if user_vec.nnz == 0:
        return []

    recs = model.recommend(
        userid=0,
        user_items=user_vec,
        N=N,
        filter_already_liked_items=True,
        recalculate_user=True,
    )

    out: List[Tuple[int, float]] = []

    # implicit compatibility: (item_ids, scores) or rows
    if isinstance(recs, tuple) and len(recs) == 2:
        item_ids, scores = recs
        for item_idx, score in zip(item_ids, scores):
            appid = int(idx_to_item[int(item_idx)])
            out.append((appid, float(score)))
    else:
        for row in recs:
            item_idx = row[0]
            score = row[1]
            appid = int(idx_to_item[int(item_idx)])
            out.append((appid, float(score)))

    return out


def recommend_for_known_user(steamid: str, N: int = 15) -> List[Tuple[int, float]]:
    model, maps, user_item = load_artifacts()
    user_to_idx = maps["user_to_idx"]
    idx_to_item = maps["idx_to_item"]

    uidx = user_to_idx[steamid]
    recs = model.recommend(
        userid=uidx,
        user_items=user_item[uidx],
        N=N,
        filter_already_liked_items=True,
    )

    out: List[Tuple[int, float]] = []
    if isinstance(recs, tuple) and len(recs) == 2:
        item_ids, scores = recs
        for item_idx, score in zip(item_ids, scores):
            appid = int(idx_to_item[int(item_idx)])
            out.append((appid, float(score)))
    else:
        for row in recs:
            appid = int(idx_to_item[int(row[0])])
            out.append((appid, float(row[1])))
    return out
