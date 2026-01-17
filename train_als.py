import sqlite3
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle

DB_PATH = "steam_recs.db"

# Training filters (tune later)
MIN_GAMES_PER_USER = 5
MIN_USERS_PER_GAME = 5

# Weighting
ALPHA = 20.0  # confidence scaling
USE_TWO_WEEKS_BOOST = True


def load_interactions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT steamid, appid,
               playtime_forever_minutes,
               playtime_2weeks_minutes
        FROM user_games
        """,
        conn,
    )
    conn.close()
    return df


def build_weights(df: pd.DataFrame) -> pd.DataFrame:
    # Convert minutes -> hours
    hours = df["playtime_forever_minutes"].fillna(0).astype(float) / 60.0

    # Log-scale so whales don't dominate
    base = np.log1p(hours)

    if USE_TWO_WEEKS_BOOST:
        recent_hours = df["playtime_2weeks_minutes"].fillna(0).astype(float) / 60.0
        recent = np.log1p(recent_hours)
        # Add a small recency boost
        w = base + 0.5 * recent
    else:
        w = base

    # Implicit ALS expects confidence; we scale by alpha
    df = df.copy()
    df["weight"] = ALPHA * w

    # Remove zero weights just in case
    df = df[df["weight"] > 0]
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    # Filter users with too few interactions
    user_counts = df.groupby("steamid")["appid"].count()
    keep_users = user_counts[user_counts >= MIN_GAMES_PER_USER].index
    df = df[df["steamid"].isin(keep_users)]

    # Filter games with too few users
    game_counts = df.groupby("appid")["steamid"].count()
    keep_games = game_counts[game_counts >= MIN_USERS_PER_GAME].index
    df = df[df["appid"].isin(keep_games)]

    return df


def make_mappings(df: pd.DataFrame):
    user_ids = df["steamid"].unique()
    item_ids = df["appid"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}

    item_to_idx = {a: i for i, a in enumerate(item_ids)}
    idx_to_item = {i: a for a, i in item_to_idx.items()}

    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def build_sparse_matrix(df: pd.DataFrame, user_to_idx, item_to_idx) -> csr_matrix:
    rows = df["steamid"].map(user_to_idx).to_numpy()
    cols = df["appid"].map(item_to_idx).to_numpy()
    data = df["weight"].to_numpy(dtype=np.float32)

    mat = coo_matrix((data, (rows, cols)), shape=(len(user_to_idx), len(item_to_idx)))
    return mat.tocsr()


def train_als(user_item: csr_matrix) -> AlternatingLeastSquares:
    # implicit library is optimized for item-user matrix
    item_user = user_item.T.tocsr()

    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=20,
        random_state=42
    )

    # This step can be slow on big data; it’s normal.
    model.fit(item_user)
    return model


def save_artifacts(model, user_to_idx, idx_to_user, item_to_idx, idx_to_item):
    with open("als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("mappings.pkl", "wb") as f:
        pickle.dump(
            {
                "user_to_idx": user_to_idx,
                "idx_to_user": idx_to_user,
                "item_to_idx": item_to_idx,
                "idx_to_item": idx_to_item,
            },
            f,
        )


def main():
    df = load_interactions()
    print("raw rows:", len(df))

    df = build_weights(df)
    print("weighted rows:", len(df))

    df = apply_filters(df)
    print("filtered rows:", len(df))
    print("users:", df["steamid"].nunique(), "games:", df["appid"].nunique())

    user_to_idx, idx_to_user, item_to_idx, idx_to_item = make_mappings(df)
    user_item = build_sparse_matrix(df, user_to_idx, item_to_idx)
    print("matrix shape:", user_item.shape, "nnz:", user_item.nnz)

    model = train_als(user_item)
    save_artifacts(model, user_to_idx, idx_to_user, item_to_idx, idx_to_item)
    print("saved als_model.pkl and mappings.pkl")


if __name__ == "__main__":
    main()
