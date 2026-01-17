import sqlite3

DB_PATH = "steam_recs.db"

with sqlite3.connect(DB_PATH) as conn:
    with open("schema.sql", "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    
print("Database Initialized: ", DB_PATH)