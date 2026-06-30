"""
SQLite 数据库模块 —— 翻译历史 + 缓存。
"""
import sqlite3
import hashlib
import time
from pathlib import Path
from contextlib import contextmanager
from config import CONFIG_DIR


DB_PATH = CONFIG_DIR / "translator.db"


def _ensure_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _hash(text: str, direction: str, mode: str = "") -> str:
    """对输入文本、翻译方向和模式做 hash，用于缓存键。"""
    raw = f"{mode}:{direction}:{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def init_db():
    """初始化数据库表。"""
    _ensure_dir()
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                hash_key TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                direction TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                direction TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(hash_key)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_time ON history(created_at DESC)
        """)
        conn.commit()


@contextmanager
def _get_conn():
    """获取数据库连接上下文。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def cache_get(source: str, direction: str, mode: str = "") -> str | None:
    """查询缓存。"""
    key = _hash(source, direction, mode)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT translated_text FROM cache WHERE hash_key = ?", (key,)
        ).fetchone()
    return row["translated_text"] if row else None


def cache_set(source: str, translated: str, direction: str, mode: str):
    """写入缓存和历史。"""
    key = _hash(source, direction, mode)
    now = time.time()
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO cache (hash_key, source_text, translated_text, direction, mode, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, source, translated, direction, mode, now),
        )
        conn.execute(
            """INSERT INTO history (source_text, translated_text, direction, mode, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source, translated, direction, mode, now),
        )
        conn.commit()


def history_list(limit: int = 100) -> list[dict]:
    """获取翻译历史。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def history_clear():
    """清空历史。"""
    with _get_conn() as conn:
        conn.execute("DELETE FROM history")
        conn.commit()


def history_delete(history_id: int):
    """删除单条历史。"""
    with _get_conn() as conn:
        conn.execute("DELETE FROM history WHERE id = ?", (history_id,))
        conn.commit()


def cache_clear():
    """清空缓存。"""
    with _get_conn() as conn:
        conn.execute("DELETE FROM cache")
        conn.commit()


def prune(max_history: int = 500):
    """裁剪历史记录到指定条数。"""
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM history").fetchone()["cnt"]
        if count > max_history:
            conn.execute(
                """DELETE FROM history WHERE id NOT IN (
                    SELECT id FROM history ORDER BY created_at DESC LIMIT ?
                )""",
                (max_history,),
            )
            conn.commit()
