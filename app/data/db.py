"""Database connection and initialization."""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from app.core.settings import settings
from app.core.logger import logger


def get_db_path() -> Path:
    """Get the database path."""
    db_path = Path(settings.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database with schema."""
    schema_path = Path(__file__).parent / "schema.sql"
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    with get_connection() as conn:
        conn.executescript(schema_sql)
    
    logger.info("Database initialized successfully")


def upsert_candle(ts: str, symbol: str, o: float, h: float, l: float, c: float, v: float):
    """Insert or update a candle."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO candles (ts, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ts, symbol) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
        """, (ts, symbol, o, h, l, c, v))

