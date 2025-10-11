#!/usr/bin/env python3
"""Migrate database to add new tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.db import get_connection
from app.core.logger import logger

def migrate():
    """Run database migrations."""
    schema_path = Path(__file__).parent.parent / 'app/data/schema.sql'
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    with get_connection() as conn:
        # Execute schema (CREATE TABLE IF NOT EXISTS will only create missing tables)
        conn.executescript(schema_sql)
    
    logger.info("Database migration complete")
    print("✅ Database migrated successfully!")

if __name__ == '__main__':
    migrate()

