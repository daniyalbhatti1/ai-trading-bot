"""Initialize database schema."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.db import init_db
from app.core.logger import logger


def main():
    """Initialize the database."""
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("✓ Database initialized successfully!")
        print("\n✓ Database initialized successfully!")
        print(f"  Location: {Path(__file__).parent.parent / 'data' / 'trades.db'}")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

