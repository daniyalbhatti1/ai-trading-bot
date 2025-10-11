"""Job scheduler for periodic tasks."""
import asyncio
import os
import ssl
import certifi
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
from app.ingestion.alpaca_stream import start_stream
from app.ingestion.symbols import get_universe
from app.trading.executor import on_new_bar
from app.core.settings import config
from app.core.logger import logger
from app.api.routes import is_trading_active

# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


scheduler = AsyncIOScheduler()


async def trading_loop():
    """Main trading loop - runs every minute."""
    if not is_trading_active():
        logger.debug("Trading inactive, skipping bar processing")
        return
    
    try:
        symbols = get_universe()
        logger.info(f"Processing bar for {len(symbols)} symbols")
        on_new_bar(symbols, config)
    except Exception as e:
        logger.error(f"Error in trading loop: {e}")


async def nightly_cleanup():
    """Nightly maintenance tasks."""
    try:
        logger.info("Running nightly cleanup...")
        
        # Archive old logs
        from pathlib import Path
        import shutil
        from datetime import timedelta
        
        log_dir = Path(__file__).parent.parent.parent / "logs"
        archive_dir = log_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Archive logs older than 7 days
        cutoff = datetime.now() - timedelta(days=7)
        for log_file in log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff.timestamp():
                shutil.move(str(log_file), str(archive_dir / log_file.name))
                logger.info(f"Archived old log: {log_file.name}")
        
        # Vacuum database
        from app.data.db import get_connection
        with get_connection() as conn:
            conn.execute("VACUUM")
        
        logger.info("Nightly cleanup completed")
    
    except Exception as e:
        logger.error(f"Error in nightly cleanup: {e}")


async def backfill_job():
    """Weekly backfill of historical data."""
    try:
        logger.info("Running weekly backfill...")
        from app.ingestion.history_yf import backfill_all_symbols
        
        symbols = get_universe()
        results = backfill_all_symbols(symbols, period="7d", interval="1m")
        
        logger.info(f"Backfill completed: {results}")
    
    except Exception as e:
        logger.error(f"Error in backfill job: {e}")


async def main():
    """Main entry point for scheduler."""
    logger.info("Starting trading bot scheduler...")
    
    # Schedule trading loop - every minute
    scheduler.add_job(
        trading_loop,
        IntervalTrigger(minutes=1),
        id='trading_loop',
        name='Trading Loop',
        replace_existing=True
    )
    
    # Schedule nightly cleanup - 2 AM daily
    scheduler.add_job(
        nightly_cleanup,
        CronTrigger(hour=2, minute=0),
        id='nightly_cleanup',
        name='Nightly Cleanup',
        replace_existing=True
    )
    
    # Schedule weekly backfill - Sunday 3 AM
    scheduler.add_job(
        backfill_job,
        CronTrigger(day_of_week='sun', hour=3, minute=0),
        id='weekly_backfill',
        name='Weekly Backfill',
        replace_existing=True
    )
    
    # Start scheduler
    scheduler.start()
    logger.info("Scheduler started successfully")
    
    # Start Alpaca stream in parallel
    symbols = get_universe()
    logger.info(f"Starting Alpaca stream for: {', '.join(symbols)}")
    
    await start_stream(symbols)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")

