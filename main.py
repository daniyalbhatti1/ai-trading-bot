import asyncio
import sys
import os
import ssl
import certifi
from pathlib import Path

# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.jobs.scheduler import main as scheduler_main
from app.core.logger import logger


async def main():
    """Main entry point."""
    logger.info("🤖 Starting Algorithmic Trading Bot...")
    logger.info("📈 Pure technical analysis strategy (no AI/LLM)")
    logger.info("🎯 Trading: SPY, QQQ, GLD, USO")
    logger.info("⚡ Paper trading mode")
    
    try:
        await scheduler_main()
    except KeyboardInterrupt:
        logger.info("🛑 Trading bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Trading bot error: {e}")
        raise


if __name__ == "__main__":
    print("🚀 Algorithmic Trading Bot")
    print("=" * 50)
    print("📊 Strategy: Mean Reversion + Trend Following")
    print("🎯 Symbols: SPY, QQQ, GLD, USO")
    print("⚡ Mode: Paper Trading")
    print("📱 Dashboard: streamlit run app/dashboard/app.py")
    print("=" * 50)
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
