#!/usr/bin/env python3
"""Start the complete trading bot system."""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.logger import logger


def start_api_server():
    """Start the FastAPI server in a subprocess."""
    try:
        logger.info("🚀 Starting API server...")
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.api.server:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        return process
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return None


async def start_trading_bot():
    """Start the trading bot scheduler."""
    try:
        logger.info("🤖 Starting trading bot...")
        from app.jobs.scheduler import main as scheduler_main
        await scheduler_main()
    except Exception as e:
        logger.error(f"Trading bot error: {e}")


async def main():
    """Main entry point."""
    print("🚀 Algorithmic Trading Bot System")
    print("=" * 50)
    print("📊 Strategy: Mean Reversion + Trend Following")
    print("🎯 Symbols: SPY, QQQ, GLD, USO")
    print("⚡ Mode: Paper Trading")
    print("🌐 API: http://localhost:8000")
    print("📱 Dashboard: streamlit run app/dashboard/app.py")
    print("=" * 50)
    print()
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("❌ Failed to start API server")
        return
    
    # Wait a moment for API to start
    await asyncio.sleep(2)
    
    try:
        # Start trading bot
        await start_trading_bot()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        # Clean up API process
        if api_process:
            api_process.terminate()
            api_process.wait()
        logger.info("👋 Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
