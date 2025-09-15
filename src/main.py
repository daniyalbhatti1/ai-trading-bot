"""
AI Trading Bot - Main Entry Point

This is the main entry point for the AI trading bot application.
"""

import asyncio
import logging
from pathlib import Path

from loguru import logger

from config.config import load_config
from utils.logger import setup_logging


async def main():
    """Main function to run the trading bot."""
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config()
    
    logger.info("Starting AI Trading Bot...")
    logger.info(f"Trading mode: {config.trading.mode}")
    logger.info(f"Initial balance: {config.trading.initial_balance}")
    
    # TODO: Initialize trading bot components
    # - Data fetcher
    # - AI models
    # - Trading strategy
    # - Risk manager
    # - Portfolio manager
    
    logger.info("AI Trading Bot started successfully!")
    
    # Keep the bot running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down AI Trading Bot...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
