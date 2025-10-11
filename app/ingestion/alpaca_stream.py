"""Alpaca streaming for real-time market data."""
import asyncio
import ssl
import certifi
from datetime import datetime
from alpaca.data.live import StockDataStream
from app.core.settings import settings
from app.core.logger import logger
from app.data.db import upsert_candle

# Fix SSL certificate issues
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED


class AlpacaStreamer:
    """Alpaca data streaming handler."""
    
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.stream = StockDataStream(
            settings.ALPACA_API_KEY_ID,
            settings.ALPACA_API_SECRET
        )
        
    async def on_bar(self, bar):
        """Handle incoming bar data."""
        try:
            upsert_candle(
                ts=bar.timestamp.isoformat(),
                symbol=bar.symbol,
                o=float(bar.open),
                h=float(bar.high),
                l=float(bar.low),
                c=float(bar.close),
                v=float(bar.volume)
            )
            logger.debug(f"Bar received: {bar.symbol} @ {bar.timestamp} - Close: ${bar.close:.2f}")
        except Exception as e:
            logger.error(f"Error processing bar for {bar.symbol}: {e}")
    
    async def run(self):
        """Start streaming bars."""
        try:
            # Subscribe to minute bars for all symbols
            for symbol in self.symbols:
                self.stream.subscribe_bars(self.on_bar, symbol)
            
            logger.info(f"Starting Alpaca stream for symbols: {', '.join(self.symbols)}")
            
            # Try to connect with retry logic
            max_retries = 3
            retry_delay = 10
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to connect to Alpaca stream (attempt {attempt + 1}/{max_retries})")
                    await self.stream._run_forever()
                    break
                except Exception as e:
                    logger.warning(f"Stream connection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} connection attempts failed")
                        logger.info("Continuing without live data stream...")
                        return  # Don't raise - let the bot continue
            
        except Exception as e:
            logger.error(f"Alpaca stream error: {e}")
            logger.info("Continuing without live data stream...")
            # Don't raise - let the bot continue without live data


async def start_stream(symbols: list):
    """Start the Alpaca streaming service."""
    streamer = AlpacaStreamer(symbols)
    await streamer.run()


if __name__ == "__main__":
    from app.ingestion.symbols import get_universe
    symbols = get_universe()
    asyncio.run(start_stream(symbols))

