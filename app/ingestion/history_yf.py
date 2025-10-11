"""Historical data backfill using yfinance."""
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List
from app.core.logger import logger
from app.data.db import upsert_candle
from app.core.settings import config


def backfill_symbol(symbol: str, period: str = "60d", interval: str = "1m") -> int:
    """Backfill historical data for a symbol.
    
    Args:
        symbol: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        Number of candles inserted
    """
    try:
        logger.info(f"Backfilling {symbol} - period: {period}, interval: {interval}")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return 0
        
        count = 0
        for idx, row in df.iterrows():
            upsert_candle(
                ts=idx.isoformat(),
                symbol=symbol,
                o=float(row['Open']),
                h=float(row['High']),
                l=float(row['Low']),
                c=float(row['Close']),
                v=float(row['Volume'])
            )
            count += 1
        
        logger.info(f"Backfilled {count} candles for {symbol}")
        return count
        
    except Exception as e:
        logger.error(f"Error backfilling {symbol}: {e}")
        return 0


def backfill_all_symbols(symbols: List[str] = None, period: str = None, interval: str = None) -> dict:
    """Backfill all symbols in the universe.
    
    Returns:
        Dictionary with symbol -> count mapping
    """
    if symbols is None:
        symbols = config.get('universe', ['SPY', 'QQQ', 'GLD', 'USO'])
    
    if period is None:
        period = config['bars']['history_period']
    
    if interval is None:
        interval = config['bars']['history_interval']
    
    results = {}
    
    for symbol in symbols:
        count = backfill_symbol(symbol, period, interval)
        results[symbol] = count
    
    total = sum(results.values())
    logger.info(f"Total candles backfilled: {total}")
    
    return results


if __name__ == "__main__":
    backfill_all_symbols()

