"""Backfill historical data from yfinance."""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.history_yf import backfill_all_symbols, backfill_symbol
from app.ingestion.symbols import get_universe
from app.core.logger import logger


def main():
    """Backfill historical data."""
    parser = argparse.ArgumentParser(description='Backfill historical market data')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backfill (default: all from config)')
    parser.add_argument('--period', default='60d', help='Period to backfill (default: 60d)')
    parser.add_argument('--interval', default='1m', help='Data interval (default: 1m)')
    
    args = parser.parse_args()
    
    symbols = args.symbols if args.symbols else get_universe()
    
    logger.info(f"Backfilling {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"\nBackfilling data for: {', '.join(symbols)}")
    print(f"Period: {args.period}, Interval: {args.interval}\n")
    
    try:
        results = backfill_all_symbols(symbols, args.period, args.interval)
        
        print("\n" + "="*50)
        print("BACKFILL RESULTS")
        print("="*50)
        
        total = 0
        for symbol, count in results.items():
            print(f"  {symbol}: {count:,} candles")
            total += count
        
        print("="*50)
        print(f"Total candles: {total:,}")
        print("✓ Backfill completed successfully!\n")
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

