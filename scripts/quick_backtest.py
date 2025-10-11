"""Quick backtest script."""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backtest.engine import run_backtest
from app.backtest.metrics import format_metrics
from app.ingestion.symbols import get_universe
from app.core.settings import config
from app.core.logger import logger


def main():
    """Run a quick backtest."""
    parser = argparse.ArgumentParser(description='Run backtest on historical data')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest (default: all from config)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    symbols = args.symbols if args.symbols else get_universe()
    
    logger.info(f"Running backtest for {len(symbols)} symbols")
    print(f"\n🔄 Running backtest for: {', '.join(symbols)}\n")
    
    try:
        # Run backtest with current config
        results = run_backtest(
            symbols=symbols,
            cfg=config,
            start_date=args.start,
            end_date=args.end
        )
        
        # Display results
        if results.get('metrics'):
            print(format_metrics(results['metrics']))
        
        # Show all trades
        if results.get('trades'):
            trades = results['trades']
            print(f"\nAll Trades ({len(trades)} total):")
            print("-" * 100)
            for i, trade in enumerate(trades, 1):
                pnl_sign = "+" if trade['pnl'] > 0 else ""
                entry_time = trade.get('entry_time', 'N/A')
                exit_time = trade.get('exit_time', 'N/A')
                reason = trade.get('reason', 'N/A')
                print(f"{i:3d}. {trade['symbol']} {trade['side']:4s} | "
                      f"Entry: ${trade['entry_price']:7.2f} ({entry_time}) | "
                      f"Exit: ${trade['exit_price']:7.2f} ({exit_time}) | "
                      f"P&L: {pnl_sign}${trade['pnl']:7.2f} | "
                      f"Reason: {reason}")
        else:
            print("\nNo trades executed during backtest period.")
        
        print("\n✓ Backtest completed!\n")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

