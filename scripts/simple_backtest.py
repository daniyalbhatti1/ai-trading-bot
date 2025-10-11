#!/usr/bin/env python3
"""Simple backtest script without complex features."""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.settings import config
from app.core.logger import logger
from app.data.repos import get_recent_candles
from app.signals.features import compute_features
from app.signals.rules import rules_signal

def simple_backtest(symbols, days=7):
    """Run a simple backtest with basic metrics."""
    logger.info(f"Running simple backtest for {symbols}")
    
    results = {
        'trades': [],
        'total_return': 0,
        'win_rate': 0,
        'num_trades': 0
    }
    
    initial_capital = 10000
    capital = initial_capital
    positions = {}
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        # Get data
        df = get_recent_candles(symbol, limit=1000)
        if df is None or df.empty:
            logger.warning(f"No data for {symbol}")
            continue
        
        # Compute features
        try:
            df_features = compute_features(df, config)
            if df_features.empty:
                logger.warning(f"No features for {symbol}")
                continue
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            continue
        
        # Simple strategy: RSI mean reversion
        for i in range(50, len(df_features)):
            row = df_features.iloc[i]
            rsi = row.get('rsi', 50)
            close = row.get('close', 0)
            
            # Simple RSI strategy
            if rsi < 30 and symbol not in positions:  # Oversold - buy
                qty = 10
                positions[symbol] = {
                    'side': 'LONG',
                    'entry_price': close,
                    'qty': qty,
                    'entry_time': df_features.index[i]
                }
                logger.info(f"BUY {symbol} at ${close:.2f} (RSI: {rsi:.1f})")
                
            elif rsi > 70 and symbol in positions:  # Overbought - sell
                pos = positions[symbol]
                if pos['side'] == 'LONG':
                    pnl = (close - pos['entry_price']) * pos['qty']
                    capital += pnl
                    
                    results['trades'].append({
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_price': pos['entry_price'],
                        'exit_price': close,
                        'pnl': pnl,
                        'entry_time': pos['entry_time'],
                        'exit_time': df_features.index[i]
                    })
                    
                    logger.info(f"SELL {symbol} at ${close:.2f} (RSI: {rsi:.1f}) - P&L: ${pnl:.2f}")
                    del positions[symbol]
    
    # Calculate metrics
    if results['trades']:
        total_pnl = sum(trade['pnl'] for trade in results['trades'])
        winning_trades = [t for t in results['trades'] if t['pnl'] > 0]
        
        results['total_return'] = (total_pnl / initial_capital) * 100
        results['win_rate'] = (len(winning_trades) / len(results['trades'])) * 100
        results['num_trades'] = len(results['trades'])
    
    return results

def main():
    """Main function."""
    symbols = ['SPY', 'QQQ', 'GLD', 'USO']
    
    print("🔄 Running Simple Backtest...")
    print(f"Symbols: {', '.join(symbols)}")
    print("Strategy: RSI Mean Reversion (RSI < 30 buy, RSI > 70 sell)")
    print("-" * 60)
    
    try:
        results = simple_backtest(symbols)
        
        print(f"\n📊 Backtest Results:")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        
        if results['trades']:
            print(f"\n📈 All Trades ({len(results['trades'])} total):")
            print("-" * 100)
            for i, trade in enumerate(results['trades']):
                pnl_sign = "+" if trade['pnl'] > 0 else ""
                entry_time = trade.get('entry_time', 'N/A')
                exit_time = trade.get('exit_time', 'N/A')
                reason = trade.get('reason', 'N/A')
                print(f"{i+1:3d}. {trade['symbol']} {trade['side']:4s} | "
                      f"Entry: ${trade['entry_price']:7.2f} ({entry_time}) | "
                      f"Exit: ${trade['exit_price']:7.2f} ({exit_time}) | "
                      f"P&L: {pnl_sign}${trade['pnl']:7.2f} | "
                      f"Reason: {reason}")
        else:
            print("\nNo trades executed during backtest period.")
        
        print("\n✅ Simple backtest completed!")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
