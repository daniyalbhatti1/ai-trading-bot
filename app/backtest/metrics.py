"""Backtest metrics calculation."""
import pandas as pd
import numpy as np
from typing import Dict


def calculate_backtest_metrics(results: Dict, initial_capital: float) -> Dict:
    """Calculate comprehensive backtest metrics.
    
    Args:
        results: Backtest results dictionary
        initial_capital: Starting capital
    
    Returns:
        Dictionary of metrics
    """
    trades = results.get('trades', [])
    equity_curve = results.get('equity_curve', [])
    
    if not trades and not equity_curve:
        return {}
    
    metrics = {}
    
    # Trade-based metrics
    if trades:
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        metrics['total_trades'] = total_trades
        metrics['winning_trades'] = winning_trades
        metrics['losing_trades'] = losing_trades
        metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        metrics['total_pnl'] = total_pnl
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        metrics['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Equity-based metrics
    if equity_curve:
        df_equity = pd.DataFrame(equity_curve)
        
        final_equity = df_equity['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        metrics['initial_capital'] = initial_capital
        metrics['final_equity'] = final_equity
        metrics['total_return'] = total_return
        metrics['total_return_pct'] = total_return * 100
        
        # Calculate returns
        df_equity['returns'] = df_equity['equity'].pct_change()
        
        # Sharpe ratio (annualized, assuming daily data)
        returns = df_equity['returns'].dropna()
        if len(returns) > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            metrics['sharpe_ratio'] = sharpe
        
        # Maximum drawdown
        df_equity['cummax'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['cummax']) / df_equity['cummax']
        max_dd = df_equity['drawdown'].min()
        
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_pct'] = max_dd * 100
    
    return metrics


def format_metrics(metrics: Dict) -> str:
    """Format metrics for display.
    
    Args:
        metrics: Metrics dictionary
    
    Returns:
        Formatted string
    """
    lines = ["=" * 50]
    lines.append("BACKTEST RESULTS")
    lines.append("=" * 50)
    
    if 'total_trades' in metrics:
        lines.append(f"\nTrading Performance:")
        lines.append(f"  Total Trades: {metrics['total_trades']}")
        lines.append(f"  Win Rate: {metrics['win_rate']:.2%}")
        lines.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    if 'total_return' in metrics:
        lines.append(f"\nReturns:")
        lines.append(f"  Initial Capital: ${metrics['initial_capital']:,.2f}")
        lines.append(f"  Final Equity: ${metrics['final_equity']:,.2f}")
        lines.append(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    
    if 'sharpe_ratio' in metrics:
        lines.append(f"\nRisk Metrics:")
        lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        lines.append(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    
    if 'total_pnl' in metrics:
        lines.append(f"\nP&L:")
        lines.append(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
        lines.append(f"  Avg Win: ${metrics['avg_win']:,.2f}")
        lines.append(f"  Avg Loss: ${metrics['avg_loss']:,.2f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)

