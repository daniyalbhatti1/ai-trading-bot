"""Learning module for trade analysis and model improvement."""
from app.learning.trade_journal import (
    log_trade_to_journal,
    get_trade_history,
    get_pattern_performance,
    analyze_trade_quality,
    analyze_all_unanalyzed_trades
)

__all__ = [
    'log_trade_to_journal',
    'get_trade_history',
    'get_pattern_performance',
    'analyze_trade_quality',
    'analyze_all_unanalyzed_trades'
]

