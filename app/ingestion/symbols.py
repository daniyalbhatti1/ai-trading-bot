"""Symbol management."""
from typing import List
from app.core.settings import config


def get_universe() -> List[str]:
    """Get the list of symbols to trade."""
    return config.get('universe', ['SPY', 'QQQ', 'GLD', 'USO'])


def get_futures_proxies() -> dict:
    """Get futures proxy mappings."""
    return {
        'ES': 'SPY',   # S&P 500
        'NQ': 'QQQ',   # Nasdaq 100
        'YM': 'DIA',   # Dow Jones
        'GC': 'GLD',   # Gold
        'CL': 'USO',   # Crude Oil
    }

