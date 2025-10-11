"""Utility functions."""
from datetime import datetime, timezone
import pytz


def now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def now_et() -> datetime:
    """Return current Eastern Time datetime."""
    et_tz = pytz.timezone('US/Eastern')
    return datetime.now(et_tz)


def is_market_hours() -> bool:
    """Check if current time is within market hours (9:35 AM - 3:55 PM ET)."""
    et_time = now_et()
    
    # Check if weekday
    if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check time range (9:35 AM to 3:55 PM)
    market_open = et_time.replace(hour=9, minute=35, second=0, microsecond=0)
    market_close = et_time.replace(hour=15, minute=55, second=0, microsecond=0)
    
    return market_open <= et_time <= market_close


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2%}"

