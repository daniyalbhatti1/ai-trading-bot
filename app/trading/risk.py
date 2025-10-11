"""Risk management and position sizing."""
from typing import Optional
from app.core.logger import logger
from app.data.repos import get_open_positions


def within_risk_limits(symbol: str, cfg: dict) -> bool:
    """Check if we can open a new position within risk limits.
    
    Args:
        symbol: Symbol to trade
        cfg: Configuration dictionary
    
    Returns:
        True if within risk limits, False otherwise
    """
    try:
        # Get current positions
        positions_df = get_open_positions()
        
        # Check max positions limit
        max_positions = cfg['risk']['max_positions']
        current_positions = len(positions_df)
        
        if current_positions >= max_positions:
            logger.warning(f"Max positions ({max_positions}) reached, cannot open new position for {symbol}")
            return False
        
        # Check if we already have a position in this symbol
        if symbol in positions_df['symbol'].values:
            logger.info(f"Already have position in {symbol}, skipping")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking risk limits: {e}")
        return False


def size_order(symbol: str, cfg: dict, price: float) -> int:
    """Calculate position size.
    
    Args:
        symbol: Symbol to trade
        cfg: Configuration
        price: Current price
    
    Returns:
        Quantity to trade
    """
    try:
        # Simple fixed quantity for MVP
        qty = cfg['sizing']['fixed_qty']
        
        # Alternative: Calculate based on dollar risk
        # dollar_risk = cfg['risk']['dollar_risk_per_trade']
        # stop_loss_pct = cfg['risk']['stop_loss_pct']
        # risk_per_share = price * stop_loss_pct
        # qty = int(dollar_risk / risk_per_share)
        
        logger.debug(f"Position size for {symbol}: {qty} shares")
        
        return qty
        
    except Exception as e:
        logger.error(f"Error sizing order: {e}")
        return 0


def calculate_stop_loss(entry_price: float, side: str, cfg: dict) -> float:
    """Calculate stop loss price.
    
    Args:
        entry_price: Entry price
        side: LONG or SHORT
        cfg: Configuration
    
    Returns:
        Stop loss price
    """
    stop_loss_pct = cfg['risk']['stop_loss_pct']
    
    if side == "LONG":
        stop_price = entry_price * (1 - stop_loss_pct)
    else:  # SHORT
        stop_price = entry_price * (1 + stop_loss_pct)
    
    return round(stop_price, 2)


def calculate_take_profit(entry_price: float, side: str, cfg: dict) -> float:
    """Calculate take profit price.
    
    Args:
        entry_price: Entry price
        side: LONG or SHORT
        cfg: Configuration
    
    Returns:
        Take profit price
    """
    take_profit_pct = cfg['risk']['take_profit_pct']
    
    if side == "LONG":
        tp_price = entry_price * (1 + take_profit_pct)
    else:  # SHORT
        tp_price = entry_price * (1 - take_profit_pct)
    
    return round(tp_price, 2)

