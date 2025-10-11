"""Alpaca broker integration."""
import ssl
import certifi
from typing import Optional, Dict
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from app.core.settings import settings
from app.core.logger import logger
from app.data.repos import log_order, refresh_positions
from app.trading.risk import calculate_stop_loss, calculate_take_profit

# Fix SSL certificate issues
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED


class Broker:
    """Alpaca broker wrapper."""
    
    def __init__(self):
        """Initialize Alpaca trading client."""
        self.client = TradingClient(
            settings.ALPACA_API_KEY_ID,
            settings.ALPACA_API_SECRET,
            paper=True
        )
        logger.info("Alpaca broker initialized")
    
    def place_with_risk(self, symbol: str, side: str, qty: int, price: float, cfg: dict):
        """Place order with stop loss and take profit.
        
        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            qty: Quantity
            price: Current market price
            cfg: Configuration
        """
        try:
            # Determine order side
            order_side = OrderSide.BUY if side == "LONG" else OrderSide.SELL
            
            # Place main market order
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.client.submit_order(order_data=req)
            
            # Log the order
            log_order({
                'symbol': symbol,
                'side': order_side.value,
                'qty': qty,
                'type': 'market',
                'limit_price': None,
                'stop_price': None,
                'client_order_id': order.client_order_id,
                'status': order.status.value,
                'filled_avg_price': order.filled_avg_price,
                'filled_at': order.filled_at
            })
            
            logger.info(f"Order placed: {side} {qty} {symbol} @ market")
            
            # Calculate stop loss and take profit levels
            stop_price = calculate_stop_loss(price, side, cfg)
            tp_price = calculate_take_profit(price, side, cfg)
            
            logger.info(f"Risk levels - Stop: ${stop_price:.2f}, Target: ${tp_price:.2f}")
            
            # Note: For MVP, we'll monitor these levels in the executor
            # In production, you could place bracket orders if supported
            
            # Refresh positions
            refresh_positions(self.client)
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            raise
    
    def close_position(self, symbol: str, reason: str = ""):
        """Close an existing position.
        
        Args:
            symbol: Symbol to close
            reason: Reason for closing
        """
        try:
            # Get current position
            try:
                position = self.client.get_open_position(symbol)
            except:
                logger.warning(f"No position found for {symbol}")
                return
            
            # Determine close side
            qty = abs(float(position.qty))
            side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY
            
            # Place closing order
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.client.submit_order(order_data=req)
            
            log_order({
                'symbol': symbol,
                'side': side.value,
                'qty': qty,
                'type': 'market',
                'limit_price': None,
                'stop_price': None,
                'client_order_id': order.client_order_id,
                'status': order.status.value,
                'filled_avg_price': order.filled_avg_price,
                'filled_at': order.filled_at
            })
            
            logger.info(f"Position closed: {symbol} - {reason}")
            
            # Refresh positions
            refresh_positions(self.client)
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    def close_partial_position(self, symbol: str, qty_to_close: int, reason: str = ""):
        """Close a partial position.
        
        Args:
            symbol: Symbol to partially close
            qty_to_close: Quantity to close
            reason: Reason for closing
        """
        try:
            # Get current position
            try:
                position = self.client.get_open_position(symbol)
            except:
                logger.warning(f"No position found for {symbol}")
                return
            
            # Validate quantity
            current_qty = abs(float(position.qty))
            if qty_to_close >= current_qty:
                logger.warning(f"Partial close quantity ({qty_to_close}) >= current position ({current_qty}), closing full position instead")
                self.close_position(symbol, reason)
                return
            
            # Determine close side
            side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY
            
            # Place partial closing order
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty_to_close,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.client.submit_order(order_data=req)
            
            log_order({
                'symbol': symbol,
                'side': side.value,
                'qty': qty_to_close,
                'type': 'market',
                'limit_price': None,
                'stop_price': None,
                'client_order_id': order.client_order_id,
                'status': order.status.value,
                'filled_avg_price': order.filled_avg_price,
                'filled_at': order.filled_at
            })
            
            logger.info(f"Partial position closed: {symbol} - {qty_to_close} shares - {reason}")
            
            # Refresh positions
            refresh_positions(self.client)
            
        except Exception as e:
            logger.error(f"Error partially closing position for {symbol}: {e}")
    
    def get_equity(self) -> float:
        """Get current account equity.
        
        Returns:
            Current equity value
        """
        try:
            account = self.client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Error getting equity: {e}")
            return 0.0
    
    def get_buying_power(self) -> float:
        """Get current buying power.
        
        Returns:
            Available buying power
        """
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return 0.0
    
    def get_account_status(self) -> Dict:
        """Get account status summary.
        
        Returns:
            Dictionary with account details
        """
        try:
            account = self.client.get_account()
            positions = self.client.get_all_positions()
            
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'open_positions': len(positions),
                'status': account.status.value
            }
        except Exception as e:
            logger.error(f"Error getting account status: {e}")
            return {}

