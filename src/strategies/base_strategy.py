"""
Base strategy class for all trading strategies in AI Trading Bot.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order data class."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: Optional[datetime] = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0

@dataclass
class Position:
    """Position data class."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    timestamp: Optional[datetime] = None

@dataclass
class Signal:
    """Trading signal data class."""
    symbol: str
    side: OrderSide
    strength: float  # Signal strength from 0 to 1
    price: float
    quantity: float
    confidence: float  # Confidence level from 0 to 1
    timestamp: datetime
    strategy_name: str
    metadata: Optional[Dict[str, Any]] = None

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base strategy.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Strategy parameters
        self.name = config.get('name', self.__class__.__name__)
        self.symbols = config.get('symbols', [])
        self.timeframe = config.get('timeframe', '1h')
        self.lookback_period = config.get('lookback_period', 100)
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)
        self.stop_loss = config.get('stop_loss', 0.02)
        self.take_profit = config.get('take_profit', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        
        # Trading parameters
        self.min_signal_strength = config.get('min_signal_strength', 0.6)
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_daily_trades = config.get('max_daily_trades', 10)
        
        # State tracking
        self.positions = {}
        self.orders = []
        self.signals = []
        self.performance_metrics = {}
        self.is_active = False
        self.daily_trades = 0
        self.last_trade_date = None
        
        # Performance tracking
        self.initial_capital = config.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown_reached = 0.0
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_capital: float) -> float:
        """Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            available_capital: Available capital for trading
            
        Returns:
            Position size
        """
        pass
    
    def validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Check signal strength
        if signal.strength < self.min_signal_strength:
            return False
        
        # Check confidence level
        if signal.confidence < self.min_confidence:
            return False
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            self.logger.warning(f"Daily trade limit reached: {self.daily_trades}")
            return False
        
        # Check if symbol is in allowed list
        if self.symbols and signal.symbol not in self.symbols:
            return False
        
        return True
    
    def create_order(self, signal: Signal, current_price: float, 
                    available_capital: float) -> Optional[Order]:
        """Create an order from a signal.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            available_capital: Available capital
            
        Returns:
            Order object or None if order cannot be created
        """
        if not self.validate_signal(signal):
            return None
        
        # Calculate position size
        quantity = self.calculate_position_size(signal, current_price, available_capital)
        
        if quantity <= 0:
            return None
        
        # Create order
        order = Order(
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=signal.price,
            timestamp=datetime.now(),
            order_id=f"{self.name}_{len(self.orders)}_{datetime.now().timestamp()}"
        )
        
        self.orders.append(order)
        self.daily_trades += 1
        self.total_trades += 1
        
        self.logger.info(f"Created order: {order.symbol} {order.side} {order.quantity} @ {order.price}")
        
        return order
    
    def update_position(self, symbol: str, quantity: float, price: float, 
                       side: OrderSide) -> None:
        """Update position after order execution.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Execution price
            side: Order side
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                timestamp=datetime.now()
            )
        
        position = self.positions[symbol]
        
        if side == OrderSide.BUY:
            # Add to position
            total_value = (position.quantity * position.average_price) + (quantity * price)
            total_quantity = position.quantity + quantity
            
            if total_quantity > 0:
                position.average_price = total_value / total_quantity
                position.quantity = total_quantity
        else:
            # Reduce position
            position.quantity -= quantity
            
            # Calculate realized P&L
            if position.quantity <= 0:
                realized_pnl = (price - position.average_price) * (position.quantity + quantity)
                position.realized_pnl += realized_pnl
                position.quantity = 0.0
                position.average_price = 0.0
        
        position.timestamp = datetime.now()
        
        self.logger.info(f"Updated position {symbol}: quantity={position.quantity}, "
                        f"avg_price={position.average_price:.4f}")
    
    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L for a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        if position.quantity == 0:
            return 0.0
        
        unrealized_pnl = (current_price - position.average_price) * position.quantity
        position.unrealized_pnl = unrealized_pnl
        position.market_value = current_price * position.quantity
        
        return unrealized_pnl
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss should be triggered.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            True if stop loss should be triggered
        """
        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            return False
        
        position = self.positions[symbol]
        price_change = (current_price - position.average_price) / position.average_price
        
        return price_change <= -self.stop_loss
    
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if take profit should be triggered.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            True if take profit should be triggered
        """
        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            return False
        
        position = self.positions[symbol]
        price_change = (current_price - position.average_price) / position.average_price
        
        return price_change >= self.take_profit
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.
        
        Args:
            current_prices: Dictionary of current prices for all symbols
            
        Returns:
            Total portfolio value
        """
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if position.quantity > 0 and symbol in current_prices:
                market_value = position.quantity * current_prices[symbol]
                total_value += market_value
        
        return total_value
    
    def calculate_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            current_prices: Dictionary of current prices for all symbols
            
        Returns:
            Dictionary of performance metrics
        """
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        # Calculate returns
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate realized P&L
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        # Calculate unrealized P&L
        total_unrealized_pnl = sum(
            self.calculate_unrealized_pnl(symbol, current_prices.get(symbol, 0))
            for symbol in self.positions.keys()
        )
        
        # Calculate win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate average win/loss
        avg_win = self.winning_trades / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.losing_trades / self.losing_trades if self.losing_trades > 0 else 0
        
        # Calculate profit factor
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = total_return / 0.1 if total_return > 0 else 0  # Assuming 10% volatility
        
        # Calculate maximum drawdown
        if portfolio_value > self.initial_capital:
            current_drawdown = 0
        else:
            current_drawdown = (self.initial_capital - portfolio_value) / self.initial_capital
        
        self.max_drawdown_reached = max(self.max_drawdown_reached, current_drawdown)
        
        self.performance_metrics = {
            'total_return': total_return,
            'portfolio_value': portfolio_value,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown_reached,
            'current_drawdown': current_drawdown
        }
        
        return self.performance_metrics
    
    def reset_daily_trades(self) -> None:
        """Reset daily trade counter."""
        self.daily_trades = 0
        self.last_trade_date = datetime.now().date()
    
    def start(self) -> None:
        """Start the strategy."""
        self.is_active = True
        self.logger.info(f"Started strategy: {self.name}")
    
    def stop(self) -> None:
        """Stop the strategy."""
        self.is_active = False
        self.logger.info(f"Stopped strategy: {self.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status.
        
        Returns:
            Dictionary with strategy status information
        """
        return {
            'name': self.name,
            'is_active': self.is_active,
            'symbols': self.symbols,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'average_price': pos.average_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl
            } for symbol, pos in self.positions.items()},
            'total_trades': self.total_trades,
            'daily_trades': self.daily_trades,
            'performance_metrics': self.performance_metrics
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        return {
            'strategy_name': self.name,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return': self.performance_metrics.get('total_return', 0),
            'total_trades': self.total_trades,
            'win_rate': self.performance_metrics.get('win_rate', 0),
            'profit_factor': self.performance_metrics.get('profit_factor', 0),
            'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
            'max_drawdown': self.performance_metrics.get('max_drawdown', 0)
        }
