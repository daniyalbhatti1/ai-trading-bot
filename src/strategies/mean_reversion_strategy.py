"""
Mean Reversion Trading Strategy for AI Trading Bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging

from .base_strategy import BaseStrategy, Signal, OrderSide

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy using statistical indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mean reversion strategy.
        
        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        
        # Mean reversion specific parameters
        self.zscore_threshold = config.get('zscore_threshold', 2.0)
        self.zscore_exit = config.get('zscore_exit', 0.5)
        self.lookback_period = config.get('lookback_period', 20)
        self.bollinger_period = config.get('bollinger_period', 20)
        self.bollinger_std = config.get('bollinger_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.hurst_period = config.get('hurst_period', 50)
        self.hurst_threshold = config.get('hurst_threshold', 0.5)
        
        # Advanced parameters
        self.use_cointegration = config.get('use_cointegration', True)
        self.use_half_life = config.get('use_half_life', True)
        self.use_volume_confirmation = config.get('use_volume_confirmation', True)
        self.use_volatility_filter = config.get('use_volatility_filter', True)
        
        # Cointegration parameters
        self.cointegration_lookback = config.get('cointegration_lookback', 100)
        self.cointegration_threshold = config.get('cointegration_threshold', 0.05)
        
        # Half-life parameters
        self.half_life_min = config.get('half_life_min', 1)
        self.half_life_max = config.get('half_life_max', 50)
        
    def calculate_zscore(self, prices: pd.Series, window: int = None) -> pd.Series:
        """Calculate Z-score for mean reversion.
        
        Args:
            prices: Price series
            window: Rolling window size
            
        Returns:
            Z-score series
        """
        if window is None:
            window = self.lookback_period
        
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        return (prices - rolling_mean) / rolling_std
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        rolling_mean = prices.rolling(window=self.bollinger_period).mean()
        rolling_std = prices.rolling(window=self.bollinger_period).std()
        
        return {
            'upper': rolling_mean + (rolling_std * self.bollinger_std),
            'middle': rolling_mean,
            'lower': rolling_mean - (rolling_std * self.bollinger_std)
        }
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent to measure mean reversion tendency.
        
        Args:
            prices: Price series
            
        Returns:
            Hurst exponent
        """
        if len(prices) < self.hurst_period:
            return 0.5
        
        # Use log returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        if len(returns) < self.hurst_period:
            return 0.5
        
        # Calculate Hurst exponent using R/S analysis
        lags = range(2, min(len(returns) // 4, 50))
        tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
        
        # Fit linear regression
        if len(tau) < 2:
            return 0.5
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
        
        return max(0, min(1, hurst))
    
    def calculate_half_life(self, prices: pd.Series) -> float:
        """Calculate half-life of mean reversion.
        
        Args:
            prices: Price series
            
        Returns:
            Half-life in periods
        """
        if len(prices) < 10:
            return 1
        
        # Calculate price differences
        price_diff = prices.diff().dropna()
        price_lag = prices.shift(1).dropna()
        
        # Align series
        min_len = min(len(price_diff), len(price_lag))
        price_diff = price_diff.iloc[-min_len:]
        price_lag = price_lag.iloc[-min_len:]
        
        if len(price_diff) < 5:
            return 1
        
        # Fit linear regression: price_diff = alpha + beta * price_lag
        X = price_lag.values.reshape(-1, 1)
        y = price_diff.values
        
        try:
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            
            if beta >= 0:
                return 1
            
            half_life = -np.log(2) / np.log(1 + beta)
            return max(self.half_life_min, min(self.half_life_max, half_life))
        
        except:
            return 1
    
    def check_cointegration(self, price1: pd.Series, price2: pd.Series) -> Dict[str, Any]:
        """Check cointegration between two price series.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Cointegration test results
        """
        if len(price1) < self.cointegration_lookback or len(price2) < self.cointegration_lookback:
            return {'is_cointegrated': False, 'p_value': 1.0}
        
        # Align series
        min_len = min(len(price1), len(price2))
        price1 = price1.iloc[-min_len:]
        price2 = price2.iloc[-min_len:]
        
        # Perform cointegration test
        try:
            from statsmodels.tsa.stattools import coint
            score, p_value, _ = coint(price1, price2)
            
            return {
                'is_cointegrated': p_value < self.cointegration_threshold,
                'p_value': p_value,
                'score': score
            }
        except:
            return {'is_cointegrated': False, 'p_value': 1.0}
    
    def calculate_volume_confirmation(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Calculate volume confirmation for mean reversion signals.
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            Volume confirmation signal
        """
        # Volume moving average
        volume_ma = volumes.rolling(window=20).mean()
        
        # Price change
        price_change = prices.pct_change()
        
        # Volume confirmation: high volume on price moves away from mean
        volume_confirmation = (volumes / volume_ma) * np.abs(price_change)
        
        return volume_confirmation
    
    def calculate_volatility_filter(self, prices: pd.Series) -> pd.Series:
        """Calculate volatility filter to avoid trading in low volatility periods.
        
        Args:
            prices: Price series
            
        Returns:
            Volatility filter signal
        """
        returns = prices.pct_change()
        volatility = returns.rolling(window=20).std()
        volatility_ma = volatility.rolling(window=50).mean()
        
        # Only trade when volatility is above average
        return volatility > volatility_ma
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate mean reversion trading signals.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if len(data) < self.lookback_period:
            return signals
        
        for symbol in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else [data.columns.name or 'close']:
            try:
                # Extract price data
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data[symbol]['close']
                    volumes = data[symbol]['volume'] if 'volume' in data[symbol].columns else None
                else:
                    prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
                    volumes = data['volume'] if 'volume' in data.columns else None
                
                if len(prices) < self.lookback_period:
                    continue
                
                # Calculate indicators
                zscore = self.calculate_zscore(prices)
                bollinger = self.calculate_bollinger_bands(prices)
                rsi = self.calculate_rsi(prices)
                hurst = self.calculate_hurst_exponent(prices)
                half_life = self.calculate_half_life(prices)
                
                # Get latest values
                current_price = prices.iloc[-1]
                current_zscore = zscore.iloc[-1]
                current_rsi = rsi.iloc[-1]
                current_bb_position = (current_price - bollinger['lower'].iloc[-1]) / (bollinger['upper'].iloc[-1] - bollinger['lower'].iloc[-1])
                
                # Mean reversion conditions
                is_oversold = (current_zscore < -self.zscore_threshold or 
                             current_bb_position < 0.1 or 
                             current_rsi < self.rsi_oversold)
                
                is_overbought = (current_zscore > self.zscore_threshold or 
                               current_bb_position > 0.9 or 
                               current_rsi > self.rsi_overbought)
                
                # Hurst exponent filter (mean reversion when H < 0.5)
                is_mean_reverting = hurst < self.hurst_threshold
                
                # Half-life filter
                is_fast_mean_reversion = self.half_life_min <= half_life <= self.half_life_max
                
                # Volume confirmation
                volume_confirmation = True
                if self.use_volume_confirmation and volumes is not None:
                    volume_signal = self.calculate_volume_confirmation(prices, volumes)
                    volume_confirmation = volume_signal.iloc[-1] > volume_signal.rolling(window=20).mean().iloc[-1]
                
                # Volatility filter
                volatility_ok = True
                if self.use_volatility_filter:
                    volatility_signal = self.calculate_volatility_filter(prices)
                    volatility_ok = volatility_signal.iloc[-1]
                
                # Generate buy signal (oversold condition)
                if (is_oversold and is_mean_reverting and is_fast_mean_reversion and 
                    volume_confirmation and volatility_ok):
                    
                    signal_strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                    confidence = min(1.0, (1 - hurst) * (1 - current_bb_position) * (1 - current_rsi / 100))
                    
                    signal = Signal(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        strength=signal_strength,
                        price=current_price,
                        quantity=0,  # Will be calculated later
                        confidence=confidence,
                        timestamp=pd.Timestamp.now(),
                        strategy_name=self.name,
                        metadata={
                            'zscore': current_zscore,
                            'rsi': current_rsi,
                            'hurst': hurst,
                            'half_life': half_life,
                            'bb_position': current_bb_position
                        }
                    )
                    signals.append(signal)
                
                # Generate sell signal (overbought condition)
                elif (is_overbought and is_mean_reverting and is_fast_mean_reversion and 
                      volume_confirmation and volatility_ok):
                    
                    signal_strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                    confidence = min(1.0, (1 - hurst) * current_bb_position * (current_rsi / 100))
                    
                    signal = Signal(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        strength=signal_strength,
                        price=current_price,
                        quantity=0,  # Will be calculated later
                        confidence=confidence,
                        timestamp=pd.Timestamp.now(),
                        strategy_name=self.name,
                        metadata={
                            'zscore': current_zscore,
                            'rsi': current_rsi,
                            'hurst': hurst,
                            'half_life': half_life,
                            'bb_position': current_bb_position
                        }
                    )
                    signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}: {e}")
                continue
        
        return signals
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_capital: float) -> float:
        """Calculate position size for mean reversion strategy.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            available_capital: Available capital for trading
            
        Returns:
            Position size
        """
        # Base position size
        base_size = available_capital * self.max_position_size / current_price
        
        # Adjust based on signal strength and confidence
        strength_multiplier = signal.strength * signal.confidence
        
        # Adjust based on half-life (faster mean reversion = larger position)
        half_life = signal.metadata.get('half_life', 1)
        half_life_multiplier = max(0.5, min(2.0, 10 / half_life))
        
        # Adjust based on Z-score (stronger deviation = larger position)
        zscore = abs(signal.metadata.get('zscore', 0))
        zscore_multiplier = min(2.0, zscore / self.zscore_threshold)
        
        # Final position size
        position_size = base_size * strength_multiplier * half_life_multiplier * zscore_multiplier
        
        return max(0, position_size)
    
    def should_exit_position(self, symbol: str, current_price: float, 
                           current_data: pd.DataFrame) -> bool:
        """Check if position should be exited based on mean reversion criteria.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_data: Current market data
            
        Returns:
            True if position should be exited
        """
        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            return False
        
        try:
            # Extract price data
            if isinstance(current_data.columns, pd.MultiIndex):
                prices = current_data[symbol]['close']
            else:
                prices = current_data['close'] if 'close' in current_data.columns else current_data.iloc[:, 0]
            
            if len(prices) < self.lookback_period:
                return False
            
            # Calculate current Z-score
            zscore = self.calculate_zscore(prices)
            current_zscore = zscore.iloc[-1]
            
            # Exit if Z-score returns to neutral zone
            if abs(current_zscore) < self.zscore_exit:
                return True
            
            # Exit if position is profitable and Z-score is reversing
            position = self.positions[symbol]
            if position.quantity > 0:  # Long position
                if current_price > position.average_price and current_zscore > 0:
                    return True
            else:  # Short position
                if current_price < position.average_price and current_zscore < 0:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit condition for {symbol}: {e}")
            return False
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return {
            'zscore_threshold': self.zscore_threshold,
            'zscore_exit': self.zscore_exit,
            'lookback_period': self.lookback_period,
            'bollinger_period': self.bollinger_period,
            'bollinger_std': self.bollinger_std,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'hurst_period': self.hurst_period,
            'hurst_threshold': self.hurst_threshold,
            'use_cointegration': self.use_cointegration,
            'use_half_life': self.use_half_life,
            'use_volume_confirmation': self.use_volume_confirmation,
            'use_volatility_filter': self.use_volatility_filter
        }
