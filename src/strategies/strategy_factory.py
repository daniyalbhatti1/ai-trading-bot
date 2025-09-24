"""
Strategy factory for creating and managing trading strategies in AI Trading Bot.
"""

from typing import Dict, Any, Optional, List, Type
import logging
from enum import Enum

from .base_strategy import BaseStrategy
from .mean_reversion_strategy import MeanReversionStrategy

class StrategyType(str, Enum):
    """Supported strategy types."""
    MEAN_REVERSION = "mean_reversion"

class StrategyFactory:
    """Factory class for creating and managing trading strategies."""
    
    # Strategy registry
    STRATEGY_REGISTRY = {
        StrategyType.MEAN_REVERSION: MeanReversionStrategy
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategies = {}
        
    def create_strategy(self, strategy_type: str, strategy_config: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """Create a strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            strategy_config: Strategy-specific configuration
            
        Returns:
            Strategy instance
        """
        if strategy_type not in self.STRATEGY_REGISTRY:
            raise ValueError(f"Unsupported strategy type: {strategy_type}. "
                           f"Supported types: {list(self.STRATEGY_REGISTRY.keys())}")
        
        # Merge base config with strategy-specific config
        if strategy_config is None:
            strategy_config = {}
        
        merged_config = {**self.config, **strategy_config}
        
        # Create strategy instance
        strategy_class = self.STRATEGY_REGISTRY[StrategyType(strategy_type)]
        strategy = strategy_class(merged_config)
        
        self.logger.info(f"Created {strategy_type} strategy with config: {merged_config}")
        
        return strategy
    
    def create_strategy_from_config(self, strategy_config: Dict[str, Any]) -> BaseStrategy:
        """Create strategy from configuration dictionary.
        
        Args:
            strategy_config: Strategy configuration with 'type' key
            
        Returns:
            Strategy instance
        """
        if 'type' not in strategy_config:
            raise ValueError("Strategy configuration must contain 'type' key")
        
        strategy_type = strategy_config['type']
        config = {k: v for k, v in strategy_config.items() if k != 'type'}
        
        return self.create_strategy(strategy_type, config)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy types.
        
        Returns:
            List of strategy type names
        """
        return list(self.STRATEGY_REGISTRY.keys())
    
    def get_strategy_info(self, strategy_type: str) -> Dict[str, Any]:
        """Get information about a strategy type.
        
        Args:
            strategy_type: Strategy type name
            
        Returns:
            Dictionary with strategy information
        """
        if strategy_type not in self.STRATEGY_REGISTRY:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
        
        strategy_class = self.STRATEGY_REGISTRY[StrategyType(strategy_type)]
        
        return {
            'name': strategy_type,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'description': strategy_class.__doc__ or "No description available"
        }
    
    def get_recommended_strategies(self, market_condition: str = 'normal') -> List[str]:
        """Get recommended strategies for a specific market condition.
        
        Args:
            market_condition: Market condition ('bull', 'bear', 'sideways', 'volatile')
            
        Returns:
            List of recommended strategy types
        """
        if market_condition == 'bull':
            return [StrategyType.MOMENTUM, StrategyType.ARBITRAGE]
        elif market_condition == 'bear':
            return [StrategyType.MEAN_REVERSION, StrategyType.ARBITRAGE]
        elif market_condition == 'sideways':
            return [StrategyType.MEAN_REVERSION, StrategyType.ARBITRAGE]
        elif market_condition == 'volatile':
            return [StrategyType.ARBITRAGE, StrategyType.MEAN_REVERSION]
        else:
            return list(self.STRATEGY_REGISTRY.keys())
    
    def get_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison of all available strategies.
        
        Returns:
            Dictionary with strategy comparisons
        """
        comparison = {}
        
        for strategy_type in self.STRATEGY_REGISTRY.keys():
            strategy_info = self.get_strategy_info(strategy_type)
            
            # Add strategy characteristics
            characteristics = {
                'market_conditions': self._get_market_conditions(strategy_type),
                'risk_level': self._get_risk_level(strategy_type),
                'capital_requirement': self._get_capital_requirement(strategy_type),
                'time_horizon': self._get_time_horizon(strategy_type),
                'complexity': self._get_complexity(strategy_type)
            }
            
            comparison[strategy_type] = {
                **strategy_info,
                'characteristics': characteristics
            }
        
        return comparison
    
    def _get_market_conditions(self, strategy_type: str) -> List[str]:
        """Get suitable market conditions for a strategy."""
        conditions = {
            StrategyType.MEAN_REVERSION: ['sideways', 'volatile'],
            StrategyType.MOMENTUM: ['bull', 'trending'],
            StrategyType.ARBITRAGE: ['all']
        }
        return conditions.get(strategy_type, ['all'])
    
    def _get_risk_level(self, strategy_type: str) -> str:
        """Get risk level for a strategy."""
        risk_levels = {
            StrategyType.MEAN_REVERSION: 'medium',
            StrategyType.MOMENTUM: 'high',
            StrategyType.ARBITRAGE: 'low'
        }
        return risk_levels.get(strategy_type, 'medium')
    
    def _get_capital_requirement(self, strategy_type: str) -> str:
        """Get capital requirement for a strategy."""
        capital_requirements = {
            StrategyType.MEAN_REVERSION: 'medium',
            StrategyType.MOMENTUM: 'medium',
            StrategyType.ARBITRAGE: 'high'
        }
        return capital_requirements.get(strategy_type, 'medium')
    
    def _get_time_horizon(self, strategy_type: str) -> str:
        """Get time horizon for a strategy."""
        time_horizons = {
            StrategyType.MEAN_REVERSION: 'short',
            StrategyType.MOMENTUM: 'medium',
            StrategyType.ARBITRAGE: 'very_short'
        }
        return time_horizons.get(strategy_type, 'medium')
    
    def _get_complexity(self, strategy_type: str) -> str:
        """Get complexity level for a strategy."""
        complexities = {
            StrategyType.MEAN_REVERSION: 'medium',
            StrategyType.MOMENTUM: 'medium',
            StrategyType.ARBITRAGE: 'high'
        }
        return complexities.get(strategy_type, 'medium')
    
    def validate_strategy_config(self, strategy_config: Dict[str, Any]) -> List[str]:
        """Validate strategy configuration.
        
        Args:
            strategy_config: Strategy configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required keys
        if 'type' not in strategy_config:
            errors.append("Strategy configuration must contain 'type' key")
        
        # Validate strategy type
        if 'type' in strategy_config:
            strategy_type = strategy_config['type']
            if strategy_type not in self.STRATEGY_REGISTRY:
                errors.append(f"Unsupported strategy type: {strategy_type}")
        
        return errors
    
    def get_default_config(self, strategy_type: str) -> Dict[str, Any]:
        """Get default configuration for a strategy type.
        
        Args:
            strategy_type: Strategy type name
            
        Returns:
            Default configuration dictionary
        """
        if strategy_type not in self.STRATEGY_REGISTRY:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
        
        # Default configurations for each strategy type
        default_configs = {
            StrategyType.MEAN_REVERSION: {
                'zscore_threshold': 2.0,
                'zscore_exit': 0.5,
                'lookback_period': 20,
                'bollinger_period': 20,
                'bollinger_std': 2.0,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'hurst_period': 50,
                'hurst_threshold': 0.5
            },
            StrategyType.MOMENTUM: {
                'momentum_period': 10,
                'roc_period': 12,
                'roc_threshold': 0.05,
                'ma_short': 10,
                'ma_long': 30,
                'ma_trend': 50,
                'adx_period': 14,
                'adx_threshold': 25,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            StrategyType.ARBITRAGE: {
                'min_spread': 0.001,
                'max_spread': 0.05,
                'spread_threshold': 0.005,
                'arbitrage_timeout': 300,
                'lookback_period': 100,
                'zscore_threshold': 2.0,
                'zscore_exit': 0.5,
                'cointegration_threshold': 0.05,
                'half_life_max': 30
            }
        }
        
        return default_configs.get(StrategyType(strategy_type), {})
    
    def create_strategy_pipeline(self, strategy_configs: List[Dict[str, Any]]) -> Dict[str, BaseStrategy]:
        """Create multiple strategies from configuration list.
        
        Args:
            strategy_configs: List of strategy configurations
            
        Returns:
            Dictionary mapping strategy names to strategy instances
        """
        strategies = {}
        
        for config in strategy_configs:
            if 'name' not in config:
                raise ValueError("Strategy configuration must contain 'name' key")
            
            strategy_name = config['name']
            strategy_config = {k: v for k, v in config.items() if k != 'name'}
            
            strategies[strategy_name] = self.create_strategy_from_config(strategy_config)
        
        return strategies
