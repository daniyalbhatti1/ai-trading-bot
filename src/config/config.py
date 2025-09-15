"""
Configuration management for the AI Trading Bot.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field


class TradingConfig(BaseModel):
    """Trading configuration settings."""
    mode: str = "paper"
    base_currency: str = "USDT"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1
    stop_loss: float = 0.02
    take_profit: float = 0.05
    max_daily_trades: int = 50


class DataSourceConfig(BaseModel):
    """Data source configuration."""
    primary: str = "binance"
    backup: str = "alpha_vantage"
    update_interval: int = 60


class ModelConfig(BaseModel):
    """AI model configuration."""
    prediction_model: Dict[str, Any] = Field(default_factory=dict)
    risk_model: Dict[str, Any] = Field(default_factory=dict)


class APIConfig(BaseModel):
    """API configuration."""
    binance: Dict[str, Any] = Field(default_factory=dict)
    alpha_vantage: Dict[str, Any] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    name: str = "trading_bot"
    pool_size: int = 10


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/trading_bot.log"
    max_size: str = "10MB"
    backup_count: int = 5


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enable_metrics: bool = True
    metrics_port: int = 8000
    health_check_interval: int = 30


class SecurityConfig(BaseModel):
    """Security configuration."""
    encrypt_api_keys: bool = True
    session_timeout: int = 3600
    max_login_attempts: int = 3


class Config(BaseModel):
    """Main configuration class."""
    trading: TradingConfig = Field(default_factory=TradingConfig)
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    apis: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object with loaded settings
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Return default configuration if file doesn't exist
        return Config()
    
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    
    return Config(**config_data)
