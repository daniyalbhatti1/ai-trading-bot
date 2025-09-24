"""
Advanced configuration management for the AI Trading Bot.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from enum import Enum
import logging

class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"

class ModelType(str, Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"

class RiskModelType(str, Enum):
    VAR = "var"
    CVAR = "cvar"
    MONTE_CARLO = "monte_carlo"
    GARCH = "garch"
    COPULA = "copula"

class DataSource(str, Enum):
    BINANCE = "binance"
    ALPACA = "alpaca"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO = "yahoo"
    COINBASE = "coinbase"

class TradingConfig(BaseModel):
    """Advanced trading configuration settings."""
    mode: TradingMode = TradingMode.PAPER
    base_currency: str = "USDT"
    initial_balance: float = Field(10000, ge=0)
    max_position_size: float = Field(0.1, ge=0, le=1)
    stop_loss: float = Field(0.02, ge=0, le=1)
    take_profit: float = Field(0.05, ge=0, le=1)
    max_daily_trades: int = Field(50, ge=1)
    max_drawdown: float = Field(0.15, ge=0, le=1)
    risk_free_rate: float = Field(0.02, ge=0)
    
    @validator('take_profit')
    def take_profit_must_be_greater_than_stop_loss(cls, v, values):
        if 'stop_loss' in values and v <= values['stop_loss']:
            raise ValueError('take_profit must be greater than stop_loss')
        return v

class DataSourcesConfig(BaseModel):
    """Data sources configuration."""
    primary: DataSource = DataSource.BINANCE
    backup: DataSource = DataSource.ALPHA_VANTAGE
    update_interval: int = Field(60, ge=1)
    historical_days: int = Field(365, ge=1)
    enable_websocket: bool = True
    enable_fundamental_data: bool = True
    enable_sentiment_data: bool = True

class ModelConfig(BaseModel):
    """AI model configuration."""
    type: ModelType = ModelType.LSTM
    lookback_window: int = Field(100, ge=1)
    prediction_horizon: int = Field(24, ge=1)
    retrain_frequency: int = Field(24, ge=1)
    ensemble_models: List[ModelType] = [ModelType.LSTM, ModelType.XGBOOST]
    feature_engineering: bool = True
    hyperparameter_optimization: bool = True
    cross_validation_folds: int = Field(5, ge=2)

class RiskModelConfig(BaseModel):
    """Risk management configuration."""
    type: RiskModelType = RiskModelType.VAR
    confidence_level: float = Field(0.95, ge=0, le=1)
    time_horizon: int = Field(1, ge=1)
    monte_carlo_simulations: int = Field(10000, ge=1000)
    max_portfolio_var: float = Field(0.05, ge=0, le=1)
    enable_stress_testing: bool = True
    enable_scenario_analysis: bool = True

class APIConfig(BaseModel):
    """API configuration."""
    binance_base_url: str = "https://api.binance.com"
    binance_testnet_url: str = "https://testnet.binance.vision"
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    rate_limit: int = Field(1200, ge=1)
    timeout: int = Field(30, ge=1)
    retry_attempts: int = Field(3, ge=1)

class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "postgresql"
    host: str = "localhost"
    port: int = Field(5432, ge=1, le=65535)
    name: str = "trading_bot"
    pool_size: int = Field(10, ge=1)
    max_overflow: int = Field(20, ge=0)
    echo: bool = False

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/trading_bot.log"
    max_size: str = "10MB"
    backup_count: int = Field(5, ge=1)
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_console: bool = True
    enable_file: bool = True

class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enable_metrics: bool = True
    metrics_port: int = Field(8000, ge=1, le=65535)
    health_check_interval: int = Field(30, ge=1)
    enable_dashboard: bool = True
    dashboard_port: int = Field(8080, ge=1, le=65535)
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = {
        "max_drawdown": 0.1,
        "min_sharpe_ratio": 1.0,
        "max_var": 0.05
    }

class SecurityConfig(BaseModel):
    """Security configuration."""
    encrypt_api_keys: bool = True
    session_timeout: int = Field(3600, ge=60)
    max_login_attempts: int = Field(3, ge=1)
    enable_2fa: bool = False
    jwt_secret_key: Optional[str] = None
    encryption_key: Optional[str] = None

class PerformanceConfig(BaseModel):
    """Performance configuration."""
    enable_gpu: bool = True
    num_workers: int = Field(4, ge=1)
    batch_size: int = Field(32, ge=1)
    cache_ttl: int = Field(300, ge=1)
    enable_parallel_processing: bool = True
    max_memory_usage: float = Field(0.8, ge=0, le=1)

class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling features."""
    enable_sentiment_analysis: bool = True
    enable_news_analysis: bool = True
    enable_social_media_analysis: bool = True
    enable_options_flow: bool = True
    enable_dark_pool_analysis: bool = True
    enable_macro_indicators: bool = True
    enable_technical_indicators: bool = True
    enable_volume_profile: bool = True

class Config(BaseModel):
    """Advanced configuration management for the AI Trading Bot."""
    
    trading: TradingConfig = TradingConfig()
    data_sources: DataSourcesConfig = DataSourcesConfig()
    models: ModelConfig = ModelConfig()
    risk_model: RiskModelConfig = RiskModelConfig()
    api: APIConfig = APIConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    feature_flags: FeatureFlags = FeatureFlags()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class ConfigManager:
    """Advanced configuration manager with environment variable support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        # Load environment variables
        load_dotenv()
        
        # Set default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Config:
        """Load configuration from YAML file and environment variables."""
        try:
            # Load YAML config if exists
            yaml_config = {}
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    yaml_config = yaml.safe_load(file) or {}
            
            # Override with environment variables
            env_config = self._load_env_config()
            
            # Merge configurations
            merged_config = self._merge_configs(yaml_config, env_config)
            
            return Config(**merged_config)
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return Config()  # Return default config
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Trading configuration
        if os.getenv('TRADING_MODE'):
            env_config.setdefault('trading', {})['mode'] = os.getenv('TRADING_MODE')
        if os.getenv('INITIAL_BALANCE'):
            env_config.setdefault('trading', {})['initial_balance'] = float(os.getenv('INITIAL_BALANCE'))
        if os.getenv('MAX_POSITION_SIZE'):
            env_config.setdefault('trading', {})['max_position_size'] = float(os.getenv('MAX_POSITION_SIZE'))
        
        # Model configuration
        if os.getenv('MODEL_TYPE'):
            env_config.setdefault('models', {})['type'] = os.getenv('MODEL_TYPE')
        if os.getenv('LOOKBACK_WINDOW'):
            env_config.setdefault('models', {})['lookback_window'] = int(os.getenv('LOOKBACK_WINDOW'))
        if os.getenv('PREDICTION_HORIZON'):
            env_config.setdefault('models', {})['prediction_horizon'] = int(os.getenv('PREDICTION_HORIZON'))
        
        # Risk management
        if os.getenv('VAR_CONFIDENCE_LEVEL'):
            env_config.setdefault('risk_model', {})['confidence_level'] = float(os.getenv('VAR_CONFIDENCE_LEVEL'))
        if os.getenv('MONTE_CARLO_SIMULATIONS'):
            env_config.setdefault('risk_model', {})['monte_carlo_simulations'] = int(os.getenv('MONTE_CARLO_SIMULATIONS'))
        
        # Database
        if os.getenv('DATABASE_URL'):
            env_config.setdefault('database', {})['url'] = os.getenv('DATABASE_URL')
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            env_config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            env_config.setdefault('logging', {})['file'] = os.getenv('LOG_FILE')
        
        return env_config
    
    def _merge_configs(self, yaml_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge YAML and environment configurations."""
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(yaml_config, env_config)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self._config.logging
        
        # Create logs directory if it doesn't exist
        log_file_path = Path(log_config.file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.level.upper()),
            format=log_config.format,
            handlers=[
                logging.StreamHandler() if log_config.enable_console else logging.NullHandler(),
                logging.FileHandler(log_file_path) if log_config.enable_file else logging.NullHandler()
            ]
        )
    
    @property
    def config(self) -> Config:
        """Get the current configuration."""
        return self._config
    
    def get_api_key(self, exchange: str) -> Optional[str]:
        """Get API key for exchange (encrypted if enabled)."""
        key_name = f"{exchange.upper()}_API_KEY"
        return os.getenv(key_name)
    
    def get_secret_key(self, exchange: str) -> Optional[str]:
        """Get secret key for exchange (encrypted if enabled)."""
        key_name = f"{exchange.upper()}_SECRET_KEY"
        return os.getenv(key_name)
    
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self._config.trading.mode == TradingMode.PAPER
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return os.getenv('DATABASE_URL', f"sqlite:///{self._config.database.name}.db")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    def get_influxdb_config(self) -> Dict[str, str]:
        """Get InfluxDB configuration."""
        return {
            'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
            'token': os.getenv('INFLUXDB_TOKEN', ''),
            'org': os.getenv('INFLUXDB_ORG', ''),
            'bucket': os.getenv('INFLUXDB_BUCKET', 'trading_data')
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            'sentry_dsn': os.getenv('SENTRY_DSN'),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'slack_webhook_url': os.getenv('SLACK_WEBHOOK_URL')
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required API keys for live trading
        if not self.is_paper_trading():
            if not self.get_api_key('binance'):
                issues.append("Binance API key required for live trading")
            if not self.get_secret_key('binance'):
                issues.append("Binance secret key required for live trading")
        
        # Check database connection
        if not self.get_database_url():
            issues.append("Database URL not configured")
        
        # Check risk parameters
        if self._config.trading.take_profit <= self._config.trading.stop_loss:
            issues.append("Take profit must be greater than stop loss")
        
        if self._config.trading.max_position_size > 1.0:
            issues.append("Max position size cannot exceed 100%")
        
        return issues
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to YAML file."""
        if config_path is None:
            config_path = self.config_path
        
        config_dict = self._config.dict()
        
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
    
    def reload_config(self):
        """Reload configuration from file and environment."""
        self._config = self._load_config()
        self._setup_logging()

# Global configuration instance
config_manager = ConfigManager()
config = config_manager.config