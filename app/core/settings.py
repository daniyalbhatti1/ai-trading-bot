"""Settings and configuration management."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    
    # Alpaca API
    ALPACA_API_KEY_ID: str = Field(default="")
    ALPACA_API_SECRET: str = Field(default="")
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets")
    
    # LLM Settings
    OPENAI_API_KEY: str = Field(default="")
    LLM_PROVIDER: str = Field(default="none")  # openai, ollama, or none
    
    # Database
    DB_PATH: str = Field(default="./data/trades.db")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def load_config():
    """Load YAML configuration."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Global instances
settings = Settings()
config = load_config()

