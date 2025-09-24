#!/usr/bin/env python3
"""
Stock API Switcher for AI Trading Bot
Allows easy switching between different stock trading APIs
"""

import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any

class StockAPISwitcher:
    """Helper class to switch between different stock trading APIs."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Available stock APIs with their configurations
        self.stock_apis = {
            "alpha_vantage": {
                "name": "Alpha Vantage",
                "description": "Free tier: 5 requests/min, 500/day",
                "features": ["Real-time data", "Historical data", "Free tier"],
                "setup_required": "API key (free)",
                "rate_limits": "5 requests/minute, 500/day",
                "cost": "Free tier available"
            },
            "yahoo": {
                "name": "Yahoo Finance",
                "description": "Free, unlimited requests",
                "features": ["Real-time data", "No API key", "Unlimited requests"],
                "setup_required": "None",
                "rate_limits": "None (unofficial)",
                "cost": "Completely free"
            },
            "alpaca": {
                "name": "Alpaca",
                "description": "Commission-free trading",
                "features": ["Paper trading", "Live trading", "Good API"],
                "setup_required": "Account registration",
                "rate_limits": "200 requests/minute",
                "cost": "Free for paper trading"
            },
            "interactive_brokers": {
                "name": "Interactive Brokers",
                "description": "Professional trading platform",
                "features": ["Global markets", "Advanced features", "Professional grade"],
                "setup_required": "Account + approval",
                "rate_limits": "Varies by account",
                "cost": "Commission-based"
            },
            "td_ameritrade": {
                "name": "TD Ameritrade",
                "description": "Professional platform with Thinkorswim",
                "features": ["Advanced tools", "Good API", "Paper trading"],
                "setup_required": "Account registration",
                "rate_limits": "120 requests/minute",
                "cost": "Free paper trading"
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the current configuration."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            sys.exit(1)
    
    def _save_config(self) -> None:
        """Save the updated configuration."""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            sys.exit(1)
    
    def list_available_apis(self) -> None:
        """List all available stock APIs with details."""
        print("📈 Available Stock Trading APIs")
        print("=" * 60)
        
        for api_id, api_info in self.stock_apis.items():
            print(f"\n🔹 {api_info['name']} ({api_id})")
            print(f"   Description: {api_info['description']}")
            print(f"   Features: {', '.join(api_info['features'])}")
            print(f"   Setup Required: {api_info['setup_required']}")
            print(f"   Rate Limits: {api_info['rate_limits']}")
            print(f"   Cost: {api_info['cost']}")
    
    def switch_to_api(self, api_name: str) -> None:
        """Switch to a specific stock API."""
        if api_name not in self.stock_apis:
            print(f"❌ Unknown API: {api_name}")
            print("Available APIs:", list(self.stock_apis.keys()))
            return
        
        api_info = self.stock_apis[api_name]
        print(f"🔄 Switching to {api_info['name']}...")
        
        # Update data source
        self.config['data_sources']['primary'] = api_name
        
        # Update stock exchange in asset_classes
        self.config['asset_classes']['stocks']['exchange'] = api_name
        
        # Update trading configuration based on API
        if api_name == "alpha_vantage":
            self.config['trading']['base_currency'] = 'USD'
            self.config['data_sources']['update_interval'] = 300  # 5 minutes (rate limit friendly)
        elif api_name == "yahoo":
            self.config['trading']['base_currency'] = 'USD'
            self.config['data_sources']['update_interval'] = 60   # 1 minute (no rate limits)
        elif api_name == "alpaca":
            self.config['trading']['base_currency'] = 'USD'
            self.config['data_sources']['update_interval'] = 60   # 1 minute
        else:
            self.config['trading']['base_currency'] = 'USD'
            self.config['data_sources']['update_interval'] = 300  # 5 minutes
        
        self._save_config()
        print(f"✅ Successfully switched to {api_info['name']}!")
        print(f"📊 API: {api_info['name']}")
        print(f"💰 Base Currency: USD")
        print(f"⏰ Update Interval: {self.config['data_sources']['update_interval']} seconds")
        print(f"🔧 Setup Required: {api_info['setup_required']}")
    
    def show_current_api(self) -> None:
        """Display current stock API configuration."""
        current_api = self.config['data_sources']['primary']
        current_stock_exchange = self.config['asset_classes']['stocks']['exchange']
        
        print(f"\n🔍 Current Stock API Configuration:")
        print(f"📊 Primary Data Source: {current_api}")
        print(f"🏢 Stock Exchange: {current_stock_exchange}")
        print(f"💰 Base Currency: {self.config['trading']['base_currency']}")
        print(f"⏰ Update Interval: {self.config['data_sources']['update_interval']} seconds")
        
        if current_api in self.stock_apis:
            api_info = self.stock_apis[current_api]
            print(f"📋 API Details:")
            print(f"   Name: {api_info['name']}")
            print(f"   Description: {api_info['description']}")
            print(f"   Features: {', '.join(api_info['features'])}")
            print(f"   Rate Limits: {api_info['rate_limits']}")
            print(f"   Cost: {api_info['cost']}")
    
    def get_setup_instructions(self, api_name: str) -> None:
        """Get setup instructions for a specific API."""
        if api_name not in self.stock_apis:
            print(f"❌ Unknown API: {api_name}")
            return
        
        api_info = self.stock_apis[api_name]
        print(f"\n🔧 Setup Instructions for {api_info['name']}")
        print("=" * 50)
        
        if api_name == "alpha_vantage":
            print("1. Go to: https://www.alphavantage.co/support/#api-key")
            print("2. Enter your email address")
            print("3. Click 'GET FREE API KEY'")
            print("4. Copy your API key")
            print("5. Add to your .env file:")
            print("   ALPHA_VANTAGE_API_KEY=your_api_key_here")
            print("\n📝 Note: Free tier allows 5 requests/minute, 500/day")
            
        elif api_name == "yahoo":
            print("1. No setup required!")
            print("2. Yahoo Finance API is completely free")
            print("3. No API key needed")
            print("4. No rate limits")
            print("\n📝 Note: This is an unofficial API, may be less stable")
            
        elif api_name == "alpaca":
            print("1. Go to: https://alpaca.markets/")
            print("2. Sign up for a free account")
            print("3. Get your API keys from the dashboard")
            print("4. Add to your .env file:")
            print("   ALPACA_API_KEY=your_api_key_here")
            print("   ALPACA_SECRET_KEY=your_secret_key_here")
            print("\n📝 Note: Free paper trading, commission-free live trading")
            
        elif api_name == "interactive_brokers":
            print("1. Go to: https://www.interactivebrokers.com/")
            print("2. Open a trading account")
            print("3. Get API access approved")
            print("4. Configure TWS or IB Gateway")
            print("5. Add to your .env file:")
            print("   IB_HOST=127.0.0.1")
            print("   IB_PORT=7497")
            print("   IB_CLIENT_ID=1")
            print("\n📝 Note: Professional platform, requires account approval")
            
        elif api_name == "td_ameritrade":
            print("1. Go to: https://developer.tdameritrade.com/")
            print("2. Create a developer account")
            print("3. Create a new app")
            print("4. Get your consumer key")
            print("5. Add to your .env file:")
            print("   TD_CONSUMER_KEY=your_consumer_key_here")
            print("   TD_REDIRECT_URI=http://localhost:8080")
            print("\n📝 Note: Free paper trading, requires account registration")

def main():
    """Main function to handle command line arguments."""
    switcher = StockAPISwitcher()
    
    if len(sys.argv) < 2:
        print("🤖 AI Trading Bot - Stock API Switcher")
        print("\nUsage:")
        print("  python switch_stock_api.py list                    # List all available APIs")
        print("  python switch_stock_api.py alpha_vantage           # Switch to Alpha Vantage")
        print("  python switch_stock_api.py yahoo                   # Switch to Yahoo Finance")
        print("  python switch_stock_api.py alpaca                  # Switch to Alpaca")
        print("  python switch_stock_api.py status                  # Show current API")
        print("  python switch_stock_api.py setup <api_name>        # Get setup instructions")
        print("\nExamples:")
        print("  python switch_stock_api.py list")
        print("  python switch_stock_api.py alpha_vantage")
        print("  python switch_stock_api.py setup yahoo")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        switcher.list_available_apis()
    elif command == "status":
        switcher.show_current_api()
    elif command == "setup":
        if len(sys.argv) > 2:
            api_name = sys.argv[2].lower()
            switcher.get_setup_instructions(api_name)
        else:
            print("❌ Please specify an API name")
            print("Usage: python switch_stock_api.py setup <api_name>")
    elif command in switcher.stock_apis:
        switcher.switch_to_api(command)
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: list, status, setup, or an API name")

if __name__ == "__main__":
    main()
