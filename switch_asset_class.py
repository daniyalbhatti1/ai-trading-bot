#!/usr/bin/env python3
"""
Asset Class Switcher for AI Trading Bot
Allows easy switching between crypto and stock trading
"""

import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any

class AssetClassSwitcher:
    """Helper class to switch between crypto and stock trading configurations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
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
    
    def switch_to_crypto(self) -> None:
        """Switch to crypto trading configuration."""
        print("🪙 Switching to CRYPTO trading...")
        
        # Update data source
        self.config['data_sources']['primary'] = 'binance'
        
        # Update trading base currency
        self.config['trading']['base_currency'] = 'USDT'
        
        # Update active asset class
        self.config['asset_classes']['active'] = 'crypto'
        
        # Update strategy symbols
        crypto_symbols = self.config['asset_classes']['crypto']['symbols']
        self.config['strategies'][0]['symbols'] = crypto_symbols[:2]  # Use first 2 for safety
        
        # Update timeframe
        self.config['strategies'][0]['timeframe'] = self.config['asset_classes']['crypto']['timeframe']
        
        self._save_config()
        print("✅ Successfully switched to CRYPTO trading!")
        print(f"📊 Trading symbols: {crypto_symbols[:2]}")
        print("🌐 Exchange: Binance")
        print("⏰ Trading hours: 24/7")
    
    def switch_to_stocks(self) -> None:
        """Switch to stock trading configuration."""
        print("📈 Switching to STOCK trading...")
        
        # Get current stock data source (don't change it)
        current_stock_api = self.config['data_sources']['primary']
        
        # Update trading base currency
        self.config['trading']['base_currency'] = 'USD'
        
        # Update active asset class
        self.config['asset_classes']['active'] = 'stocks'
        
        # Update strategy symbols
        stock_symbols = self.config['asset_classes']['stocks']['symbols']
        self.config['strategies'][0]['symbols'] = stock_symbols[:2]  # Use first 2 for safety
        
        # Update timeframe
        self.config['strategies'][0]['timeframe'] = self.config['asset_classes']['stocks']['timeframe']
        
        self._save_config()
        print("✅ Successfully switched to STOCK trading!")
        print(f"📊 Trading symbols: {stock_symbols[:2]}")
        # Map API names to display names
        api_display_names = {
            'alpha_vantage': 'Alpha Vantage',
            'yahoo': 'Yahoo Finance',
            'alpaca': 'Alpaca',
            'interactive_brokers': 'Interactive Brokers',
            'td_ameritrade': 'TD Ameritrade'
        }
        display_name = api_display_names.get(current_stock_api, current_stock_api.title())
        print(f"🌐 Exchange: {display_name}")
        print("⏰ Trading hours: 9:30-16:00 EST")
    
    def show_current_config(self) -> None:
        """Display current configuration."""
        active = self.config['asset_classes']['active']
        active_config = self.config['asset_classes'][active]
        
        print(f"\n🔍 Current Configuration:")
        print(f"📊 Asset Class: {active.upper()}")
        print(f"🌐 Exchange: {active_config['exchange']}")
        print(f"💰 Base Currency: {active_config['base_currency']}")
        print(f"📈 Trading Symbols: {self.config['strategies'][0]['symbols']}")
        print(f"⏰ Timeframe: {active_config['timeframe']}")
        print(f"🕐 Trading Hours: {active_config['trading_hours']}")
        
        # Show available symbols for current asset class
        all_symbols = active_config['symbols']
        print(f"🎯 Available symbols: {', '.join(all_symbols)}")
    
    def list_available_symbols(self, asset_class: str = None) -> None:
        """List available symbols for an asset class."""
        if asset_class is None:
            asset_class = self.config['asset_classes']['active']
        
        if asset_class not in self.config['asset_classes']:
            print(f"❌ Invalid asset class: {asset_class}")
            return
        
        symbols = self.config['asset_classes'][asset_class]['symbols']
        print(f"\n📋 Available {asset_class.upper()} symbols:")
        
        if asset_class == 'crypto':
            print("🪙 Cryptocurrencies:")
            for symbol in symbols:
                crypto_name = symbol.replace('USDT', '')
                print(f"  • {symbol} ({crypto_name})")
        else:
            print("📈 Stocks:")
            for symbol in symbols:
                print(f"  • {symbol}")
    
    def customize_symbols(self, asset_class: str, symbols: list) -> None:
        """Customize symbols for an asset class."""
        if asset_class not in self.config['asset_classes']:
            print(f"❌ Invalid asset class: {asset_class}")
            return
        
        print(f"🎯 Updating {asset_class} symbols...")
        self.config['asset_classes'][asset_class]['symbols'] = symbols
        
        # If this is the active asset class, update strategy too
        if self.config['asset_classes']['active'] == asset_class:
            self.config['strategies'][0]['symbols'] = symbols[:2]
        
        self._save_config()
        print(f"✅ Updated {asset_class} symbols: {symbols}")

def main():
    """Main function to handle command line arguments."""
    switcher = AssetClassSwitcher()
    
    if len(sys.argv) < 2:
        print("🤖 AI Trading Bot - Asset Class Switcher")
        print("\nUsage:")
        print("  python switch_asset_class.py crypto     # Switch to crypto trading")
        print("  python switch_asset_class.py stocks     # Switch to stock trading")
        print("  python switch_asset_class.py status     # Show current configuration")
        print("  python switch_asset_class.py list       # List available symbols")
        print("  python switch_asset_class.py list crypto  # List crypto symbols")
        print("  python switch_asset_class.py list stocks # List stock symbols")
        print("\nExamples:")
        print("  python switch_asset_class.py crypto")
        print("  python switch_asset_class.py stocks")
        print("  python switch_asset_class.py status")
        return
    
    command = sys.argv[1].lower()
    
    if command == "crypto":
        switcher.switch_to_crypto()
    elif command == "stocks":
        switcher.switch_to_stocks()
    elif command == "status":
        switcher.show_current_config()
    elif command == "list":
        asset_class = sys.argv[2] if len(sys.argv) > 2 else None
        switcher.list_available_symbols(asset_class)
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: crypto, stocks, status, or list")

if __name__ == "__main__":
    main()
