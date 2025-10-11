"""Trading execution engine."""
from typing import List
from app.signals.features import compute_features
from app.signals.rules import rules_signal, check_exit_signal
from app.models import LGBMTradingModel
from app.trading.broker_alpaca import Broker
from app.trading.risk import size_order, within_risk_limits, calculate_stop_loss, calculate_take_profit
from app.data.repos import get_recent_candles, insert_signal, insert_equity_point, get_open_positions, update_position_tracking
from app.core.logger import logger
from app.core.utils import is_market_hours


def on_new_bar(symbols: List[str], cfg: dict):
    """Execute trading logic on new bar.
    
    Args:
        symbols: List of symbols to process
        cfg: Configuration dictionary
    """
    # Check market hours
    if not is_market_hours():
        logger.debug("Outside market hours, skipping execution")
        return
    
    broker = Broker()

    # Optional ML model (load once)
    ml_model = None
    try:
        from pathlib import Path
        model_path = Path('trained_models/lgbm.pkl')
        if model_path.exists():
            ml_model = LGBMTradingModel()
            ml_model.load(str(model_path))
            logger.debug("Loaded LightGBM model for live trading")
    except Exception:
        ml_model = None
    
    # Process each symbol
    for symbol in symbols:
        try:
            process_symbol(symbol, broker, cfg, ml_model)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Update equity curve
    try:
        equity = broker.get_equity()
        insert_equity_point(equity)
        logger.debug(f"Equity updated: ${equity:,.2f}")
    except Exception as e:
        logger.error(f"Error updating equity: {e}")


def process_symbol(symbol: str, broker: Broker, cfg: dict, ml_model: LGBMTradingModel | None = None):
    """Process trading logic for a single symbol.
    
    Args:
        symbol: Symbol to process
        broker: Broker instance
        cfg: Configuration
    """
    # Get recent candles
    df = get_recent_candles(symbol, limit=300)
    
    if df is None or len(df) < 50:
        logger.debug(f"Insufficient data for {symbol}")
        return
    
    # Compute features
    try:
        feats = compute_features(df, cfg)
        if feats.empty:
            logger.warning(f"No valid features for {symbol}")
            return
        
        latest = feats.iloc[-1]
    except Exception as e:
        logger.error(f"Feature computation failed for {symbol}: {e}")
        return
    
    # Get signal (rules + optional ML)
    side, conf, reason = rules_signal(latest.to_dict(), cfg, ml_model)
    
    # Check confidence threshold
    if conf < cfg['strategy'].get('confidence_min', 0.5):
        side = "FLAT"
        reason = f"Low confidence ({conf:.2f})"
    
    # Log signal
    insert_signal(symbol, side, conf, reason)
    
    # Check for exit conditions on existing positions
    check_position_exits(symbol, latest, broker, cfg)
    
    # Execute new entries
    if side == "FLAT":
        return
    
    # Check risk limits
    if not within_risk_limits(symbol, cfg):
        return
    
    # Calculate position size
    price = float(latest['close'])
    qty = size_order(symbol, cfg, price)
    
    if qty <= 0:
        logger.warning(f"Invalid quantity for {symbol}")
        return
    
    # Place order
    try:
        broker.place_with_risk(symbol, side, qty, price, cfg)
        logger.info(f"✓ Signal executed: {side} {qty} {symbol} @ ${price:.2f} - {reason}")
    except Exception as e:
        logger.error(f"Order execution failed for {symbol}: {e}")


def check_position_exits(symbol: str, latest, broker: Broker, cfg: dict):
    """Check and execute position exits based on signals or risk levels.
    
    Args:
        symbol: Symbol to check
        latest: Latest candle with features
        broker: Broker instance
        cfg: Configuration
    """
    try:
        # Get open positions
        positions_df = get_open_positions()
        
        if symbol not in positions_df['symbol'].values:
            return
        
        position = positions_df[positions_df['symbol'] == symbol].iloc[0]
        position_side = "LONG" if position['qty'] > 0 else "SHORT"
        current_qty = abs(float(position['qty']))
        entry_qty = float(position.get('entry_qty', current_qty))
        first_tp_hit = bool(position.get('first_tp_hit', 0))
        current_stop_loss_pct = position.get('stop_loss_pct')
        
        # Use stored stop loss if available, otherwise use config default
        if current_stop_loss_pct is None:
            current_stop_loss_pct = cfg['risk']['stop_loss_pct']
        
        # Check exit signals
        should_exit, exit_reason = check_exit_signal(position_side, latest, cfg)
        
        if should_exit:
            broker.close_position(symbol, exit_reason)
            logger.info(f"Position exited: {symbol} - {exit_reason}")
            return
        
        # Get price levels
        current_price = float(latest['close'])
        entry_price = float(position['avg_price'])
        
        # Get take profit levels
        take_profit_pct = cfg['risk']['take_profit_pct']
        take_profit_2_pct = cfg['risk'].get('take_profit_2_pct', take_profit_pct * 2)
        partial_profit_pct = cfg['risk'].get('partial_profit_pct', 0.5)
        move_sl_to_breakeven = cfg['risk'].get('move_sl_to_breakeven', True)
        
        # Calculate price levels
        if position_side == "LONG":
            stop_price = entry_price * (1 - current_stop_loss_pct)
            tp1_price = entry_price * (1 + take_profit_pct)
            tp2_price = entry_price * (1 + take_profit_2_pct)
            
            # Check stop loss
            if current_price <= stop_price:
                exit_reason = "Stop loss" if not first_tp_hit else "Breakeven stop"
                broker.close_position(symbol, f"{exit_reason} @ ${current_price:.2f}")
                logger.info(f"Position exited: {symbol} - {exit_reason}")
                return
            
            # Check second take profit (only if first TP already hit)
            if first_tp_hit and current_price >= tp2_price:
                broker.close_position(symbol, f"Take profit 2 @ ${current_price:.2f}")
                logger.info(f"Position exited: {symbol} - Take profit 2")
                return
            
            # Check first take profit (partial exit)
            if not first_tp_hit and current_price >= tp1_price:
                # Calculate partial close quantity
                qty_to_close = int(entry_qty * partial_profit_pct)
                
                if qty_to_close > 0 and qty_to_close < current_qty:
                    # Close partial position
                    broker.close_partial_position(symbol, qty_to_close, f"Take profit 1 (partial) @ ${current_price:.2f}")
                    logger.info(f"Partial exit: {symbol} closed {qty_to_close}/{entry_qty} shares @ ${current_price:.2f}")
                    
                    # Update tracking: mark first TP as hit and move stop to breakeven
                    if move_sl_to_breakeven:
                        update_position_tracking(symbol, first_tp_hit=True, stop_loss_pct=0.0)
                        logger.info(f"Moved {symbol} stop loss to breakeven")
                    else:
                        update_position_tracking(symbol, first_tp_hit=True)
                    
                    return
        
        else:  # SHORT
            stop_price = entry_price * (1 + current_stop_loss_pct)
            tp1_price = entry_price * (1 - take_profit_pct)
            tp2_price = entry_price * (1 - take_profit_2_pct)
            
            # Check stop loss
            if current_price >= stop_price:
                exit_reason = "Stop loss" if not first_tp_hit else "Breakeven stop"
                broker.close_position(symbol, f"{exit_reason} @ ${current_price:.2f}")
                logger.info(f"Position exited: {symbol} - {exit_reason}")
                return
            
            # Check second take profit (only if first TP already hit)
            if first_tp_hit and current_price <= tp2_price:
                broker.close_position(symbol, f"Take profit 2 @ ${current_price:.2f}")
                logger.info(f"Position exited: {symbol} - Take profit 2")
                return
            
            # Check first take profit (partial exit)
            if not first_tp_hit and current_price <= tp1_price:
                # Calculate partial close quantity
                qty_to_close = int(entry_qty * partial_profit_pct)
                
                if qty_to_close > 0 and qty_to_close < current_qty:
                    # Close partial position
                    broker.close_partial_position(symbol, qty_to_close, f"Take profit 1 (partial) @ ${current_price:.2f}")
                    logger.info(f"Partial exit: {symbol} closed {qty_to_close}/{entry_qty} shares @ ${current_price:.2f}")
                    
                    # Update tracking: mark first TP as hit and move stop to breakeven
                    if move_sl_to_breakeven:
                        update_position_tracking(symbol, first_tp_hit=True, stop_loss_pct=0.0)
                        logger.info(f"Moved {symbol} stop loss to breakeven")
                    else:
                        update_position_tracking(symbol, first_tp_hit=True)
                    
                    return
        
    except Exception as e:
        logger.error(f"Error checking exits for {symbol}: {e}")

