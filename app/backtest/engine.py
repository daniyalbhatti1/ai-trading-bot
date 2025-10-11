"""Backtesting engine."""
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from app.signals.features import compute_features
from app.signals.rules import rules_signal
from app.models.lgbm_model import LGBMTradingModel
from app.core.logger import logger
from app.data.repos import get_recent_candles
from app.data.db import get_connection
from app.learning.trade_journal import log_trade_to_journal


def _log_trade_with_context(trade: Dict, pos: Dict, current_features: Dict):
    """Helper to log trade to journal with full context."""
    try:
        # Log to journal with entry and exit context
        log_trade_to_journal(
            trade=trade,
            entry_context=pos.get('entry_context', {}),
            exit_context=current_features,
            ml_info={
                'confidence': pos.get('ml_confidence'),
                'prediction': pos.get('ml_reason')
            }
        )
    except Exception as e:
        logger.warning(f"Could not log trade to journal: {e}")


def run_backtest(symbols: List[str], cfg: dict, start_date: str = None, end_date: str = None, log_to_journal: bool = True) -> Dict:
    """Run backtest on historical data.
    
    Args:
        symbols: List of symbols to backtest
        cfg: Configuration dictionary
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
    
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Starting backtest for {len(symbols)} symbols")
    
    results = {
        'trades': [],
        'equity_curve': [],
        'metrics': {}
    }
    
    initial_capital = 100000.0
    cash = initial_capital
    positions = {}
    equity_curve = []
    
    # Get all candles from database
    with get_connection() as conn:
        query = """
            SELECT ts, symbol, open, high, low, close, volume
            FROM candles
            WHERE symbol IN ({})
        """.format(','.join(['?' for _ in symbols]))
        
        if start_date:
            query += " AND ts >= ?"
        if end_date:
            query += " AND ts <= ?"
        
        query += " ORDER BY ts"
        
        params = list(symbols)
        if start_date:
            params.append(start_date)
        if end_date:
            params.append(end_date)
        
        df_all = pd.read_sql_query(query, conn, params=params)
    
    if df_all.empty:
        logger.warning("No data available for backtest")
        return results
    
    df_all['ts'] = pd.to_datetime(df_all['ts'], utc=True)
    
    # Get unique timestamps
    timestamps = sorted(df_all['ts'].unique())
    
    # Optional ML model
    ml_model = None
    try:
        from pathlib import Path
        model_path = Path('trained_models/lgbm.pkl')
        if model_path.exists():
            ml_model = LGBMTradingModel()
            ml_model.load(str(model_path))
            logger.info("Loaded LightGBM model for backtest")
    except Exception as e:
        logger.warning(f"Could not load ML model: {e}")
        ml_model = None

    # Simulate trading bar by bar
    for i, ts in enumerate(timestamps):
        if i < 50:  # Need enough data for indicators
            continue
        
        current_bars = df_all[df_all['ts'] == ts]
        
        for _, row in current_bars.iterrows():
            symbol = row['symbol']
            price = row['close']
            
            # Get historical data up to this point
            hist_df = df_all[(df_all['symbol'] == symbol) & (df_all['ts'] <= ts)].tail(300).copy()
            hist_df = hist_df.set_index('ts')[['open', 'high', 'low', 'close', 'volume']]
            
            if len(hist_df) < 50:
                continue
            
            # Compute features
            try:
                feats = compute_features(hist_df, cfg)
                if feats.empty:
                    continue
                latest = feats.iloc[-1]
            except Exception as e:
                logger.debug(f"Feature computation failed for {symbol} at {ts}: {e}")
                continue
            
            # Get signal (rules + optional ML)
            side, conf, reason = rules_signal(latest.to_dict(), cfg, ml_model)
            
            # Check confidence
            if conf < cfg['strategy'].get('confidence_min', 0.5):
                side = "FLAT"
            
            # Exit logic
            if symbol in positions:
                pos = positions[symbol]
                entry_price = pos['entry_price']
                qty = pos['qty']
                position_side = pos['side']
                
                # Get current stop loss (may have been moved to breakeven)
                current_stop_loss = pos.get('stop_loss_pct', cfg['risk']['stop_loss_pct'])
                
                # Check if first TP already hit
                first_tp_hit = pos.get('first_tp_hit', False)
                
                # Get take profit levels
                take_profit_pct = cfg['risk']['take_profit_pct']
                take_profit_2_pct = cfg['risk'].get('take_profit_2_pct', take_profit_pct * 2)
                partial_profit_pct = cfg['risk'].get('partial_profit_pct', 0.5)
                move_sl_to_breakeven = cfg['risk'].get('move_sl_to_breakeven', True)
                
                should_exit = False
                should_partial_exit = False
                exit_reason = ""
                
                if position_side == "LONG":
                    # Check stop loss
                    if price <= entry_price * (1 - current_stop_loss):
                        should_exit = True
                        exit_reason = "Stop loss" if not first_tp_hit else "Breakeven stop"
                    # Check second take profit (only if first TP already hit)
                    elif first_tp_hit and price >= entry_price * (1 + take_profit_2_pct):
                        should_exit = True
                        exit_reason = "Take profit 2"
                    # Check first take profit
                    elif not first_tp_hit and price >= entry_price * (1 + take_profit_pct):
                        should_partial_exit = True
                        exit_reason = "Take profit 1 (partial)"
                else:  # SHORT
                    # Check stop loss
                    if price >= entry_price * (1 + current_stop_loss):
                        should_exit = True
                        exit_reason = "Stop loss" if not first_tp_hit else "Breakeven stop"
                    # Check second take profit (only if first TP already hit)
                    elif first_tp_hit and price <= entry_price * (1 - take_profit_2_pct):
                        should_exit = True
                        exit_reason = "Take profit 2"
                    # Check first take profit
                    elif not first_tp_hit and price <= entry_price * (1 - take_profit_pct):
                        should_partial_exit = True
                        exit_reason = "Take profit 1 (partial)"
                
                # Handle partial exit (first take profit)
                if should_partial_exit:
                    # Calculate quantity to close (e.g., 50%)
                    qty_to_close = int(qty * partial_profit_pct)
                    qty_remaining = qty - qty_to_close
                    
                    if qty_to_close > 0:
                        # Calculate P&L for partial close
                        if position_side == "LONG":
                            pnl = (price - entry_price) * qty_to_close
                        else:  # SHORT
                            pnl = (entry_price - price) * qty_to_close
                        
                        cash += pnl
                        
                        # Record the partial exit as a trade
                        trade_record = {
                            'entry_time': pos['entry_time'],
                            'exit_time': ts,
                            'symbol': symbol,
                            'side': position_side,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'qty': qty_to_close,
                            'pnl': pnl,
                            'reason': exit_reason,
                            'max_favorable_excursion': pos.get('max_favorable_excursion', 0),
                            'max_adverse_excursion': pos.get('max_adverse_excursion', 0)
                        }
                        results['trades'].append(trade_record)
                        
                        # Log to journal
                        if log_to_journal:
                            _log_trade_with_context(trade_record, pos, latest.to_dict())
                        
                        # Update position: reduce quantity and move stop loss to breakeven
                        pos['qty'] = qty_remaining
                        pos['first_tp_hit'] = True
                        
                        if move_sl_to_breakeven:
                            # Move stop loss to breakeven (0% loss)
                            pos['stop_loss_pct'] = 0.0
                            logger.info(f"Moved {symbol} stop loss to breakeven after taking partial profit")
                        
                        logger.info(f"Partial exit {symbol}: closed {qty_to_close}/{qty} shares at ${price:.2f}, P&L: ${pnl:.2f}")
                        continue
                
                # Handle full exit
                if should_exit:
                    # Close remaining position
                    if position_side == "LONG":
                        pnl = (price - entry_price) * qty
                    else:  # SHORT
                        pnl = (entry_price - price) * qty
                    
                    cash += pnl
                    
                    trade_record = {
                        'entry_time': pos['entry_time'],
                        'exit_time': ts,
                        'symbol': symbol,
                        'side': position_side,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'qty': qty,
                        'pnl': pnl,
                        'reason': exit_reason,
                        'max_favorable_excursion': pos.get('max_favorable_excursion', 0),
                        'max_adverse_excursion': pos.get('max_adverse_excursion', 0)
                    }
                    results['trades'].append(trade_record)
                    
                    # Log to journal
                    if log_to_journal:
                        _log_trade_with_context(trade_record, pos, latest.to_dict())
                    
                    del positions[symbol]
                    continue
            
            # Entry logic
            if side != "FLAT" and symbol not in positions:
                # Check position limits
                if len(positions) >= cfg['risk']['max_positions']:
                    continue
                
                # Size position
                qty = cfg['sizing']['fixed_qty']
                cost = price * qty
                
                if cost > cash:
                    continue
                
                # Open position (don't reduce cash - just track the position)
                # Cash will be adjusted when the position is closed based on actual P&L
                
                # Store entry context for learning
                entry_context = latest.to_dict()
                
                positions[symbol] = {
                    'entry_time': ts,
                    'entry_price': price,
                    'qty': qty,
                    'side': side,
                    'entry_context': entry_context,  # Store full entry context
                    'ml_confidence': conf,
                    'ml_reason': reason,
                    'max_favorable_excursion': 0,
                    'max_adverse_excursion': 0
                }
        
        # Calculate equity and track max excursions
        position_value = 0
        for symbol, pos in positions.items():
            current_price = df_all[(df_all['symbol'] == symbol) & (df_all['ts'] == ts)]['close'].values
            if len(current_price) > 0:
                current_price = current_price[0]
                
                # Calculate unrealized P&L
                if pos['side'] == "LONG":
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['qty']
                    position_value += current_price * pos['qty']
                else:  # SHORT
                    unrealized_pnl = (pos['entry_price'] - current_price) * pos['qty']
                    position_value += (2 * pos['entry_price'] - current_price) * pos['qty']
                
                # Track max favorable and adverse excursions
                if unrealized_pnl > pos['max_favorable_excursion']:
                    pos['max_favorable_excursion'] = unrealized_pnl
                if unrealized_pnl < pos['max_adverse_excursion']:
                    pos['max_adverse_excursion'] = unrealized_pnl
        
        equity = cash + position_value
        equity_curve.append({'ts': ts, 'equity': equity})
    
    # Close any remaining open positions at the end of the backtest
    if positions:
        logger.warning(f"Closing {len(positions)} open positions at end of backtest")
        last_ts = timestamps[-1]
        for symbol, pos in list(positions.items()):
            # Get final price
            final_price = df_all[(df_all['symbol'] == symbol) & (df_all['ts'] == last_ts)]['close'].values
            if len(final_price) > 0:
                final_price = final_price[0]
            else:
                # If no price at last timestamp, use the last available price
                final_price = df_all[df_all['symbol'] == symbol]['close'].iloc[-1]
            
            # Calculate P&L
            if pos['side'] == "LONG":
                pnl = (final_price - pos['entry_price']) * pos['qty']
            else:  # SHORT
                pnl = (pos['entry_price'] - final_price) * pos['qty']
            
            cash += pnl
            
            trade_record = {
                'entry_time': pos['entry_time'],
                'exit_time': last_ts,
                'symbol': symbol,
                'side': pos['side'],
                'entry_price': pos['entry_price'],
                'exit_price': final_price,
                'qty': pos['qty'],
                'pnl': pnl,
                'reason': 'End of backtest',
                'max_favorable_excursion': pos.get('max_favorable_excursion', 0),
                'max_adverse_excursion': pos.get('max_adverse_excursion', 0)
            }
            results['trades'].append(trade_record)
            
            # Log to journal (get final features for exit context)
            if log_to_journal:
                try:
                    final_df = df_all[(df_all['symbol'] == symbol) & (df_all['ts'] <= last_ts)].tail(300).copy()
                    final_df = final_df.set_index('ts')[['open', 'high', 'low', 'close', 'volume']]
                    final_feats = compute_features(final_df, cfg)
                    if not final_feats.empty:
                        _log_trade_with_context(trade_record, pos, final_feats.iloc[-1].to_dict())
                except:
                    pass
            
            logger.info(f"Closed {symbol} {pos['side']} position at end: P&L ${pnl:.2f}")
        
        positions.clear()
        
        # Update final equity point with closed positions
        if equity_curve:
            equity_curve[-1]['equity'] = cash
    
    results['equity_curve'] = equity_curve
    
    # Calculate metrics
    if equity_curve:
        results['metrics'] = calculate_metrics(results, initial_capital)
    
    logger.info(f"Backtest complete: {len(results['trades'])} trades")
    
    return results


def calculate_metrics(results: Dict, initial_capital: float) -> Dict:
    """Calculate backtest metrics."""
    from app.backtest.metrics import calculate_backtest_metrics
    return calculate_backtest_metrics(results, initial_capital)

