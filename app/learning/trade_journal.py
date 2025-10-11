"""Trade Journal: Capture and store all trade details for learning."""
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from app.data.db import get_connection
from app.core.logger import logger


def log_trade_to_journal(
    trade: Dict,
    entry_context: Dict,
    exit_context: Optional[Dict] = None,
    ml_info: Optional[Dict] = None
):
    """Log a completed trade to the journal with full context.
    
    Args:
        trade: Basic trade info (entry_time, exit_time, symbol, side, prices, pnl, etc.)
        entry_context: Market conditions and ICT features at entry
        exit_context: Market conditions at exit
        ml_info: ML model prediction and confidence
    """
    try:
        # Calculate duration
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
        
        # Convert timestamps to ISO strings for database
        entry_time_str = entry_time.isoformat()
        exit_time_str = exit_time.isoformat()
        
        # Calculate P&L percentage
        pnl_pct = (trade['pnl'] / (trade['entry_price'] * trade['qty'])) * 100
        
        # Calculate risk/reward ratio
        risk_reward_ratio = None
        if 'max_adverse_excursion' in trade and trade['max_adverse_excursion'] != 0:
            risk_reward_ratio = abs(trade.get('max_favorable_excursion', 0) / trade['max_adverse_excursion'])
        
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO trade_journal (
                    entry_time, exit_time, symbol, side,
                    entry_price, exit_price, qty, pnl, pnl_pct,
                    exit_reason, duration_minutes,
                    
                    -- Entry context
                    entry_rsi, entry_macd, entry_ema_fast, entry_ema_slow,
                    entry_atr, entry_volume,
                    
                    -- ICT context
                    entry_liq_sweep_bull, entry_liq_sweep_bear,
                    entry_bos_bull, entry_bos_bear,
                    entry_fvg_bull, entry_fvg_bear,
                    entry_retrace_to_fvg_bull, entry_retrace_to_fvg_bear,
                    entry_engulfing_bull, entry_engulfing_bear,
                    entry_ict_setup_bull, entry_ict_setup_bear,
                    
                    -- Key levels
                    entry_pdh, entry_pdl,
                    entry_4h_high, entry_4h_low,
                    entry_1h_high, entry_1h_low,
                    
                    -- Exit context
                    exit_rsi, exit_macd, exit_ema_fast, exit_ema_slow, exit_atr,
                    
                    -- Performance metrics
                    max_favorable_excursion, max_adverse_excursion, risk_reward_ratio,
                    
                    -- ML info
                    ml_confidence, ml_prediction,
                    
                    -- Timestamps
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?)
            """, (
                entry_time_str, exit_time_str, trade['symbol'], trade['side'],
                trade['entry_price'], trade['exit_price'], trade['qty'], trade['pnl'], pnl_pct,
                trade.get('reason', 'Unknown'), duration_minutes,
                
                # Entry context
                entry_context.get('rsi'), entry_context.get('macd'),
                entry_context.get('ema_fast'), entry_context.get('ema_slow'),
                entry_context.get('atr'), entry_context.get('volume'),
                
                # ICT context
                entry_context.get('liq_sweep_bull', 0), entry_context.get('liq_sweep_bear', 0),
                entry_context.get('bos_bull', 0), entry_context.get('bos_bear', 0),
                entry_context.get('fvg_bull', 0), entry_context.get('fvg_bear', 0),
                entry_context.get('retrace_to_fvg_bull', 0), entry_context.get('retrace_to_fvg_bear', 0),
                entry_context.get('engulfing_bull', 0), entry_context.get('engulfing_bear', 0),
                entry_context.get('ict_setup_bull', 0), entry_context.get('ict_setup_bear', 0),
                
                # Key levels
                entry_context.get('pdh'), entry_context.get('pdl'),
                entry_context.get('4h_high'), entry_context.get('4h_low'),
                entry_context.get('1h_high'), entry_context.get('1h_low'),
                
                # Exit context
                exit_context.get('rsi') if exit_context else None,
                exit_context.get('macd') if exit_context else None,
                exit_context.get('ema_fast') if exit_context else None,
                exit_context.get('ema_slow') if exit_context else None,
                exit_context.get('atr') if exit_context else None,
                
                # Performance metrics
                trade.get('max_favorable_excursion'),
                trade.get('max_adverse_excursion'),
                risk_reward_ratio,
                
                # ML info
                ml_info.get('confidence') if ml_info else None,
                ml_info.get('prediction') if ml_info else None,
                
                # Timestamp
                datetime.utcnow().isoformat()
            ))
        
        logger.info(f"Trade logged to journal: {trade['symbol']} {trade['side']} P&L: ${trade['pnl']:.2f}")
        
    except Exception as e:
        logger.error(f"Error logging trade to journal: {e}")


def get_trade_history(symbol: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
    """Get trade history from journal.
    
    Args:
        symbol: Filter by symbol (optional)
        limit: Maximum number of trades to return
    
    Returns:
        DataFrame of trades
    """
    with get_connection() as conn:
        if symbol:
            query = """
                SELECT * FROM trade_journal
                WHERE symbol = ?
                ORDER BY entry_time DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, limit))
        else:
            query = """
                SELECT * FROM trade_journal
                ORDER BY entry_time DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        return df


def get_pattern_performance() -> pd.DataFrame:
    """Get performance statistics by pattern type.
    
    Returns:
        DataFrame with pattern analysis
    """
    with get_connection() as conn:
        query = """
            SELECT 
                CASE 
                    WHEN entry_ict_setup_bull = 1 THEN 'ICT Bullish Setup'
                    WHEN entry_ict_setup_bear = 1 THEN 'ICT Bearish Setup'
                    WHEN entry_liq_sweep_bull = 1 AND entry_bos_bull = 1 THEN 'Liq Sweep + BOS Bull'
                    WHEN entry_liq_sweep_bear = 1 AND entry_bos_bear = 1 THEN 'Liq Sweep + BOS Bear'
                    WHEN entry_engulfing_bull = 1 AND entry_retrace_to_fvg_bull = 1 THEN 'Engulfing at FVG Bull'
                    WHEN entry_engulfing_bear = 1 AND entry_retrace_to_fvg_bear = 1 THEN 'Engulfing at FVG Bear'
                    ELSE 'Traditional Strategy'
                END as pattern,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                CAST(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 as win_rate,
                AVG(pnl) as avg_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                SUM(pnl) as total_pnl
            FROM trade_journal
            GROUP BY pattern
            ORDER BY win_rate DESC
        """
        df = pd.read_sql_query(query, conn)
        return df


def analyze_trade_quality(trade_id: int) -> Dict:
    """Analyze a specific trade and assign quality score.
    
    Args:
        trade_id: ID of trade to analyze
    
    Returns:
        Dictionary with analysis and quality score
    """
    with get_connection() as conn:
        trade = pd.read_sql_query(
            "SELECT * FROM trade_journal WHERE id = ?",
            conn, params=(trade_id,)
        ).iloc[0]
    
    quality_score = 50  # Start at neutral
    what_worked = []
    what_failed = []
    
    # Analyze setup quality
    if trade['entry_ict_setup_bull'] == 1 or trade['entry_ict_setup_bear'] == 1:
        quality_score += 20
        what_worked.append("Complete ICT setup present at entry")
    
    if trade['entry_liq_sweep_bull'] == 1 or trade['entry_liq_sweep_bear'] == 1:
        quality_score += 10
        what_worked.append("Liquidity sweep identified")
    
    if trade['entry_bos_bull'] == 1 or trade['entry_bos_bear'] == 1:
        quality_score += 10
        what_worked.append("Break of structure confirmed")
    
    # Analyze execution
    if trade['pnl'] > 0:
        quality_score += 20
        what_worked.append(f"Profitable trade: ${trade['pnl']:.2f}")
    else:
        quality_score -= 20
        what_failed.append(f"Losing trade: ${trade['pnl']:.2f}")
    
    # Analyze risk management
    if trade['risk_reward_ratio'] and trade['risk_reward_ratio'] > 2:
        quality_score += 15
        what_worked.append(f"Good risk/reward ratio: {trade['risk_reward_ratio']:.2f}")
    elif trade['risk_reward_ratio'] and trade['risk_reward_ratio'] < 1:
        quality_score -= 15
        what_failed.append(f"Poor risk/reward ratio: {trade['risk_reward_ratio']:.2f}")
    
    # Analyze ML confidence
    if trade['ml_confidence'] and trade['ml_confidence'] > 0.8:
        quality_score += 10
        what_worked.append(f"High ML confidence: {trade['ml_confidence']:.2f}")
    elif trade['ml_confidence'] and trade['ml_confidence'] < 0.6:
        quality_score -= 5
        what_failed.append(f"Low ML confidence: {trade['ml_confidence']:.2f}")
    
    # Analyze RSI conditions
    if trade['side'] == 'LONG':
        if trade['entry_rsi'] and 30 < trade['entry_rsi'] < 50:
            quality_score += 5
            what_worked.append(f"Good entry RSI for LONG: {trade['entry_rsi']:.1f}")
        elif trade['entry_rsi'] and trade['entry_rsi'] > 70:
            quality_score -= 10
            what_failed.append(f"Overbought entry RSI: {trade['entry_rsi']:.1f}")
    else:  # SHORT
        if trade['exit_rsi'] and 50 < trade['entry_rsi'] < 70:
            quality_score += 5
            what_worked.append(f"Good entry RSI for SHORT: {trade['entry_rsi']:.1f}")
        elif trade['entry_rsi'] and trade['entry_rsi'] < 30:
            quality_score -= 10
            what_failed.append(f"Oversold entry RSI: {trade['entry_rsi']:.1f}")
    
    # Cap quality score between 0-100
    quality_score = max(0, min(100, quality_score))
    
    # Update database with analysis
    with get_connection() as conn:
        conn.execute("""
            UPDATE trade_journal
            SET quality_score = ?,
                what_worked = ?,
                what_failed = ?,
                analyzed_at = ?
            WHERE id = ?
        """, (
            quality_score,
            "; ".join(what_worked) if what_worked else None,
            "; ".join(what_failed) if what_failed else None,
            datetime.utcnow().isoformat(),
            trade_id
        ))
    
    return {
        'quality_score': quality_score,
        'what_worked': what_worked,
        'what_failed': what_failed
    }


def analyze_all_unanalyzed_trades():
    """Analyze all trades that haven't been analyzed yet."""
    with get_connection() as conn:
        unanalyzed = pd.read_sql_query(
            "SELECT id FROM trade_journal WHERE analyzed_at IS NULL",
            conn
        )
    
    for trade_id in unanalyzed['id']:
        try:
            analyze_trade_quality(trade_id)
            logger.info(f"Analyzed trade {trade_id}")
        except Exception as e:
            logger.error(f"Error analyzing trade {trade_id}: {e}")

