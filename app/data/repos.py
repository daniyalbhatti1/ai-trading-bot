"""Data repository functions."""
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
from app.data.db import get_connection
from app.core.logger import logger


def get_recent_candles(symbol: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch recent candles for a symbol."""
    try:
        with get_connection() as conn:
            query = """
                SELECT ts, open, high, low, close, volume
                FROM candles
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, limit))
            
            if df.empty:
                return None
            
            # Reverse to chronological order
            df = df.iloc[::-1].reset_index(drop=True)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.set_index('ts')
            
            return df
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
        return None


def insert_signal(symbol: str, side: str, confidence: float, reason: str):
    """Insert a trading signal."""
    ts = datetime.utcnow().isoformat()
    
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO signals (ts, symbol, side, confidence, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, symbol, side, confidence, reason))
    
    logger.info(f"Signal logged: {symbol} {side} ({confidence:.2f}) - {reason}")


def log_order(order_data: Dict):
    """Log an order to the database."""
    ts = datetime.utcnow().isoformat()
    
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO orders (ts, symbol, side, qty, type, limit_price, stop_price, 
                              client_order_id, status, fill_price, fill_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts,
            order_data.get('symbol'),
            order_data.get('side'),
            order_data.get('qty'),
            order_data.get('type'),
            order_data.get('limit_price'),
            order_data.get('stop_price'),
            order_data.get('client_order_id'),
            order_data.get('status'),
            order_data.get('filled_avg_price'),
            order_data.get('filled_at')
        ))


def refresh_positions(client):
    """Refresh positions from broker."""
    try:
        positions = client.get_all_positions()
        ts = datetime.utcnow().isoformat()
        
        with get_connection() as conn:
            # Get existing position tracking data before clearing
            existing_data = {}
            cursor = conn.execute("SELECT symbol, entry_qty, first_tp_hit, stop_loss_pct FROM positions")
            for row in cursor.fetchall():
                existing_data[row[0]] = {
                    'entry_qty': row[1],
                    'first_tp_hit': row[2],
                    'stop_loss_pct': row[3]
                }
            
            # Clear existing positions
            conn.execute("DELETE FROM positions")
            
            # Insert current positions, preserving tracking data if it exists
            for pos in positions:
                symbol = pos.symbol
                existing = existing_data.get(symbol, {})
                
                # If no existing data, this is a new position
                entry_qty = existing.get('entry_qty', float(pos.qty))
                first_tp_hit = existing.get('first_tp_hit', 0)
                stop_loss_pct = existing.get('stop_loss_pct')
                
                conn.execute("""
                    INSERT INTO positions (symbol, avg_price, qty, unrealized_pl, last_update, 
                                          entry_qty, first_tp_hit, stop_loss_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    float(pos.avg_entry_price),
                    float(pos.qty),
                    float(pos.unrealized_pl),
                    ts,
                    entry_qty,
                    first_tp_hit,
                    stop_loss_pct
                ))
        
        logger.debug(f"Refreshed {len(positions)} positions")
    except Exception as e:
        logger.error(f"Error refreshing positions: {e}")


def insert_equity_point(equity: float):
    """Insert equity curve point."""
    ts = datetime.utcnow().isoformat()
    
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO equity_curve (ts, equity)
            VALUES (?, ?)
        """, (ts, equity))


def get_equity_curve(days: int = 30) -> pd.DataFrame:
    """Get equity curve for the last N days."""
    with get_connection() as conn:
        query = """
            SELECT ts, equity
            FROM equity_curve
            WHERE ts >= datetime('now', ?)
            ORDER BY ts
        """
        df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
        df['ts'] = pd.to_datetime(df['ts'])
        return df


def get_recent_orders(limit: int = 100) -> pd.DataFrame:
    """Get recent orders."""
    with get_connection() as conn:
        query = """
            SELECT * FROM orders
            ORDER BY ts DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df


def get_recent_signals(limit: int = 50) -> pd.DataFrame:
    """Get recent signals."""
    with get_connection() as conn:
        query = """
            SELECT * FROM signals
            ORDER BY ts DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df


def get_open_positions() -> pd.DataFrame:
    """Get current open positions."""
    with get_connection() as conn:
        query = "SELECT * FROM positions WHERE qty != 0"
        df = pd.read_sql_query(query, conn)
        return df


def update_position_tracking(symbol: str, first_tp_hit: bool = None, stop_loss_pct: float = None):
    """Update position tracking fields for partial profit management.
    
    Args:
        symbol: Symbol to update
        first_tp_hit: Whether first take profit was hit
        stop_loss_pct: New stop loss percentage
    """
    with get_connection() as conn:
        updates = []
        params = []
        
        if first_tp_hit is not None:
            updates.append("first_tp_hit = ?")
            params.append(1 if first_tp_hit else 0)
        
        if stop_loss_pct is not None:
            updates.append("stop_loss_pct = ?")
            params.append(stop_loss_pct)
        
        if updates:
            params.append(symbol)
            query = f"UPDATE positions SET {', '.join(updates)} WHERE symbol = ?"
            conn.execute(query, params)
            logger.debug(f"Updated tracking for {symbol}: first_tp_hit={first_tp_hit}, stop_loss_pct={stop_loss_pct}")

