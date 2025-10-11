"""Feature engineering including TA, ICT structures, and SMT features."""
import pandas as pd
import numpy as np
import ta
from app.core.logger import logger


def compute_features(df: pd.DataFrame, cfg: dict, dxy_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute TA + ICT + SMT features on OHLCV data.
    
    Args:
        df: DataFrame with OHLC columns
        cfg: Configuration dictionary with indicator parameters
        dxy_df: Optional DataFrame with DXY data for SMT features
    
    Returns:
        DataFrame with added indicator columns
    """
    try:
        df = df.copy()
        
        indicators = cfg['strategy']['indicators']
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=indicators['rsi_period']).rsi()
        
        # EMAs
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=indicators['ema_fast']).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=indicators['ema_slow']).ema_indicator()
        
        # MACD
        macd_indicator = ta.trend.MACD(
            df['close'],
            window_fast=indicators['macd_fast'],
            window_slow=indicators['macd_slow'],
            window_sign=indicators['macd_signal']
        )
        df['MACD_12_26_9'] = macd_indicator.macd()
        df['MACD_12_26_9_h'] = macd_indicator.macd_diff()
        df['MACD_12_26_9_s'] = macd_indicator.macd_signal()
        
        # Additional features (optional)
        # ATR for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # ICT primitives
        df = _add_ict_features(df)

        # SMT with DXY (if provided)
        if dxy_df is not None and not dxy_df.empty:
            # Extract DXY close prices and align with main dataframe
            dxy_close = dxy_df['close'] if 'close' in dxy_df.columns else dxy_df.iloc[:, 0]
            df = _add_smt_with_dxy(df, dxy_close)

        # Drop NaN rows
        df = df.dropna()
        
        logger.debug(f"Computed features, valid rows: {len(df)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error computing features: {e}")
        raise


def get_latest_features(df: pd.DataFrame, cfg: dict, dxy_df: pd.DataFrame = None) -> pd.Series:
    """Get features for the latest candle."""
    df_features = compute_features(df, cfg, dxy_df)
    if df_features.empty:
        raise ValueError("No valid features computed")
    
    return df_features.iloc[-1]


def _add_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sophisticated ICT structure features for the advanced strategy.
    
    Strategy sequence:
    1. Find liquidity sweep (PDL/PDH - 4h/1h highs/lows, previous session highs/lows)
    2. Wait for BOS that creates FVG
    3. Wait for another BOS that creates another FVG
    4. Retrace to 2nd FVG
    5. Enter on reaction with good push (engulfing preferred)
    """
    out = df.copy()
    
    # 1. LIQUIDITY SWEEPS (PDL/PDH)
    # Previous Day High/Low (PDH/PDL) - using 24 periods as proxy for daily
    out['pdh'] = out['high'].rolling(24, min_periods=1).max().shift(1)
    out['pdl'] = out['low'].rolling(24, min_periods=1).min().shift(1)
    
    # 4H High/Low (using 4 periods as proxy for 4H in 1H data)
    out['h4_high'] = out['high'].rolling(4, min_periods=1).max().shift(1)
    out['h4_low'] = out['low'].rolling(4, min_periods=1).min().shift(1)
    
    # 1H High/Low (using 1 period as proxy for 1H in 1H data)
    out['h1_high'] = out['high'].shift(1)
    out['h1_low'] = out['low'].shift(1)
    
    # Previous Session High/Low (using 8 periods as proxy for session)
    out['session_high'] = out['high'].rolling(8, min_periods=1).max().shift(1)
    out['session_low'] = out['low'].rolling(8, min_periods=1).min().shift(1)
    
    # Detect liquidity sweeps
    out['liq_sweep_pdh'] = (out['high'] > out['pdh']).astype(int)
    out['liq_sweep_pdl'] = (out['low'] < out['pdl']).astype(int)
    out['liq_sweep_h4_high'] = (out['high'] > out['h4_high']).astype(int)
    out['liq_sweep_h4_low'] = (out['low'] < out['h4_low']).astype(int)
    out['liq_sweep_session_high'] = (out['high'] > out['session_high']).astype(int)
    out['liq_sweep_session_low'] = (out['low'] < out['session_low']).astype(int)
    
    # Combined liquidity sweep signal
    out['liq_sweep_bull'] = (out['liq_sweep_pdh'] | out['liq_sweep_h4_high'] | out['liq_sweep_session_high']).astype(int)
    out['liq_sweep_bear'] = (out['liq_sweep_pdl'] | out['liq_sweep_h4_low'] | out['liq_sweep_session_low']).astype(int)
    
    # 2. BREAK OF STRUCTURE (BOS)
    # Swing highs and lows for BOS detection
    swing_high = out['high'].rolling(5, min_periods=1).max().shift(1)
    swing_low = out['low'].rolling(5, min_periods=1).min().shift(1)
    
    # BOS detection
    out['bos_bull'] = (out['close'] > swing_high).astype(int)
    out['bos_bear'] = (out['close'] < swing_low).astype(int)
    
    # 3. FAIR VALUE GAPS (FVG)
    # FVG: gap between candle i-1 high and candle i+1 low (bullish) or vice versa
    prev_high = out['high'].shift(1)
    next_low = out['low'].shift(-1)
    prev_low = out['low'].shift(1)
    next_high = out['high'].shift(-1)
    
    # Bullish FVG: previous high < next low
    out['fvg_bull'] = ((prev_high < next_low) & (out['close'] > out['open'])).astype(int)
    # Bearish FVG: previous low > next high
    out['fvg_bear'] = ((prev_low > next_high) & (out['close'] < out['open'])).astype(int)
    
    # FVG levels for retracement detection
    out['fvg_bull_high'] = out['high'].where(out['fvg_bull'] == 1, np.nan)
    out['fvg_bull_low'] = out['low'].where(out['fvg_bull'] == 1, np.nan)
    out['fvg_bear_high'] = out['high'].where(out['fvg_bear'] == 1, np.nan)
    out['fvg_bear_low'] = out['low'].where(out['fvg_bear'] == 1, np.nan)
    
    # Forward fill FVG levels
    out['fvg_bull_high'] = out['fvg_bull_high'].ffill()
    out['fvg_bull_low'] = out['fvg_bull_low'].ffill()
    out['fvg_bear_high'] = out['fvg_bear_high'].ffill()
    out['fvg_bear_low'] = out['fvg_bear_low'].ffill()
    
    # 4. RETRACEMENT TO FVG
    # Check if price retraces to FVG levels
    out['retrace_to_fvg_bull'] = ((out['low'] <= out['fvg_bull_high']) & 
                                  (out['low'] >= out['fvg_bull_low'])).astype(int)
    out['retrace_to_fvg_bear'] = ((out['high'] >= out['fvg_bear_low']) & 
                                  (out['high'] <= out['fvg_bear_high'])).astype(int)
    
    # 5. ENGULFING PATTERNS (Good Push)
    # Bullish engulfing: current candle engulfs previous bearish candle
    out['engulfing_bull'] = ((out['close'] > out['open']) &  # Current bullish
                            (out['close'].shift(1) < out['open'].shift(1)) &  # Previous bearish
                            (out['open'] < out['close'].shift(1)) &  # Current open < previous close
                            (out['close'] > out['open'].shift(1))).astype(int)  # Current close > previous open
    
    # Bearish engulfing: current candle engulfs previous bullish candle
    out['engulfing_bear'] = ((out['close'] < out['open']) &  # Current bearish
                            (out['close'].shift(1) > out['open'].shift(1)) &  # Previous bullish
                            (out['open'] > out['close'].shift(1)) &  # Current open > previous close
                            (out['close'] < out['open'].shift(1))).astype(int)  # Current close < previous open
    
    # 6. STRATEGY SEQUENCE DETECTION
    # Look for the complete sequence: Liquidity Sweep -> BOS -> FVG -> BOS -> FVG -> Retrace -> Entry
    
    # Track recent liquidity sweeps (within last 20 periods)
    out['recent_liq_sweep_bull'] = out['liq_sweep_bull'].rolling(20, min_periods=1).max()
    out['recent_liq_sweep_bear'] = out['liq_sweep_bear'].rolling(20, min_periods=1).max()
    
    # Track recent BOS events (within last 10 periods)
    out['recent_bos_bull'] = out['bos_bull'].rolling(10, min_periods=1).max()
    out['recent_bos_bear'] = out['bos_bear'].rolling(10, min_periods=1).max()
    
    # Track recent FVG events (within last 5 periods)
    out['recent_fvg_bull'] = out['fvg_bull'].rolling(5, min_periods=1).max()
    out['recent_fvg_bear'] = out['fvg_bear'].rolling(5, min_periods=1).max()
    
    # Complete strategy setup detection
    out['ict_setup_bull'] = ((out['recent_liq_sweep_bull'] == 1) & 
                            (out['recent_bos_bull'] == 1) & 
                            (out['recent_fvg_bull'] == 1) & 
                            (out['retrace_to_fvg_bull'] == 1) & 
                            (out['engulfing_bull'] == 1)).astype(int)
    
    out['ict_setup_bear'] = ((out['recent_liq_sweep_bear'] == 1) & 
                            (out['recent_bos_bear'] == 1) & 
                            (out['recent_fvg_bear'] == 1) & 
                            (out['retrace_to_fvg_bear'] == 1) & 
                            (out['engulfing_bear'] == 1)).astype(int)
    
    # Clean up intermediate columns
    intermediate_cols = ['pdh', 'pdl', 'h4_high', 'h4_low', 'h1_high', 'h1_low', 
                        'session_high', 'session_low', 'fvg_bull_high', 'fvg_bull_low',
                        'fvg_bear_high', 'fvg_bear_low', 'recent_liq_sweep_bull', 
                        'recent_liq_sweep_bear', 'recent_bos_bull', 'recent_bos_bear',
                        'recent_fvg_bull', 'recent_fvg_bear']
    
    # Keep only the essential features for the model
    essential_cols = [col for col in out.columns if col not in intermediate_cols]
    out = out[essential_cols]
    
    return out


def _add_smt_with_dxy(df: pd.DataFrame, dxy: pd.Series) -> pd.DataFrame:
    """Add SMT (smart money divergence) features with DXY.
    Aligns DXY to df index and computes divergence between price and DXY momentum.
    """
    out = df.copy()
    
    # Ensure DXY has the same index as the main dataframe
    if not dxy.index.equals(out.index):
        dxy = dxy.reindex(out.index, method='nearest').ffill().bfill()
    
    # Price and DXY momentum
    price_mom = out['close'].pct_change(5)
    dxy_mom = dxy.pct_change(5)
    
    # Handle NaN values in momentum calculations
    price_mom = price_mom.fillna(0)
    dxy_mom = dxy_mom.fillna(0)
    
    out['smt_div'] = (np.sign(price_mom) != np.sign(dxy_mom)).astype(int)
    out['smt_strength'] = (price_mom - dxy_mom)
    return out

