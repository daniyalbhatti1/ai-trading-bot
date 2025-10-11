"""Hybrid signal: rules + optional ML (LightGBM) probability blending."""
from typing import Tuple, Optional
from app.core.logger import logger
from app.models.lgbm_model import LGBMTradingModel


def rules_signal(latest_row: dict, cfg: dict, ml_model: Optional[LGBMTradingModel] = None) -> Tuple[str, float, str]:
    """Generate trading signals based on sophisticated ICT strategy:
    
    Strategy sequence:
    1. Find liquidity sweep (PDL/PDH - 4h/1h highs/lows, previous session highs/lows)
    2. Wait for BOS that creates FVG
    3. Wait for another BOS that creates another FVG
    4. Retrace to 2nd FVG
    5. Enter on reaction with good push (engulfing preferred)
    
    Args:
        latest_row: Latest candle with indicators (as dict)
        cfg: Configuration
        ml_model: Optional ML model for additional confirmation
    
    Returns:
        Tuple of (side, confidence, reason)
    """
    try:
        # Extract ICT features
        ict_setup_bull = latest_row.get('ict_setup_bull', 0)
        ict_setup_bear = latest_row.get('ict_setup_bear', 0)
        
        # Individual ICT components for partial setups
        liq_sweep_bull = latest_row.get('liq_sweep_bull', 0)
        liq_sweep_bear = latest_row.get('liq_sweep_bear', 0)
        bos_bull = latest_row.get('bos_bull', 0)
        bos_bear = latest_row.get('bos_bear', 0)
        fvg_bull = latest_row.get('fvg_bull', 0)
        fvg_bear = latest_row.get('fvg_bear', 0)
        retrace_to_fvg_bull = latest_row.get('retrace_to_fvg_bull', 0)
        retrace_to_fvg_bear = latest_row.get('retrace_to_fvg_bear', 0)
        engulfing_bull = latest_row.get('engulfing_bull', 0)
        engulfing_bear = latest_row.get('engulfing_bear', 0)
        
        # Traditional indicators for confirmation
        rsi = latest_row.get('rsi', 50)
        ema_fast = latest_row.get('ema_fast', 0)
        ema_slow = latest_row.get('ema_slow', 0)
        
        # Get MACD (handle different naming conventions)
        macd_col = None
        for col in latest_row.keys():
            if 'MACD_' in str(col) and 'MACD_h' not in str(col) and 'MACD_s' not in str(col):
                macd_col = col
                break
        
        macd = latest_row[macd_col] if macd_col else 0
        
        # Get thresholds from config
        thresholds = cfg.get('strategy', {}).get('thresholds', {})
        rsi_oversold = thresholds.get('rsi_oversold', 30)
        rsi_overbought = thresholds.get('rsi_overbought', 70)
        
        # ML Model Integration (if available)
        if ml_model is not None:
            import pandas as pd
            X = pd.DataFrame([latest_row])
            try:
                proba = ml_model.predict_proba(X)[0]
                # proba = [SHORT_prob, FLAT_prob, LONG_prob]
                short_p = float(proba[0])
                flat_p = float(proba[1])
                long_p = float(proba[2])
                
                ml_confidence_min = cfg['strategy'].get('ml_confidence_min', 0.6)
                
                # Find the highest probability class
                max_prob = max(short_p, flat_p, long_p)
                
                if max_prob >= ml_confidence_min:
                    if long_p == max_prob:
                        return "LONG", long_p, f"ML long probability ({long_p:.2f})"
                    elif short_p == max_prob:
                        return "SHORT", short_p, f"ML short probability ({short_p:.2f})"
                    else:
                        # FLAT prediction - fall through to ICT rules
                        pass
            except Exception as _:
                logger.warning(f"ML prediction failed: {_}")
                pass  # Fallback to ICT rules if ML fails

        # ICT STRATEGY IMPLEMENTATION
        
        # 1. COMPLETE ICT SETUP (Highest confidence)
        if ict_setup_bull == 1:
            # Additional confirmation with traditional indicators
            if rsi > 40 and rsi < 70:  # Not extreme, room to move
                return "LONG", 0.90, "Complete ICT bullish setup + RSI confirmation"
            else:
                return "LONG", 0.85, "Complete ICT bullish setup"
        
        if ict_setup_bear == 1:
            # Additional confirmation with traditional indicators
            if rsi > 30 and rsi < 60:  # Not extreme, room to move
                return "SHORT", 0.90, "Complete ICT bearish setup + RSI confirmation"
            else:
                return "SHORT", 0.85, "Complete ICT bearish setup"
        
        # 2. PARTIAL ICT SETUPS (High confidence)
        
        # Liquidity sweep + BOS + FVG + Retrace (missing engulfing)
        if (liq_sweep_bull == 1 and bos_bull == 1 and fvg_bull == 1 and 
            retrace_to_fvg_bull == 1 and engulfing_bull == 0):
            if rsi > 35 and rsi < 65:  # Reasonable RSI level
                return "LONG", 0.80, "ICT setup: Liq sweep + BOS + FVG + Retrace (waiting for push)"
        
        if (liq_sweep_bear == 1 and bos_bear == 1 and fvg_bear == 1 and 
            retrace_to_fvg_bear == 1 and engulfing_bear == 0):
            if rsi > 35 and rsi < 65:  # Reasonable RSI level
                return "SHORT", 0.80, "ICT setup: Liq sweep + BOS + FVG + Retrace (waiting for push)"
        
        # 3. ENGULFING PATTERNS AT FVG LEVELS (Medium-high confidence)
        
        # Bullish engulfing at FVG retracement
        if engulfing_bull == 1 and retrace_to_fvg_bull == 1:
            if rsi > 40 and rsi < 70:
                return "LONG", 0.75, "Bullish engulfing at FVG retracement"
        
        # Bearish engulfing at FVG retracement
        if engulfing_bear == 1 and retrace_to_fvg_bear == 1:
            if rsi > 30 and rsi < 60:
                return "SHORT", 0.75, "Bearish engulfing at FVG retracement"
        
        # 4. LIQUIDITY SWEEP + BOS (Medium confidence)
        
        # Liquidity sweep followed by BOS
        if liq_sweep_bull == 1 and bos_bull == 1:
            if rsi > 45 and rsi < 70:
                return "LONG", 0.70, "Liquidity sweep + BOS bullish"
        
        if liq_sweep_bear == 1 and bos_bear == 1:
            if rsi > 30 and rsi < 55:
                return "SHORT", 0.70, "Liquidity sweep + BOS bearish"
        
        # 5. FALLBACK TO TRADITIONAL STRATEGY (Lower confidence)
        
        # Strong trend with momentum
        is_uptrend = ema_fast > ema_slow
        is_downtrend = ema_fast < ema_slow
        
        if (is_uptrend and macd > 0 and rsi > 50 and rsi < 70):
            return "LONG", 0.65, "Traditional: Strong bullish trend"
        
        if (is_downtrend and macd < 0 and rsi < 50 and rsi > 30):
            return "SHORT", 0.65, "Traditional: Strong bearish trend"
        
        # RSI extremes with trend support
        if rsi <= rsi_oversold and is_uptrend:
            return "LONG", 0.60, "Traditional: RSI oversold + uptrend"
        
        if rsi >= rsi_overbought and is_downtrend:
            return "SHORT", 0.60, "Traditional: RSI overbought + downtrend"
        
        # No high-probability setup found
        return "FLAT", 0.0, "No ICT or traditional setup"
        
    except Exception as e:
        logger.error(f"Error in rules_signal: {e}")
        return "FLAT", 0.0, f"Error: {str(e)}"


def check_exit_signal(position_side: str, latest_row: dict, cfg: dict) -> Tuple[bool, str]:
    """Check if position should be exited based on technical signals.
    
    Args:
        position_side: Current position side (LONG or SHORT)
        latest_row: Latest candle with indicators
        cfg: Configuration
    
    Returns:
        Tuple of (should_exit, reason)
    """
    try:
        rsi = latest_row.get('rsi', 50)
        ema_fast = latest_row.get('ema_fast', 0)
        ema_slow = latest_row.get('ema_slow', 0)
        
        # Get MACD
        macd_col = None
        for col in latest_row.keys():
            if 'MACD_' in str(col) and 'MACD_h' not in str(col) and 'MACD_s' not in str(col):
                macd_col = col
                break
        
        macd = latest_row[macd_col] if macd_col else 0
        
        # Exit LONG positions
        if position_side == "LONG":
            # RSI overbought + trend weakening = exit
            if rsi > 75:
                return True, "Exit: RSI overbought (75+)"
            
            # Trend reversal (EMA crossover down) = exit
            if ema_fast < ema_slow and macd < 0:
                return True, "Exit: Trend reversal (EMA cross down + MACD negative)"
            
            # MACD turning negative in strong overbought = exit
            if rsi > 70 and macd < 0:
                return True, "Exit: Overbought + MACD bearish"
        
        # Exit SHORT positions
        elif position_side == "SHORT":
            # RSI oversold + trend weakening = exit
            if rsi < 25:
                return True, "Exit: RSI oversold (25-)"
            
            # Trend reversal (EMA crossover up) = exit
            if ema_fast > ema_slow and macd > 0:
                return True, "Exit: Trend reversal (EMA cross up + MACD positive)"
            
            # MACD turning positive in strong oversold = exit
            if rsi < 30 and macd > 0:
                return True, "Exit: Oversold + MACD bullish"
        
        return False, "Hold position"
        
    except Exception as e:
        logger.error(f"Error in check_exit_signal: {e}")
        return False, f"Error: {str(e)}"
