#!/usr/bin/env python3
"""Train a LightGBM model using ICT + SMT features.

Usage:
  source venv/bin/activate
  python scripts/train_lgbm.py --symbols SPY QQQ GLD USO --lookahead 5 --save trained_models/lgbm.pkl
"""
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.settings import config
from app.data.db import get_connection
from app.signals.features import compute_features
from app.models import LGBMTradingModel


def load_symbol_df(symbol: str) -> pd.DataFrame:
    with get_connection() as conn:
        query = """
            SELECT ts, open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
            ORDER BY ts
        """
        df = pd.read_sql_query(query, conn, params=(symbol,))
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.set_index('ts')
    return df


def make_dataset(symbols, lookahead: int, dxy: pd.Series | None) -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    for sym in symbols:
        df = load_symbol_df(sym)
        cfg = config.copy()
        if dxy is not None:
            cfg.setdefault('external', {})['DXY'] = dxy
        
        # Compute features first
        feats = compute_features(df, cfg)
        
        # Create labels BEFORE dropping NaN
        # Multi-class labels: 0=SHORT (down), 1=FLAT (no significant move), 2=LONG (up)
        future = feats['close'].shift(-lookahead)
        pct_change = (future - feats['close']) / feats['close']
        
        # Define thresholds for significant moves (e.g., 0.5% = 0.005)
        threshold = 0.005
        
        # Create 3-class labels
        y = pd.Series(1, index=feats.index)  # Default to FLAT (1)
        y[pct_change > threshold] = 2  # UP = LONG (2)
        y[pct_change < -threshold] = 0  # DOWN = SHORT (0)
        
        # Now drop NaN from both features and labels, keeping them aligned
        valid_idx = feats.dropna().index
        feats = feats.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Ensure we have enough data
        if len(feats) < 50:
            print(f"Warning: Not enough data for {sym} ({len(feats)} rows)")
            continue
            
        feats['symbol'] = sym
        frames.append((feats, y))

    if not frames:
        raise ValueError("No valid data found for any symbol")
        
    X = pd.concat([f for f, _ in frames], axis=0)
    y = pd.concat([y for _, y in frames], axis=0)
    
    # Drop columns not in features
    if 'symbol' in X:
        X = X.drop(columns=['symbol'])
    
    # Final alignment check
    if len(X) != len(y):
        print(f"Warning: X length ({len(X)}) != y length ({len(y)})")
        # Align them
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
    
    return X, y


def load_dxy_series() -> pd.Series | None:
    # Option A: If DXY candles exist in DB as symbol 'DXY'
    try:
        df = load_symbol_df('DXY')
        return df['close']
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=config.get('universe', ['SPY','QQQ','GLD','USO']))
    parser.add_argument('--lookahead', type=int, default=5)
    parser.add_argument('--save', type=str, default='trained_models/lgbm.pkl')
    args = parser.parse_args()

    dxy = load_dxy_series()
    X, y = make_dataset(args.symbols, args.lookahead, dxy)

    model = LGBMTradingModel()
    model.fit(X, y)

    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()


