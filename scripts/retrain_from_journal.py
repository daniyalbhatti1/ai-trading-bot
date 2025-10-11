#!/usr/bin/env python3
"""Retrain ML model from trade journal (reinforcement learning)."""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.db import get_connection
from app.models.lgbm_model import LGBMTradingModel
from app.core.settings import config
from app.core.logger import logger

ROOT = Path(__file__).parent.parent


def load_journal_data(min_quality_score: float = 0, min_trades: int = 50):
    """Load trade journal data for retraining.
    
    Args:
        min_quality_score: Minimum quality score to include trade
        min_trades: Minimum number of trades required
    
    Returns:
        X (features), y (labels)
    """
    with get_connection() as conn:
        # Load all analyzed trades
        query = """
            SELECT * FROM trade_journal
            WHERE analyzed_at IS NOT NULL
        """
        if min_quality_score > 0:
            query += f" AND quality_score >= {min_quality_score}"
        
        df = pd.read_sql_query(query, conn)
    
    if len(df) < min_trades:
        raise ValueError(f"Not enough trades in journal: {len(df)} < {min_trades}")
    
    logger.info(f"Loaded {len(df)} trades from journal")
    
    # Extract features (entry context)
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_ema_fast', 'entry_ema_slow', 'entry_atr',
        'entry_liq_sweep_bull', 'entry_liq_sweep_bear',
        'entry_bos_bull', 'entry_bos_bear',
        'entry_fvg_bull', 'entry_fvg_bear',
        'entry_retrace_to_fvg_bull', 'entry_retrace_to_fvg_bear',
        'entry_engulfing_bull', 'entry_engulfing_bear',
        'entry_ict_setup_bull', 'entry_ict_setup_bear'
    ]
    
    X = df[feature_cols].copy()
    
    # Create labels based on actual trade outcome
    # Weight by quality score to prioritize learning from high-quality setups
    y = pd.Series(1, index=df.index)  # Default to FLAT
    
    # Use actual trade outcome and quality
    for idx, row in df.iterrows():
        if row['pnl'] > 0:  # Winning trade
            if row['side'] == 'LONG':
                y[idx] = 2  # LONG was correct
            else:
                y[idx] = 0  # SHORT was correct
        elif row['pnl'] < 0:  # Losing trade
            # Learn from mistakes: opposite direction would have been better
            if row['side'] == 'LONG':
                y[idx] = 0  # Should have been SHORT
            else:
                y[idx] = 2  # Should have been LONG
        else:
            y[idx] = 1  # FLAT (breakeven)
    
    # Weight by quality score (optional: duplicate high-quality trades)
    if 'quality_score' in df.columns:
        # Duplicate high-quality winning trades to emphasize learning
        high_quality = df[df['quality_score'] > 80]
        if len(high_quality) > 0:
            X = pd.concat([X, X.loc[high_quality.index]], ignore_index=True)
            y = pd.concat([y, y.loc[high_quality.index]], ignore_index=True)
            logger.info(f"Emphasized {len(high_quality)} high-quality trades")
    
    # Drop NaN values
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    logger.info(f"Final dataset: {len(X)} samples")
    logger.info(f"Label distribution: SHORT={sum(y==0)}, FLAT={sum(y==1)}, LONG={sum(y==2)}")
    
    return X, y


def evaluate_model(model: LGBMTradingModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    y_pred = model.model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_short': precision[0] if len(precision) > 0 else 0,
        'precision_flat': precision[1] if len(precision) > 1 else 0,
        'precision_long': precision[2] if len(precision) > 2 else 0,
        'recall_short': recall[0] if len(recall) > 0 else 0,
        'recall_flat': recall[1] if len(recall) > 1 else 0,
        'recall_long': recall[2] if len(recall) > 2 else 0,
        'f1_short': f1[0] if len(f1) > 0 else 0,
        'f1_flat': f1[1] if len(f1) > 1 else 0,
        'f1_long': f1[2] if len(f1) > 2 else 0
    }


def save_training_history(trades_used: int, metrics_before: dict, metrics_after: dict, model_version: str):
    """Save training history to database.
    
    Args:
        trades_used: Number of trades used for training
        metrics_before: Performance metrics before retraining
        metrics_after: Performance metrics after retraining
        model_version: Version identifier for the model
    """
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO model_training_history (
                training_date, model_version, trades_used,
                win_rate_before, win_rate_after,
                profit_factor_before, profit_factor_after,
                accuracy, precision_long, precision_short,
                recall_long, recall_short, f1_score,
                training_params, improvements, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            model_version,
            trades_used,
            metrics_before.get('win_rate', 0),
            metrics_after.get('win_rate', 0),
            metrics_before.get('profit_factor', 0),
            metrics_after.get('profit_factor', 0),
            metrics_after.get('accuracy', 0),
            metrics_after.get('precision_long', 0),
            metrics_after.get('precision_short', 0),
            metrics_after.get('recall_long', 0),
            metrics_after.get('recall_short', 0),
            metrics_after.get('f1_long', 0),
            json.dumps(config.get('model', {})),
            f"Accuracy: {metrics_after.get('accuracy', 0):.2%}",
            "Retrained from trade journal"
        ))
    
    logger.info("Training history saved to database")


def main():
    parser = argparse.ArgumentParser(description='Retrain model from trade journal')
    parser.add_argument('--min-quality', type=float, default=0, help='Minimum quality score (0-100)')
    parser.add_argument('--min-trades', type=int, default=50, help='Minimum number of trades required')
    parser.add_argument('--save', type=str, default='trained_models/lgbm_v2.pkl', help='Path to save retrained model')
    parser.add_argument('--backup-current', action='store_true', help='Backup current model before replacing')
    args = parser.parse_args()
    
    print("🧠 Reinforcement Learning: Retraining from Trade Journal")
    print("=" * 70)
    
    # Load journal data
    print(f"\n📚 Loading trades from journal (min quality: {args.min_quality})...")
    try:
        X, y = load_journal_data(args.min_quality, args.min_trades)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("   Run more backtests or lower --min-trades threshold")
        return 1
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Train new model
    print(f"\n🔄 Training new model...")
    model = LGBMTradingModel()
    model.fit(X_train, y_train)
    
    # Evaluate
    print(f"\n📈 Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\n✅ Model Performance:")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Precision (LONG): {metrics['precision_long']:.2%}")
    print(f"   Precision (SHORT): {metrics['precision_short']:.2%}")
    print(f"   Recall (LONG): {metrics['recall_long']:.2%}")
    print(f"   Recall (SHORT): {metrics['recall_short']:.2%}")
    print(f"   F1 Score (LONG): {metrics['f1_long']:.2%}")
    print(f"   F1 Score (SHORT): {metrics['f1_short']:.2%}")
    
    # Backup current model if requested
    if args.backup_current:
        current_model_path = ROOT / 'trained_models/lgbm.pkl'
        if current_model_path.exists():
            backup_path = ROOT / f'trained_models/lgbm_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            import shutil
            shutil.copy(current_model_path, backup_path)
            print(f"\n💾 Backed up current model to: {backup_path.name}")
    
    # Save new model
    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"\n💾 Model saved to: {save_path}")
    
    # Save training history
    model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_training_history(
        trades_used=len(X),
        metrics_before={},  # TODO: Load previous metrics
        metrics_after=metrics,
        model_version=model_version
    )
    
    print(f"\n✅ Retraining complete!")
    print(f"   Model version: {model_version}")
    print(f"   Learned from: {len(X)} trades")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

