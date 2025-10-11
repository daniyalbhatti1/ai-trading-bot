"""LightGBM model pipeline for trading signals."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier


# Base feature columns (always available)
BASE_FEATURE_COLUMNS = [
    # TA
    'rsi', 'ema_fast', 'ema_slow', 'MACD_12_26_9', 'MACD_12_26_9_h', 'MACD_12_26_9_s', 'atr',
    # ICT - Liquidity Sweeps
    'liq_sweep_pdh', 'liq_sweep_pdl', 'liq_sweep_h4_high', 'liq_sweep_h4_low', 
    'liq_sweep_session_high', 'liq_sweep_session_low', 'liq_sweep_bull', 'liq_sweep_bear',
    # ICT - Break of Structure
    'bos_bull', 'bos_bear',
    # ICT - Fair Value Gaps
    'fvg_bull', 'fvg_bear', 'retrace_to_fvg_bull', 'retrace_to_fvg_bear',
    # ICT - Engulfing Patterns
    'engulfing_bull', 'engulfing_bear',
    # ICT - Complete Setup
    'ict_setup_bull', 'ict_setup_bear',
]

# Optional SMT features (only when DXY data is available)
SMT_FEATURE_COLUMNS = ['smt_div', 'smt_strength']

# All possible features
ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + SMT_FEATURE_COLUMNS


@dataclass
class LGBMConfig:
    objective: str = 'multiclass'
    num_class: int = 3  # 0=SHORT, 1=FLAT, 2=LONG
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


class LGBMTradingModel:
    def __init__(self, config: Optional[LGBMConfig] = None):
        self.config = config or LGBMConfig()
        self.model = LGBMClassifier(
            objective=self.config.objective,
            num_class=self.config.num_class,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Use only features that are actually available in the data
        available_features = [col for col in ALL_FEATURE_COLUMNS if col in X.columns]
        X = X[available_features].copy()
        
        # Replace inf values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Find valid rows (no NaN in features)
        valid_idx = X.dropna().index
        
        # Align both X and y to the same valid indices
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Ensure we have data
        if len(X) == 0:
            raise ValueError("No valid data after cleaning")
        
        self.model.fit(X, y)
        self.feature_names = available_features

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Use the same features that were used during training
        if not hasattr(self, 'feature_names'):
            # Fallback to base features if feature_names not set
            available_features = [col for col in BASE_FEATURE_COLUMNS if col in X.columns]
        else:
            available_features = [col for col in self.feature_names if col in X.columns]
        
        X = X[available_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(X)

    def save(self, path: str):
        # Save both model and feature names
        data = {
            'model': self.model,
            'feature_names': getattr(self, 'feature_names', BASE_FEATURE_COLUMNS)
        }
        joblib.dump(data, path)

    def load(self, path: str):
        data = joblib.load(path)
        if isinstance(data, dict):
            self.model = data['model']
            self.feature_names = data.get('feature_names', BASE_FEATURE_COLUMNS)
        else:
            # Backward compatibility with old format
            self.model = data
            self.feature_names = BASE_FEATURE_COLUMNS


