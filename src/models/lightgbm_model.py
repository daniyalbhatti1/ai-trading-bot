"""
Advanced LightGBM model for time series prediction in AI Trading Bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel

class LightGBMModel(BaseModel):
    """Advanced LightGBM model with feature engineering and hyperparameter optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # LightGBM specific parameters
        self.n_estimators = config.get('n_estimators', 1000)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.num_leaves = config.get('num_leaves', 31)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.reg_alpha = config.get('reg_alpha', 0)
        self.reg_lambda = config.get('reg_lambda', 1)
        self.min_child_samples = config.get('min_child_samples', 20)
        self.min_child_weight = config.get('min_child_weight', 0.001)
        self.random_state = config.get('random_state', 42)
        self.n_jobs = config.get('n_jobs', -1)
        self.verbosity = config.get('verbosity', -1)
        
        # Early stopping
        self.early_stopping_rounds = config.get('early_stopping_rounds', 50)
        self.eval_metric = config.get('eval_metric', 'rmse')
        
        # Feature engineering
        self.use_technical_indicators = config.get('use_technical_indicators', True)
        self.use_lag_features = config.get('use_lag_features', True)
        self.use_rolling_features = config.get('use_rolling_features', True)
        self.use_volume_features = config.get('use_volume_features', True)
        
        # Data preprocessing
        self.scaler = RobustScaler()
        self.feature_names = []
        self.feature_importance_ = None
        
        # Model components
        self.model = None
        self.best_iteration = None
        
    def _build_model(self) -> lgb.LGBMRegressor:
        """Build LightGBM model."""
        model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            objective='regression',
            metric=self.eval_metric
        )
        
        return model
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the LightGBM model."""
        
        # Prepare evaluation set
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        # Get best iteration
        self.best_iteration = self.model.best_iteration_
        
        # Training history
        training_history = {
            'best_iteration': self.best_iteration,
            'feature_importance': self.feature_importance_.tolist()
        }
        
        if eval_set:
            training_history['eval_results'] = self.model.evals_result_
        
        return training_history
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained LightGBM model."""
        return self.model.predict(X)
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10, log=True)
        }
    
    def _build_model_with_params(self, params: Dict[str, Any]) -> lgb.LGBMRegressor:
        """Build model with specific parameters."""
        # Update parameters
        self.n_estimators = params['n_estimators']
        self.max_depth = params['max_depth']
        self.learning_rate = params['learning_rate']
        self.num_leaves = params['num_leaves']
        self.subsample = params['subsample']
        self.colsample_bytree = params['colsample_bytree']
        self.reg_alpha = params['reg_alpha']
        self.reg_lambda = params['reg_lambda']
        self.min_child_samples = params['min_child_samples']
        self.min_child_weight = params['min_child_weight']
        
        return self._build_model()
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for feature engineering."""
        df = data.copy()
        
        # Price-based indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create lag features."""
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]
        
        df = data.copy()
        
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'high_lag_{lag}'] = df['high'].shift(lag)
            df[f'low_lag_{lag}'] = df['low'].shift(lag)
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features."""
        df = data.copy()
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            df[f'close_skew_{window}'] = df['close'].rolling(window=window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window=window).kurt()
            
            df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
        
        # Price ratios
        df['close_to_sma_20'] = df['close'] / df['close_mean_20']
        df['close_to_sma_50'] = df['close'] / df['close_mean_50']
        df['high_to_close'] = df['high'] / df['close']
        df['low_to_close'] = df['low'] / df['close']
        
        return df
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        df = data.copy()
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price-volume features
        df['price_volume'] = df['close'] * df['volume']
        df['price_volume_sma_20'] = df['price_volume'].rolling(window=20).mean()
        df['price_volume_ratio'] = df['price_volume'] / df['price_volume_sma_20']
        
        # On-Balance Volume (OBV)
        df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                                           np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
        
        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        return df
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data with feature engineering for LightGBM model."""
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Create features
        df = data.copy()
        
        if self.use_technical_indicators:
            df = self.create_technical_indicators(df)
        
        if self.use_lag_features:
            df = self.create_lag_features(df)
        
        if self.use_rolling_features:
            df = self.create_rolling_features(df)
        
        if self.use_volume_features:
            df = self.create_volume_features(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        self.feature_names = feature_columns
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.feature_importance_ is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance."""
        importance_df = self.get_feature_importance()
        if importance_df is None:
            self.logger.warning("No feature importance available")
            return
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance (LightGBM)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def get_shap_values(self, X: np.ndarray, sample_size: int = 100) -> np.ndarray:
        """Get SHAP values for model interpretability."""
        if self.model is None:
            raise ValueError("Model must be trained before getting SHAP values")
        
        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values
    
    def plot_shap_summary(self, X: np.ndarray, sample_size: int = 100) -> None:
        """Plot SHAP summary."""
        shap_values = self.get_shap_values(X, sample_size)
        
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title('LightGBM SHAP Summary')
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get LightGBM model summary."""
        summary = super().get_model_summary()
        
        if self.model is not None:
            summary.update({
                'best_iteration': self.best_iteration,
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names[:10],  # First 10 features
                'top_features': self.get_feature_importance().head(5).to_dict('records') if self.feature_importance_ is not None else None
            })
        
        return summary
