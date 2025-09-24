"""
Base model class for all ML models in the AI Trading Bot.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import logging
import joblib
from pathlib import Path
import optuna
from optuna.integration import OptunaSearchCV
import warnings
warnings.filterwarnings('ignore')

class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        self.training_history = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model parameters
        self.lookback_window = config.get('lookback_window', 100)
        self.prediction_horizon = config.get('prediction_horizon', 24)
        self.feature_engineering = config.get('feature_engineering', True)
        self.hyperparameter_optimization = config.get('hyperparameter_optimization', True)
        self.cross_validation_folds = config.get('cross_validation_folds', 5)
        
        # Performance tracking
        self.metrics = {}
        self.predictions = []
        self.actual_values = []
        
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the specific model architecture."""
        pass
    
    @abstractmethod
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: Optional[np.ndarray] = None, 
                    y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the specific model."""
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        pass
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training/prediction.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Create features and targets
        features = []
        targets = []
        
        for i in range(self.lookback_window, len(data) - self.prediction_horizon + 1):
            # Features: lookback_window periods
            feature_window = data.iloc[i-self.lookback_window:i]
            features.append(feature_window.values.flatten())
            
            # Target: prediction_horizon periods ahead
            target_value = data.iloc[i + self.prediction_horizon - 1][target_column]
            targets.append(target_value)
        
        return np.array(features), np.array(targets)
    
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data.
        
        Args:
            data: Input time series data
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Training {self.__class__.__name__} model...")
        
        # Build model
        self.model = self._build_model()
        
        # Hyperparameter optimization if enabled
        if self.hyperparameter_optimization:
            self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Train model
        training_history = self._train_model(X_train, y_train, X_val, y_val)
        
        # Calculate metrics
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            self.metrics = self._calculate_metrics(y_val, val_predictions)
            self.logger.info(f"Validation metrics: {self.metrics}")
        
        self.is_trained = True
        self.training_history = training_history
        
        return training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self._predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions (if supported by model).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions array
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For regression models, return confidence intervals
            predictions = self.predict(X)
            # Simple confidence interval based on training residuals
            if hasattr(self, 'training_residuals'):
                std_error = np.std(self.training_residuals)
                return np.column_stack([
                    predictions - 1.96 * std_error,
                    predictions,
                    predictions + 1.96 * std_error
                ])
            return predictions
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Directional accuracy for trading
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)
        
        return metrics
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: Optional[np.ndarray] = None,
                                 y_val: Optional[np.ndarray] = None) -> None:
        """Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        self.logger.info("Starting hyperparameter optimization...")
        
        def objective(trial):
            # Get hyperparameter suggestions from subclass
            params = self._suggest_hyperparameters(trial)
            
            # Create model with suggested parameters
            model = self._build_model_with_params(params)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cross_validation_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluate
                val_pred = model.predict(X_val_fold)
                score = -mean_squared_error(y_val_fold, val_pred)  # Negative for maximization
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Update model with best parameters
        best_params = study.best_params
        self.model = self._build_model_with_params(best_params)
        self.logger.info(f"Best hyperparameters: {best_params}")
    
    @abstractmethod
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization."""
        pass
    
    @abstractmethod
    def _build_model_with_params(self, params: Dict[str, Any]) -> Any:
        """Build model with specific parameters."""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        return None
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'feature_importance_': self.feature_importance_
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.metrics = model_data.get('metrics', {})
        self.training_history = model_data.get('training_history', {})
        self.feature_importance_ = model_data.get('feature_importance_')
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information.
        
        Returns:
            Dictionary with model information
        """
        summary = {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance_available': self.get_feature_importance() is not None
        }
        
        if hasattr(self.model, 'get_params'):
            summary['model_params'] = self.model.get_params()
        
        return summary
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Cross-validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=self.cross_validation_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            self.train(X_train_fold, y_train_fold)
            
            # Evaluate
            val_pred = self.predict(X_val_fold)
            fold_metrics = self._calculate_metrics(y_val_fold, val_pred)
            cv_scores.append(fold_metrics)
        
        # Average metrics across folds
        avg_metrics = {}
        for metric in cv_scores[0].keys():
            avg_metrics[f'cv_{metric}'] = np.mean([score[metric] for score in cv_scores])
            avg_metrics[f'cv_{metric}_std'] = np.std([score[metric] for score in cv_scores])
        
        return avg_metrics
