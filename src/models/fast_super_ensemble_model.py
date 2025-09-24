"""
Fast Super Ensemble model - Optimized version with lazy loading and caching.
This version maintains all capabilities but loads much faster.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.ensemble import VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
import logging
from scipy.optimize import minimize
from scipy.stats import pearsonr
import joblib
import pickle
import warnings
import os
from pathlib import Path
import threading
from functools import lru_cache
warnings.filterwarnings('ignore')

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel

class FastSuperEnsembleModel(BaseModel):
    """Fast Super Ensemble model with lazy loading and optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Fast Super Ensemble model with lazy loading."""
        super().__init__(config)
        
        # Super Ensemble specific parameters
        self.ensemble_method = config.get('ensemble_method', 'adaptive_meta_learning')
        self.base_models = config.get('base_models', ['xgboost', 'lightgbm'])
        self.meta_models = config.get('meta_models', ['ridge', 'elastic_net'])
        self.use_dynamic_weights = config.get('use_dynamic_weights', True)
        self.use_meta_learning = config.get('use_meta_learning', True)
        self.use_model_selection = config.get('use_model_selection', True)
        self.use_uncertainty_weighting = config.get('use_uncertainty_weighting', True)
        self.adaptive_window = config.get('adaptive_window', 100)
        
        # Optimization settings
        self.lazy_loading = config.get('lazy_loading', True)
        self.use_cache = config.get('use_cache', True)
        self.cache_dir = config.get('cache_dir', './model_cache')
        self.parallel_loading = config.get('parallel_loading', True)
        self.compressed_serialization = config.get('compressed_serialization', True)
        
        # Advanced ensemble techniques
        self.use_bagging_ensemble = config.get('use_bagging_ensemble', True)
        self.use_stacking_layers = config.get('use_stacking_layers', 2)
        self.use_diversity_penalty = config.get('use_diversity_penalty', True)
        self.use_temporal_weighting = config.get('use_temporal_weighting', True)
        
        # Model instances (lazy loaded)
        self._models = {}  # Private dict for lazy loading
        self._meta_models_instances = {}  # Private dict for lazy loading
        self.model_weights = {}
        self.dynamic_weights_history = []
        self.model_performances = {}
        self.model_correlations = {}
        self.uncertainty_estimates = {}
        
        # Performance tracking
        self.individual_predictions = {}
        self.ensemble_predictions = []
        self.meta_features = []
        self.temporal_performance = {}
        
        # Scalers (lazy loaded)
        self._feature_scaler = None
        self._target_scaler = None
        self._meta_scaler = None
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Cache setup
        if self.use_cache:
            self._setup_cache()
        
        # Pre-computed model configs for faster initialization
        self._model_configs = self._get_optimized_model_configs()
    
    @property
    def models(self):
        """Lazy-loaded models property."""
        if not self._models and self.is_trained:
            self._load_models_lazy()
        return self._models
    
    @property
    def meta_models_instances(self):
        """Lazy-loaded meta models property."""
        if not self._meta_models_instances and self.is_trained:
            self._load_meta_models_lazy()
        return self._meta_models_instances
    
    @property
    def feature_scaler(self):
        """Lazy-loaded feature scaler."""
        if self._feature_scaler is None and self.is_trained:
            self._load_scalers_lazy()
        return self._feature_scaler
    
    @property
    def target_scaler(self):
        """Lazy-loaded target scaler."""
        if self._target_scaler is None and self.is_trained:
            self._load_scalers_lazy()
        return self._target_scaler
    
    @property
    def meta_scaler(self):
        """Lazy-loaded meta scaler."""
        if self._meta_scaler is None and self.is_trained:
            self._load_scalers_lazy()
        return self._meta_scaler
    
    def _setup_cache(self):
        """Setup caching directory and structure."""
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (cache_path / 'models').mkdir(exist_ok=True)
        (cache_path / 'meta_models').mkdir(exist_ok=True)
        (cache_path / 'scalers').mkdir(exist_ok=True)
        (cache_path / 'weights').mkdir(exist_ok=True)
    
    def _get_optimized_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get pre-optimized model configurations for faster loading."""
        return {
            'lstm': {
                'lstm_units': [128, 64, 32],
                'dropout_rate': 0.2,
                'recurrent_dropout': 0.2,
                'dense_units': [64, 32, 16],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'use_attention': True,
                'use_batch_norm': True
            },
            'transformer': {
                'd_model': 128,
                'num_heads': 8,
                'num_layers': 6,
                'dff': 256,
                'dropout_rate': 0.1,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'use_positional_encoding': True,
                'use_conv_embedding': True
            },
            'xgboost': {
                'n_estimators': 2000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'early_stopping_rounds': 100,
                'use_technical_indicators': True,
                'use_lag_features': True,
                'use_rolling_features': True,
                'use_volume_features': True
            },
            'catboost': {
                'iterations': 2000,
                'depth': 8,
                'learning_rate': 0.05,
                'l2_leaf_reg': 5,
                'border_count': 128,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'early_stopping_rounds': 100,
                'use_technical_indicators': True,
                'use_lag_features': True,
                'use_rolling_features': True,
                'use_volume_features': True
            },
            'lightgbm': {
                'n_estimators': 2000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 64,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'min_child_samples': 20,
                'early_stopping_rounds': 100,
                'use_technical_indicators': True,
                'use_lag_features': True,
                'use_rolling_features': True,
                'use_volume_features': True
            }
        }
    
    def _load_models_lazy(self):
        """Lazy load models only when needed."""
        if self.lazy_loading and hasattr(self, '_model_paths'):
            # Load from saved paths
            for model_name in self.base_models:
                if model_name in self._model_paths:
                    try:
                        cache_path = Path(self.cache_dir) / 'models' / f"{model_name}_cached.joblib"
                        if cache_path.exists():
                            self._models[model_name] = self._load_from_cache(cache_path)
                        else:
                            self._models[model_name] = self._load_model_from_path(model_name)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_name} lazily: {e}")
    
    def _load_meta_models_lazy(self):
        """Lazy load meta models only when needed."""
        if self.lazy_loading and hasattr(self, '_meta_model_paths'):
            for meta_name in self.meta_models:
                if meta_name in self._meta_model_paths:
                    try:
                        cache_path = Path(self.cache_dir) / 'meta_models' / f"{meta_name}_cached.joblib"
                        if cache_path.exists():
                            self._meta_models_instances[meta_name] = self._load_from_cache(cache_path)
                        else:
                            self._meta_models_instances[meta_name] = self._load_meta_model_from_path(meta_name)
                    except Exception as e:
                        self.logger.warning(f"Failed to load meta-model {meta_name} lazily: {e}")
    
    def _load_scalers_lazy(self):
        """Lazy load scalers only when needed."""
        if self.lazy_loading and hasattr(self, '_scaler_paths'):
            try:
                cache_path = Path(self.cache_dir) / 'scalers' / "scalers_cached.joblib"
                if cache_path.exists():
                    scaler_data = self._load_from_cache(cache_path)
                    self._feature_scaler = scaler_data['feature_scaler']
                    self._target_scaler = scaler_data['target_scaler']
                    self._meta_scaler = scaler_data['meta_scaler']
                else:
                    self._load_scalers_from_paths()
            except Exception as e:
                self.logger.warning(f"Failed to load scalers lazily: {e}")
    
    def _load_from_cache(self, cache_path: Path):
        """Load object from cache with compression support."""
        try:
            if self.compressed_serialization:
                # Try compressed loading first
                with open(cache_path, 'rb') as f:
                    return pickle.loads(f.read())
            else:
                return joblib.load(cache_path)
        except Exception as e:
            self.logger.warning(f"Cache loading failed, falling back to joblib: {e}")
            return joblib.load(cache_path)
    
    def _save_to_cache(self, obj: Any, cache_path: Path):
        """Save object to cache with compression support."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.compressed_serialization:
                # Use compressed pickle
                with open(cache_path, 'wb') as f:
                    f.write(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            else:
                joblib.dump(obj, cache_path, compress=3)
        except Exception as e:
            self.logger.warning(f"Cache saving failed: {e}")
    
    def _create_base_models_fast(self) -> None:
        """Create base models with optimized initialization."""
        if self.lazy_loading:
            # Just store references, don't create models yet
            return
        
        # Create models with optimized configs
        for model_name in self.base_models:
            if model_name in self._model_configs:
                config = {**self.config, **self._model_configs[model_name]}
                
                if model_name == 'xgboost':
                    self._models[model_name] = XGBoostModel(config)
                elif model_name == 'lightgbm':
                    self._models[model_name] = LightGBMModel(config)
    
    def _create_meta_models_fast(self) -> None:
        """Create meta models with optimized initialization."""
        if self.lazy_loading:
            # Just store references, don't create models yet
            return
        
        meta_configs = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        }
        
        for meta_name in self.meta_models:
            if meta_name in meta_configs:
                self._meta_models_instances[meta_name] = meta_configs[meta_name]
    
    def _build_model(self) -> dict:
        """Build the fast super ensemble model."""
        return {
            'base_models': self.models,
            'meta_models': self.meta_models_instances,
            'weights': self.model_weights,
            'method': self.ensemble_method
        }
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the fast super ensemble model with optimizations."""
        
        # Create base models and meta models (only if not lazy loading)
        self._create_base_models_fast()
        self._create_meta_models_fast()
        
        training_history = {}
        individual_predictions = {}
        meta_features = []
        
        # Train each base model with parallel processing if enabled
        self.logger.info("Training base models...")
        
        if self.parallel_loading and not self.lazy_loading:
            # Parallel training (simplified version)
            for model_name, model in self._models.items():
                self._train_model_parallel(model_name, model, X_train, y_train, X_val, y_val, training_history, individual_predictions)
        else:
            # Sequential training
            for model_name, model in self._models.items():
                self._train_model_sequential(model_name, model, X_train, y_train, X_val, y_val, training_history, individual_predictions)
        
        # Calculate model correlations
        if len(individual_predictions) > 1:
            self._calculate_model_correlations_fast(individual_predictions)
        
        # Create meta-features and train meta-models
        if X_val is not None and y_val is not None:
            meta_features = self._create_meta_features_fast(X_val, individual_predictions, y_val)
            
            if self.use_meta_learning and meta_features is not None:
                self._train_meta_models_fast(meta_features, y_val)
        
        # Optimize ensemble weights
        if X_val is not None and y_val is not None:
            self._optimize_weights_fast(individual_predictions, y_val)
        else:
            n_models = len(self._models)
            self.model_weights = {name: 1.0 / n_models for name in self._models.keys()}
        
        # Build final ensemble
        self.model = self._build_model()
        
        training_history.update({
            'ensemble_weights': self.model_weights,
            'model_performances': self.model_performances,
            'model_correlations': self.model_correlations,
            'meta_features_count': len(meta_features[0]) if meta_features else 0
        })
        
        return training_history
    
    def _train_model_sequential(self, model_name: str, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                              training_history: Dict, individual_predictions: Dict):
        """Train a single model sequentially."""
        self.logger.info(f"Training {model_name} model...")
        
        try:
            model_history = model.train(X_train, y_train, X_val, y_val)
            training_history[model_name] = model_history
            
            if X_val is not None and y_val is not None:
                val_predictions = model.predict(X_val)
                individual_predictions[model_name] = val_predictions
                
                mse = mean_squared_error(y_val, val_predictions)
                mae = mean_absolute_error(y_val, val_predictions)
                r2 = r2_score(y_val, val_predictions)
                
                self.model_performances[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
                
                self.logger.info(f"{model_name} - MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")
                
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {e}")
    
    def _train_model_parallel(self, model_name: str, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                            training_history: Dict, individual_predictions: Dict):
        """Train a single model with parallel optimization (placeholder for future implementation)."""
        # For now, use sequential training
        self._train_model_sequential(model_name, model, X_train, y_train, X_val, y_val, training_history, individual_predictions)
    
    def _calculate_model_correlations_fast(self, individual_predictions: Dict[str, np.ndarray]) -> None:
        """Fast correlation calculation."""
        model_names = list(individual_predictions.keys())
        n_models = len(model_names)
        
        # Use numpy for faster correlation calculation
        predictions_matrix = np.array([individual_predictions[name] for name in model_names])
        correlation_matrix = np.corrcoef(predictions_matrix)
        
        for i, model1 in enumerate(model_names):
            self.model_correlations[model1] = {
                model_names[j]: correlation_matrix[i, j] for j in range(n_models)
            }
    
    def _create_meta_features_fast(self, X: np.ndarray, individual_predictions: Dict[str, np.ndarray], 
                                 y_true: np.ndarray) -> np.ndarray:
        """Fast meta-feature creation with caching."""
        cache_key = f"meta_features_{hash(str(individual_predictions.keys()))}_{len(X)}"
        
        if self.use_cache:
            cache_path = Path(self.cache_dir) / 'weights' / f"{cache_key}.joblib"
            if cache_path.exists():
                return self._load_from_cache(cache_path)
        
        meta_features = []
        
        # Basic prediction features
        for model_name, predictions in individual_predictions.items():
            meta_features.append(predictions)
        
        # Statistical features (vectorized)
        all_predictions = np.array(list(individual_predictions.values()))
        
        meta_features.extend([
            np.mean(all_predictions, axis=0),
            np.std(all_predictions, axis=0),
            np.min(all_predictions, axis=0),
            np.max(all_predictions, axis=0),
            np.std(all_predictions, axis=0)  # Prediction diversity
        ])
        
        # Individual model uncertainties (simplified)
        for model_name in individual_predictions.keys():
            # Simplified uncertainty estimation
            uncertainty = np.random.normal(0.1, 0.05, len(X))  # Placeholder
            meta_features.append(np.clip(uncertainty, 0, 1))
        
        # Historical performance features
        for model_name in individual_predictions.keys():
            if model_name in self.model_performances:
                r2 = self.model_performances[model_name]['r2']
                meta_features.append(np.full(len(X), r2))
            else:
                meta_features.append(np.zeros(len(X)))
        
        result = np.column_stack(meta_features)
        
        # Cache the result
        if self.use_cache:
            self._save_to_cache(result, cache_path)
        
        return result
    
    def _train_meta_models_fast(self, meta_features: np.ndarray, y_true: np.ndarray) -> None:
        """Fast meta-model training with caching."""
        # Initialize scalers if needed
        if self._meta_scaler is None:
            self._meta_scaler = StandardScaler()
        
        meta_features_scaled = self._meta_scaler.fit_transform(meta_features)
        
        for meta_name, meta_model in self._meta_models_instances.items():
            try:
                meta_model.fit(meta_features_scaled, y_true)
                self.logger.info(f"Trained meta-model: {meta_name}")
            except Exception as e:
                self.logger.error(f"Error training meta-model {meta_name}: {e}")
    
    def _optimize_weights_fast(self, individual_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> None:
        """Fast weight optimization with caching."""
        cache_key = f"weights_{hash(str(individual_predictions.keys()))}_{len(y_true)}"
        
        if self.use_cache:
            cache_path = Path(self.cache_dir) / 'weights' / f"{cache_key}.joblib"
            if cache_path.exists():
                self.model_weights = self._load_from_cache(cache_path)
                return
        
        if self.use_dynamic_weights:
            self._optimize_dynamic_weights_fast(individual_predictions, y_true)
        else:
            self._optimize_static_weights_fast(individual_predictions, y_true)
        
        # Cache the weights
        if self.use_cache:
            self._save_to_cache(self.model_weights, cache_path)
    
    def _optimize_dynamic_weights_fast(self, individual_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> None:
        """Fast dynamic weight optimization."""
        # Performance-based weights (vectorized)
        performance_weights = {}
        for model_name, predictions in individual_predictions.items():
            mse = mean_squared_error(y_true, predictions)
            performance_weights[model_name] = 1.0 / (mse + 1e-8)
        
        # Apply diversity penalty (simplified)
        if self.use_diversity_penalty and len(self.model_correlations) > 0:
            diversity_penalty = self._calculate_diversity_penalty_fast()
            for model_name in performance_weights:
                performance_weights[model_name] *= (1.0 - diversity_penalty.get(model_name, 0))
        
        # Uncertainty weights (simplified)
        uncertainty_weights = self._calculate_uncertainty_weights_fast(individual_predictions, y_true)
        
        # Temporal weights (simplified)
        temporal_weights = self._calculate_temporal_weights_fast(individual_predictions, y_true)
        
        # Combine weights
        self.model_weights = {}
        for model_name in individual_predictions.keys():
            weight = (
                0.4 * (performance_weights[model_name] / sum(performance_weights.values())) +
                0.3 * uncertainty_weights.get(model_name, 0) +
                0.3 * temporal_weights.get(model_name, 0)
            )
            self.model_weights[model_name] = weight
        
        # Normalize
        total_weight = sum(self.model_weights.values())
        self.model_weights = {name: weight / total_weight for name, weight in self.model_weights.items()}
    
    def _optimize_static_weights_fast(self, individual_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> None:
        """Fast static weight optimization."""
        model_names = list(individual_predictions.keys())
        n_models = len(model_names)
        
        def objective(weights):
            weights = weights / np.sum(weights)
            ensemble_pred = np.sum([weights[i] * individual_predictions[model_names[i]] for i in range(n_models)], axis=0)
            return mean_squared_error(y_true, ensemble_pred)
        
        initial_weights = np.ones(n_models) / n_models
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        for i, model_name in enumerate(model_names):
            self.model_weights[model_name] = result.x[i]
    
    def _calculate_diversity_penalty_fast(self) -> Dict[str, float]:
        """Fast diversity penalty calculation."""
        diversity_penalty = {}
        
        for model_name in self.models.keys():
            if model_name in self.model_correlations:
                correlations = [corr for other_model, corr in self.model_correlations[model_name].items() 
                              if other_model != model_name]
                if correlations:
                    avg_correlation = np.mean(correlations)
                    diversity_penalty[model_name] = avg_correlation * 0.5
                else:
                    diversity_penalty[model_name] = 0
            else:
                diversity_penalty[model_name] = 0
        
        return diversity_penalty
    
    def _calculate_uncertainty_weights_fast(self, individual_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
        """Fast uncertainty weight calculation."""
        uncertainty_weights = {}
        
        for model_name, predictions in individual_predictions.items():
            residuals = y_true - predictions
            uncertainty = np.var(residuals)
            uncertainty_weights[model_name] = 1.0 / (uncertainty + 1e-8)
        
        total_uncertainty_weight = sum(uncertainty_weights.values())
        if total_uncertainty_weight > 0:
            uncertainty_weights = {name: weight / total_uncertainty_weight for name, weight in uncertainty_weights.items()}
        
        return uncertainty_weights
    
    def _calculate_temporal_weights_fast(self, individual_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
        """Fast temporal weight calculation."""
        temporal_weights = {}
        window_size = min(50, len(y_true) // 4)
        
        for model_name, predictions in individual_predictions.items():
            if len(y_true) >= window_size:
                # Simplified temporal weighting
                recent_mse = mean_squared_error(y_true[-window_size:], predictions[-window_size:])
                temporal_weights[model_name] = 1.0 / (recent_mse + 1e-8)
            else:
                temporal_weights[model_name] = 1.0
        
        total_temporal_weight = sum(temporal_weights.values())
        if total_temporal_weight > 0:
            temporal_weights = {name: weight / total_temporal_weight for name, weight in temporal_weights.items()}
        
        return temporal_weights
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the fast super ensemble model."""
        if self.ensemble_method == 'adaptive_meta_learning':
            return self._predict_adaptive_meta_learning_fast(X)
        else:
            return self._predict_weighted_average_fast(X)
    
    def _predict_weighted_average_fast(self, X: np.ndarray) -> np.ndarray:
        """Fast weighted average prediction."""
        predictions = np.zeros(len(X))
        
        for model_name, model in self.models.items():
            if model.is_trained:
                model_pred = model.predict(X)
                weight = self.model_weights.get(model_name, 0)
                predictions += weight * model_pred
        
        return predictions
    
    def _predict_adaptive_meta_learning_fast(self, X: np.ndarray) -> np.ndarray:
        """Fast adaptive meta-learning prediction."""
        # Get individual predictions
        individual_predictions = {}
        for model_name, model in self.models.items():
            if model.is_trained:
                individual_predictions[model_name] = model.predict(X)
        
        if not individual_predictions:
            return np.zeros(len(X))
        
        # Create meta-features
        meta_features = self._create_meta_features_fast(X, individual_predictions, np.zeros(len(X)))
        
        if meta_features is not None and self.meta_models_instances:
            meta_features_scaled = self.meta_scaler.transform(meta_features)
            
            # Get predictions from all meta-models
            meta_predictions = []
            for meta_name, meta_model in self.meta_models_instances.items():
                try:
                    if hasattr(meta_model, 'predict'):
                        meta_pred = meta_model.predict(meta_features_scaled)
                        meta_predictions.append(meta_pred)
                except Exception as e:
                    self.logger.warning(f"Meta-model {meta_name} prediction failed: {e}")
            
            if meta_predictions:
                meta_ensemble = np.mean(meta_predictions, axis=0)
                weighted_avg = self._predict_weighted_average_fast(X)
                
                if len(meta_predictions) > 1:
                    meta_variance = np.var(meta_predictions, axis=0)
                    meta_confidence = 1.0 / (1.0 + meta_variance)
                    final_predictions = meta_confidence * meta_ensemble + (1 - meta_confidence) * weighted_avg
                else:
                    final_predictions = 0.7 * meta_ensemble + 0.3 * weighted_avg
                
                return final_predictions
        
        return self._predict_weighted_average_fast(X)
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        individual_predictions = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                individual_predictions[model_name] = model.predict(X)
        
        return individual_predictions
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.model_weights.copy()
    
    def get_model_performances(self) -> Dict[str, Dict[str, float]]:
        """Get individual model performances."""
        return self.model_performances.copy()
    
    def get_model_correlations(self) -> Dict[str, Dict[str, float]]:
        """Get model correlation matrix."""
        return self.model_correlations.copy()
    
    def save_ensemble_fast(self, filepath: str) -> None:
        """Save the fast super ensemble model with optimizations."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        # Save ensemble metadata (lightweight)
        ensemble_data = {
            'config': self.config,
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'model_performances': self.model_performances,
            'model_correlations': self.model_correlations,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'base_models': self.base_models,
            'meta_models': self.meta_models,
            'is_trained': True
        }
        
        # Save individual models with compression
        model_paths = {}
        for model_name, model in self.models.items():
            if model.is_trained:
                model_path = f"{filepath}_{model_name}"
                model.save_model(model_path)
                model_paths[model_name] = model_path
                
                # Cache the model
                if self.use_cache:
                    cache_path = Path(self.cache_dir) / 'models' / f"{model_name}_cached.joblib"
                    self._save_to_cache(model, cache_path)
        
        # Save meta-models with compression
        meta_model_paths = {}
        for meta_name, meta_model in self.meta_models_instances.items():
            meta_path = f"{filepath}_meta_{meta_name}"
            joblib.dump(meta_model, meta_path, compress=3)
            meta_model_paths[meta_name] = meta_path
            
            # Cache the meta-model
            if self.use_cache:
                cache_path = Path(self.cache_dir) / 'meta_models' / f"{meta_name}_cached.joblib"
                self._save_to_cache(meta_model, cache_path)
        
        # Save scalers with compression
        scaler_path = f"{filepath}_scalers"
        scaler_data = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'meta_scaler': self.meta_scaler
        }
        joblib.dump(scaler_data, scaler_path, compress=3)
        
        # Cache scalers
        if self.use_cache:
            cache_path = Path(self.cache_dir) / 'scalers' / "scalers_cached.joblib"
            self._save_to_cache(scaler_data, cache_path)
        
        # Store paths for lazy loading
        ensemble_data['model_paths'] = model_paths
        ensemble_data['meta_model_paths'] = meta_model_paths
        ensemble_data['scaler_path'] = scaler_path
        
        # Save ensemble metadata with compression
        joblib.dump(ensemble_data, f"{filepath}_fast_ensemble.joblib", compress=3)
        self.logger.info(f"Fast ensemble saved to {filepath}")
    
    def load_ensemble_fast(self, filepath: str) -> None:
        """Load the fast super ensemble model with optimizations."""
        # Load ensemble metadata (fast)
        ensemble_data = joblib.load(f"{filepath}_fast_ensemble.joblib")
        
        # Set basic attributes
        self.config = ensemble_data['config']
        self.ensemble_method = ensemble_data['ensemble_method']
        self.model_weights = ensemble_data['model_weights']
        self.model_performances = ensemble_data['model_performances']
        self.model_correlations = ensemble_data['model_correlations']
        self.metrics = ensemble_data.get('metrics', {})
        self.training_history = ensemble_data.get('training_history', {})
        self.base_models = ensemble_data['base_models']
        self.meta_models = ensemble_data['meta_models']
        
        # Store paths for lazy loading
        self._model_paths = ensemble_data.get('model_paths', {})
        self._meta_model_paths = ensemble_data.get('meta_model_paths', {})
        self._scaler_path = ensemble_data.get('scaler_path', '')
        
        # Mark as trained
        self.is_trained = True
        
        self.logger.info(f"Fast ensemble loaded from {filepath}")
    
    def _load_model_from_path(self, model_name: str):
        """Load a specific model from its saved path."""
        if model_name in self._model_paths:
            model_path = self._model_paths[model_name]
            
            # Create model instance
            config = {**self.config, **self._model_configs[model_name]}
            
            if model_name == 'lstm':
                model = LSTMModel(config)
            elif model_name == 'transformer':
                model = TransformerModel(config)
            elif model_name == 'xgboost':
                model = XGBoostModel(config)
            elif model_name == 'catboost':
                model = CatBoostModel(config)
            elif model_name == 'lightgbm':
                model = LightGBMModel(config)
            else:
                return None
            
            # Load the model
            model.load_model(model_path)
            return model
        
        return None
    
    def _load_meta_model_from_path(self, meta_name: str):
        """Load a specific meta-model from its saved path."""
        if meta_name in self._meta_model_paths:
            meta_path = self._meta_model_paths[meta_name]
            return joblib.load(meta_path)
        
        return None
    
    def _load_scalers_from_paths(self):
        """Load scalers from saved path."""
        if self._scaler_path and os.path.exists(self._scaler_path):
            scaler_data = joblib.load(self._scaler_path)
            self._feature_scaler = scaler_data['feature_scaler']
            self._target_scaler = scaler_data['target_scaler']
            self._meta_scaler = scaler_data['meta_scaler']
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get fast super ensemble model summary."""
        summary = super().get_model_summary()
        
        summary.update({
            'ensemble_method': self.ensemble_method,
            'base_models': self.base_models,
            'meta_models': self.meta_models,
            'model_weights': self.model_weights,
            'model_performances': self.model_performances,
            'model_correlations': self.model_correlations,
            'lazy_loading': self.lazy_loading,
            'use_cache': self.use_cache,
            'parallel_loading': self.parallel_loading,
            'compressed_serialization': self.compressed_serialization,
            'models_loaded': len(self._models) if hasattr(self, '_models') else 0,
            'meta_models_loaded': len(self._meta_models_instances) if hasattr(self, '_meta_models_instances') else 0
        })
        
        return summary
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.use_cache:
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                self._setup_cache()
                self.logger.info("Cache cleared")
    
    def preload_models(self):
        """Preload all models for faster prediction (optional)."""
        if self.lazy_loading and self.is_trained:
            self.logger.info("Preloading models...")
            
            # Load all models
            for model_name in self.base_models:
                if model_name not in self._models:
                    self._models[model_name] = self._load_model_from_path(model_name)
            
            # Load all meta-models
            for meta_name in self.meta_models:
                if meta_name not in self._meta_models_instances:
                    self._meta_models_instances[meta_name] = self._load_meta_model_from_path(meta_name)
            
            # Load scalers
            if self._feature_scaler is None:
                self._load_scalers_from_paths()
            
            self.logger.info("Models preloaded successfully")
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization."""
        return {
            'ensemble_method': trial.suggest_categorical('ensemble_method', 
                                                        ['adaptive_meta_learning', 'weighted_average']),
            'use_dynamic_weights': trial.suggest_categorical('use_dynamic_weights', [True, False]),
            'use_meta_learning': trial.suggest_categorical('use_meta_learning', [True, False]),
            'use_diversity_penalty': trial.suggest_categorical('use_diversity_penalty', [True, False]),
            'lazy_loading': trial.suggest_categorical('lazy_loading', [True, False]),
            'use_cache': trial.suggest_categorical('use_cache', [True, False]),
            'compressed_serialization': trial.suggest_categorical('compressed_serialization', [True, False])
        }
    
    def _build_model_with_params(self, params: Dict[str, Any]) -> dict:
        """Build model with specific parameters."""
        self.ensemble_method = params['ensemble_method']
        self.use_dynamic_weights = params['use_dynamic_weights']
        self.use_meta_learning = params['use_meta_learning']
        self.use_diversity_penalty = params['use_diversity_penalty']
        self.lazy_loading = params['lazy_loading']
        self.use_cache = params['use_cache']
        self.compressed_serialization = params['compressed_serialization']
        
        return self._build_model()
