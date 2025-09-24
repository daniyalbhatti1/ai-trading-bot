"""
Model factory for creating and managing ML models in AI Trading Bot.
"""

from typing import Dict, Any, Optional, List, Type
import logging
from enum import Enum

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .fast_super_ensemble_model import FastSuperEnsembleModel
from .lightgbm_model import LightGBMModel
from .feature_engineering import FeatureEngineer

class ModelType(str, Enum):
    """Supported model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    FAST_SUPER_ENSEMBLE = "fast_super_ensemble"

class ModelFactory:
    """Factory class for creating and managing ML models."""
    
    # Model registry
    MODEL_REGISTRY = {
        ModelType.XGBOOST: XGBoostModel,
        ModelType.LIGHTGBM: LightGBMModel,
        ModelType.FAST_SUPER_ENSEMBLE: FastSuperEnsembleModel
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.feature_engineer = None
        
    def create_model(self, model_type: str, model_config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create a model instance.
        
        Args:
            model_type: Type of model to create
            model_config: Model-specific configuration
            
        Returns:
            Model instance
        """
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(self.MODEL_REGISTRY.keys())}")
        
        # Merge base config with model-specific config
        if model_config is None:
            model_config = {}
        
        merged_config = {**self.config, **model_config}
        
        # Create model instance
        model_class = self.MODEL_REGISTRY[ModelType(model_type)]
        model = model_class(merged_config)
        
        self.logger.info(f"Created {model_type} model with config: {merged_config}")
        
        return model
    
    def create_feature_engineer(self, feature_config: Optional[Dict[str, Any]] = None) -> FeatureEngineer:
        """Create feature engineer instance.
        
        Args:
            feature_config: Feature engineering configuration
            
        Returns:
            FeatureEngineer instance
        """
        if feature_config is None:
            feature_config = {}
        
        merged_config = {**self.config, **feature_config}
        self.feature_engineer = FeatureEngineer(merged_config)
        
        self.logger.info(f"Created feature engineer with config: {merged_config}")
        
        return self.feature_engineer
    
    def create_ensemble_model(self, base_models: List[str], 
                            ensemble_config: Optional[Dict[str, Any]] = None) -> EnsembleModel:
        """Create an ensemble model with specified base models.
        
        Args:
            base_models: List of base model types
            ensemble_config: Ensemble-specific configuration
            
        Returns:
            EnsembleModel instance
        """
        if ensemble_config is None:
            ensemble_config = {}
        
        # Add base models to config
        ensemble_config['base_models'] = base_models
        
        merged_config = {**self.config, **ensemble_config}
        ensemble_model = EnsembleModel(merged_config)
        
        self.logger.info(f"Created ensemble model with base models: {base_models}")
        
        return ensemble_model
    
    def create_super_ensemble_model(self, base_models: List[str], 
                                   meta_models: List[str] = None,
                                   super_ensemble_config: Optional[Dict[str, Any]] = None) -> SuperEnsembleModel:
        """Create a super ensemble model with all advanced features.
        
        Args:
            base_models: List of base model types
            meta_models: List of meta-model types (optional)
            super_ensemble_config: Super ensemble-specific configuration
            
        Returns:
            SuperEnsembleModel instance
        """
        if super_ensemble_config is None:
            super_ensemble_config = {}
        
        if meta_models is None:
            meta_models = ['ridge', 'elastic_net', 'mlp', 'svr']
        
        # Add base models and meta models to config
        super_ensemble_config['base_models'] = base_models
        super_ensemble_config['meta_models'] = meta_models
        
        merged_config = {**self.config, **super_ensemble_config}
        super_ensemble_model = SuperEnsembleModel(merged_config)
        
        self.logger.info(f"Created super ensemble model with base models: {base_models} and meta models: {meta_models}")
        
        return super_ensemble_model
    
    def create_fast_super_ensemble_model(self, base_models: List[str], 
                                        meta_models: List[str] = None,
                                        fast_ensemble_config: Optional[Dict[str, Any]] = None) -> FastSuperEnsembleModel:
        """Create a fast super ensemble model with optimized loading.
        
        Args:
            base_models: List of base model types
            meta_models: List of meta-model types (optional)
            fast_ensemble_config: Fast ensemble-specific configuration
            
        Returns:
            FastSuperEnsembleModel instance
        """
        if fast_ensemble_config is None:
            fast_ensemble_config = {}
        
        if meta_models is None:
            meta_models = ['ridge', 'elastic_net', 'mlp', 'svr']
        
        # Add base models and meta models to config
        fast_ensemble_config['base_models'] = base_models
        fast_ensemble_config['meta_models'] = meta_models
        
        # Set optimization defaults
        fast_ensemble_config.setdefault('lazy_loading', True)
        fast_ensemble_config.setdefault('use_cache', True)
        fast_ensemble_config.setdefault('parallel_loading', True)
        fast_ensemble_config.setdefault('compressed_serialization', True)
        fast_ensemble_config.setdefault('cache_dir', './model_cache')
        
        merged_config = {**self.config, **fast_ensemble_config}
        fast_ensemble_model = FastSuperEnsembleModel(merged_config)
        
        self.logger.info(f"Created fast super ensemble model with base models: {base_models} and meta models: {meta_models}")
        
        return fast_ensemble_model
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types.
        
        Returns:
            List of model type names
        """
        return list(self.MODEL_REGISTRY.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a model type.
        
        Args:
            model_type: Model type name
            
        Returns:
            Dictionary with model information
        """
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = self.MODEL_REGISTRY[ModelType(model_type)]
        
        return {
            'name': model_type,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'description': model_class.__doc__ or "No description available"
        }
    
    def create_model_from_config(self, model_config: Dict[str, Any]) -> BaseModel:
        """Create model from configuration dictionary.
        
        Args:
            model_config: Model configuration with 'type' key
            
        Returns:
            Model instance
        """
        if 'type' not in model_config:
            raise ValueError("Model configuration must contain 'type' key")
        
        model_type = model_config['type']
        config = {k: v for k, v in model_config.items() if k != 'type'}
        
        return self.create_model(model_type, config)
    
    def create_multiple_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, BaseModel]:
        """Create multiple models from configuration list.
        
        Args:
            model_configs: List of model configurations
            
        Returns:
            Dictionary mapping model names to model instances
        """
        models = {}
        
        for config in model_configs:
            if 'name' not in config:
                raise ValueError("Model configuration must contain 'name' key")
            
            model_name = config['name']
            model_config = {k: v for k, v in config.items() if k != 'name'}
            
            models[model_name] = self.create_model_from_config(model_config)
        
        return models
    
    def get_recommended_models(self, task_type: str = 'regression') -> List[str]:
        """Get recommended models for a specific task type.
        
        Args:
            task_type: Type of task ('regression', 'classification', 'time_series')
            
        Returns:
            List of recommended model types
        """
        if task_type == 'regression':
            return [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM, ModelType.ENSEMBLE, ModelType.SUPER_ENSEMBLE, ModelType.FAST_SUPER_ENSEMBLE]
        elif task_type == 'time_series':
            return [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.ENSEMBLE, ModelType.SUPER_ENSEMBLE, ModelType.FAST_SUPER_ENSEMBLE]
        elif task_type == 'classification':
            return [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM]
        else:
            return list(self.MODEL_REGISTRY.keys())
    
    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison of all available models.
        
        Returns:
            Dictionary with model comparisons
        """
        comparison = {}
        
        for model_type in self.MODEL_REGISTRY.keys():
            model_info = self.get_model_info(model_type)
            
            # Add model characteristics
            characteristics = {
                'type': 'neural_network' if model_type in [ModelType.LSTM, ModelType.TRANSFORMER] else 'tree_based' if model_type in [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM] else 'ensemble',
                'ensemble_capable': model_type not in [ModelType.ENSEMBLE, ModelType.SUPER_ENSEMBLE, ModelType.FAST_SUPER_ENSEMBLE],
                'interpretable': model_type in [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM],
                'handles_missing_values': model_type in [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM],
                'scalable': model_type in [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.LIGHTGBM],
                'time_series_optimized': model_type in [ModelType.LSTM, ModelType.TRANSFORMER],
                'advanced_ensemble': model_type in [ModelType.SUPER_ENSEMBLE, ModelType.FAST_SUPER_ENSEMBLE],
                'meta_learning': model_type in [ModelType.SUPER_ENSEMBLE, ModelType.FAST_SUPER_ENSEMBLE],
                'fast_loading': model_type == ModelType.FAST_SUPER_ENSEMBLE,
                'lazy_loading': model_type == ModelType.FAST_SUPER_ENSEMBLE,
                'caching': model_type == ModelType.FAST_SUPER_ENSEMBLE
            }
            
            comparison[model_type] = {
                **model_info,
                'characteristics': characteristics
            }
        
        return comparison
    
    def validate_model_config(self, model_config: Dict[str, Any]) -> List[str]:
        """Validate model configuration.
        
        Args:
            model_config: Model configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required keys
        if 'type' not in model_config:
            errors.append("Model configuration must contain 'type' key")
        
        # Validate model type
        if 'type' in model_config:
            model_type = model_config['type']
            if model_type not in self.MODEL_REGISTRY:
                errors.append(f"Unsupported model type: {model_type}")
        
        # Model-specific validation
        if 'type' in model_config and model_config['type'] in self.MODEL_REGISTRY:
            model_class = self.MODEL_REGISTRY[ModelType(model_config['type'])]
            
            # Check for required parameters
            if hasattr(model_class, 'REQUIRED_PARAMS'):
                for param in model_class.REQUIRED_PARAMS:
                    if param not in model_config:
                        errors.append(f"Required parameter '{param}' missing for {model_config['type']} model")
        
        return errors
    
    def get_default_config(self, model_type: str) -> Dict[str, Any]:
        """Get default configuration for a model type.
        
        Args:
            model_type: Model type name
            
        Returns:
            Default configuration dictionary
        """
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Default configurations for each model type
        default_configs = {
            ModelType.LSTM: {
                'lstm_units': [64, 32],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'patience': 15
            },
            ModelType.TRANSFORMER: {
                'd_model': 64,
                'num_heads': 8,
                'num_layers': 4,
                'dff': 128,
                'dropout_rate': 0.1,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'patience': 15
            },
            ModelType.XGBOOST: {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 50
            },
            ModelType.CATBOOST: {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'early_stopping_rounds': 50
            },
            ModelType.LIGHTGBM: {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 50
            },
            ModelType.ENSEMBLE: {
                'ensemble_method': 'weighted_average',
                'base_models': ['lstm', 'xgboost', 'catboost'],
                'optimize_weights': True,
                'weight_optimization_method': 'minimize_mse'
            },
            ModelType.SUPER_ENSEMBLE: {
                'ensemble_method': 'adaptive_meta_learning',
                'base_models': ['lstm', 'transformer', 'xgboost', 'catboost', 'lightgbm'],
                'meta_models': ['ridge', 'elastic_net', 'mlp', 'svr'],
                'use_dynamic_weights': True,
                'use_meta_learning': True,
                'use_model_selection': True,
                'use_uncertainty_weighting': True,
                'use_diversity_penalty': True,
                'adaptive_window': 100
            },
            ModelType.FAST_SUPER_ENSEMBLE: {
                'ensemble_method': 'adaptive_meta_learning',
                'base_models': ['lstm', 'transformer', 'xgboost', 'catboost', 'lightgbm'],
                'meta_models': ['ridge', 'elastic_net', 'mlp', 'svr'],
                'use_dynamic_weights': True,
                'use_meta_learning': True,
                'use_model_selection': True,
                'use_uncertainty_weighting': True,
                'use_diversity_penalty': True,
                'adaptive_window': 100,
                'lazy_loading': True,
                'use_cache': True,
                'parallel_loading': True,
                'compressed_serialization': True,
                'cache_dir': './model_cache'
            }
        }
        
        return default_configs.get(ModelType(model_type), {})
    
    def create_model_pipeline(self, model_configs: List[Dict[str, Any]], 
                            feature_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a complete model pipeline with feature engineering.
        
        Args:
            model_configs: List of model configurations
            feature_config: Feature engineering configuration
            
        Returns:
            Dictionary containing feature engineer and models
        """
        pipeline = {}
        
        # Create feature engineer
        pipeline['feature_engineer'] = self.create_feature_engineer(feature_config)
        
        # Create models
        pipeline['models'] = {}
        for config in model_configs:
            model_name = config.get('name', config['type'])
            pipeline['models'][model_name] = self.create_model_from_config(config)
        
        self.logger.info(f"Created model pipeline with {len(pipeline['models'])} models")
        
        return pipeline
    
    def save_model_registry(self, filepath: str) -> None:
        """Save model registry to file.
        
        Args:
            filepath: Path to save the registry
        """
        import json
        
        registry_data = {
            'available_models': list(self.MODEL_REGISTRY.keys()),
            'model_info': {model_type: self.get_model_info(model_type) 
                          for model_type in self.MODEL_REGISTRY.keys()},
            'default_configs': {model_type: self.get_default_config(model_type) 
                              for model_type in self.MODEL_REGISTRY.keys()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self.logger.info(f"Model registry saved to {filepath}")
    
    def load_model_registry(self, filepath: str) -> None:
        """Load model registry from file.
        
        Args:
            filepath: Path to load the registry from
        """
        import json
        
        with open(filepath, 'r') as f:
            registry_data = json.load(f)
        
        self.logger.info(f"Model registry loaded from {filepath}")
        
        return registry_data
