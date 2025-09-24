"""
Advanced feature engineering module for AI Trading Bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import talib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Advanced feature engineering for financial time series data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Feature engineering options
        self.use_technical_indicators = config.get('use_technical_indicators', True)
        self.use_price_features = config.get('use_price_features', True)
        self.use_volume_features = config.get('use_volume_features', True)
        self.use_volatility_features = config.get('use_volatility_features', True)
        self.use_momentum_features = config.get('use_momentum_features', True)
        self.use_cycle_features = config.get('use_cycle_features', True)
        self.use_statistical_features = config.get('use_statistical_features', True)
        self.use_fourier_features = config.get('use_fourier_features', True)
        self.use_wavelet_features = config.get('use_wavelet_features', False)
        
        # Feature selection
        self.feature_selection = config.get('feature_selection', True)
        self.n_features_select = config.get('n_features_select', 50)
        self.feature_selection_method = config.get('feature_selection_method', 'mutual_info')
        
        # Dimensionality reduction
        self.use_pca = config.get('use_pca', False)
        self.pca_components = config.get('pca_components', 20)
        self.use_tsne = config.get('use_tsne', False)
        self.tsne_components = config.get('tsne_components', 2)
        
        # Scaling
        self.scaling_method = config.get('scaling_method', 'robust')  # standard, robust, minmax
        self.scaler = None
        
        # Feature names tracking
        self.feature_names = []
        self.selected_features = []
        
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        df = data.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Price positions
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'])
        
        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_ratio'] = df['gap'] / df['close'].shift(1)
        
        # Intraday features
        df['intraday_range'] = df['high'] - df['low']
        df['intraday_range_ratio'] = df['intraday_range'] / df['close']
        
        return df
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        df = data.copy()
        
        # Volume ratios
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        
        # Volume-price features
        df['price_volume'] = df['close'] * df['volume']
        df['price_volume_sma'] = df['price_volume'].rolling(window=20).mean()
        df['price_volume_ratio'] = df['price_volume'] / df['price_volume_sma']
        
        # Volume-weighted average price (VWAP)
        df['vwap'] = (df['price_volume'].rolling(window=20).sum() / 
                     df['volume'].rolling(window=20).sum())
        df['close_vwap_ratio'] = df['close'] / df['vwap']
        
        # On-Balance Volume (OBV)
        df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1,
                                           np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
        
        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['price_change']).cumsum()
        
        # Accumulation/Distribution Line
        df['ad_line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['ad_line'] = df['ad_line'].cumsum()
        
        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return df
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        df = data.copy()
        
        # Historical volatility
        returns = df['close'].pct_change()
        df['volatility_5'] = returns.rolling(window=5).std() * np.sqrt(252)
        df['volatility_10'] = returns.rolling(window=10).std() * np.sqrt(252)
        df['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(window=20).mean() * 0.5).astype(int)
        
        # Keltner Channels
        df['kc_middle'] = df['close'].rolling(window=20).mean()
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * 2)
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * 2)
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # Donchian Channels
        df['dc_upper'] = df['high'].rolling(window=20).max()
        df['dc_lower'] = df['low'].rolling(window=20).min()
        df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
        df['dc_width'] = (df['dc_upper'] - df['dc_lower']) / df['dc_middle']
        
        return df
    
    def create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features."""
        df = data.copy()
        
        # Rate of Change (ROC)
        df['roc_5'] = df['close'].pct_change(5) * 100
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        return df
    
    def create_cycle_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cycle-based features."""
        df = data.copy()
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # Moving average crossovers
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Price relative to moving averages
        df['close_sma_5_ratio'] = df['close'] / df['sma_5']
        df['close_sma_20_ratio'] = df['close'] / df['sma_20']
        df['close_sma_50_ratio'] = df['close'] / df['sma_50']
        
        # Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        
        return df
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        df = data.copy()
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_skew_{window}'] = df['close'].rolling(window=window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window=window).kurt()
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            df[f'close_median_{window}'] = df['close'].rolling(window=window).median()
            df[f'close_quantile_25_{window}'] = df['close'].rolling(window=window).quantile(0.25)
            df[f'close_quantile_75_{window}'] = df['close'].rolling(window=window).quantile(0.75)
            
            # Z-score
            df[f'close_zscore_{window}'] = (df['close'] - df[f'close_mean_{window}']) / df[f'close_std_{window}']
            
            # Percentile rank
            df[f'close_percentile_{window}'] = df['close'].rolling(window=window).rank(pct=True)
        
        # Autocorrelation
        for lag in [1, 2, 3, 5, 10]:
            df[f'autocorr_{lag}'] = df['close'].rolling(window=20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        return df
    
    def create_fourier_features(self, data: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Create Fourier transform features."""
        df = data.copy()
        
        # Apply FFT to price data
        price_data = df['close'].values
        fft = np.fft.fft(price_data)
        freqs = np.fft.fftfreq(len(price_data))
        
        # Get top frequency components
        magnitude = np.abs(fft)
        top_indices = np.argsort(magnitude)[-n_components:]
        
        for i, idx in enumerate(top_indices):
            if idx > 0:  # Skip DC component
                df[f'fourier_real_{i}'] = np.real(fft[idx])
                df[f'fourier_imag_{i}'] = np.imag(fft[idx])
                df[f'fourier_magnitude_{i}'] = magnitude[idx]
                df[f'fourier_freq_{i}'] = freqs[idx]
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create lag features."""
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20, 50]
        
        df = data.copy()
        
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'high_lag_{lag}'] = df['high'].shift(lag)
            df[f'low_lag_{lag}'] = df['low'].shift(lag)
            df[f'open_lag_{lag}'] = df['open'].shift(lag)
        
        return df
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different indicators."""
        df = data.copy()
        
        # Price-volume interactions
        if 'rsi' in df.columns and 'volume_ratio_20' in df.columns:
            df['rsi_volume_interaction'] = df['rsi'] * df['volume_ratio_20']
        
        if 'macd' in df.columns and 'bb_position' in df.columns:
            df['macd_bb_interaction'] = df['macd'] * df['bb_position']
        
        # Volatility-momentum interactions
        if 'volatility_20' in df.columns and 'rsi' in df.columns:
            df['volatility_momentum_interaction'] = df['volatility_20'] * df['rsi']
        
        # Multi-timeframe interactions
        if 'sma_5' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['sma_trend_strength'] = (df['sma_5'] - df['sma_20']) / (df['sma_20'] - df['sma_50'])
        
        return df
    
    def select_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Select best features using statistical methods."""
        if not self.feature_selection:
            return X, list(range(X.shape[1]))
        
        if self.feature_selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=self.n_features_select)
        else:  # f_regression
            selector = SelectKBest(score_func=f_regression, k=self.n_features_select)
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        self.selected_features = selected_indices.tolist()
        
        return X_selected, selected_indices
    
    def apply_dimensionality_reduction(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction techniques."""
        if self.use_pca:
            pca = PCA(n_components=self.pca_components)
            X = pca.fit_transform(X)
            self.logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        if self.use_tsne:
            tsne = TSNE(n_components=self.tsne_components, random_state=42)
            X_tsne = tsne.fit_transform(X)
            # Add t-SNE features to existing features
            X = np.column_stack([X, X_tsne])
        
        return X
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale features using specified method."""
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            scaler = RobustScaler()
        else:  # minmax
            scaler = MinMaxScaler()
        
        if fit:
            X_scaled = scaler.fit_transform(X)
            self.scaler = scaler
        else:
            if self.scaler is None:
                raise ValueError("Scaler must be fitted first")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def engineer_features(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Main feature engineering pipeline."""
        self.logger.info("Starting feature engineering...")
        
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Create features
        df = data.copy()
        
        if self.use_price_features:
            df = self.create_price_features(df)
        
        if self.use_volume_features:
            df = self.create_volume_features(df)
        
        if self.use_volatility_features:
            df = self.create_volatility_features(df)
        
        if self.use_momentum_features:
            df = self.create_momentum_features(df)
        
        if self.use_cycle_features:
            df = self.create_cycle_features(df)
        
        if self.use_statistical_features:
            df = self.create_statistical_features(df)
        
        if self.use_fourier_features:
            df = self.create_fourier_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        self.feature_names = feature_columns
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        self.logger.info(f"Created {len(feature_columns)} features")
        
        # Scale features
        X = self.scale_features(X, fit=True)
        
        # Select features
        X, selected_indices = self.select_features(X, y)
        
        # Apply dimensionality reduction
        X = self.apply_dimensionality_reduction(X)
        
        # Update feature names for selected features
        if self.selected_features:
            selected_feature_names = [self.feature_names[i] for i in self.selected_features]
        else:
            selected_feature_names = self.feature_names
        
        self.logger.info(f"Final feature count: {X.shape[1]}")
        
        return X, y, selected_feature_names
    
    def transform_features(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted feature engineering pipeline."""
        if self.scaler is None:
            raise ValueError("Feature engineer must be fitted first")
        
        # Apply same feature engineering steps
        df = data.copy()
        
        if self.use_price_features:
            df = self.create_price_features(df)
        
        if self.use_volume_features:
            df = self.create_volume_features(df)
        
        if self.use_volatility_features:
            df = self.create_volatility_features(df)
        
        if self.use_momentum_features:
            df = self.create_momentum_features(df)
        
        if self.use_cycle_features:
            df = self.create_cycle_features(df)
        
        if self.use_statistical_features:
            df = self.create_statistical_features(df)
        
        if self.use_fourier_features:
            df = self.create_fourier_features(df)
        
        df = self.create_lag_features(df)
        df = self.create_interaction_features(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Get features
        feature_columns = [col for col in df.columns if col in self.feature_names]
        X = df[feature_columns].values
        
        # Scale features
        X = self.scale_features(X, fit=False)
        
        # Select features
        if self.selected_features:
            X = X[:, self.selected_features]
        
        # Apply dimensionality reduction
        X = self.apply_dimensionality_reduction(X)
        
        return X
    
    def get_feature_importance_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_selection_method == 'mutual_info':
            scores = mutual_info_regression(X, y)
        else:
            scores, _ = f_regression(X, y)
        
        feature_scores = {}
        for i, score in enumerate(scores):
            if i < len(self.feature_names):
                feature_scores[self.feature_names[i]] = score
        
        return feature_scores
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get feature engineering summary."""
        return {
            'total_features_created': len(self.feature_names),
            'selected_features': len(self.selected_features),
            'scaling_method': self.scaling_method,
            'feature_selection_method': self.feature_selection_method,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components if self.use_pca else None,
            'feature_categories': {
                'price_features': self.use_price_features,
                'volume_features': self.use_volume_features,
                'volatility_features': self.use_volatility_features,
                'momentum_features': self.use_momentum_features,
                'cycle_features': self.use_cycle_features,
                'statistical_features': self.use_statistical_features,
                'fourier_features': self.use_fourier_features
            }
        }
