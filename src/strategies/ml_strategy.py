"""Machine Learning-based strategy using Random Forest"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .base_strategy import BaseStrategy, Signal
from ..utils.indicators import TechnicalIndicators
import pickle
from pathlib import Path


class MLStrategy(BaseStrategy):
    """
    Machine Learning strategy using Random Forest Classifier.
    Predicts BUY/SELL signals based on technical indicators.
    """

    def __init__(self, params: dict = None):
        """
        Initialize ML strategy.

        Args:
            params: Strategy parameters
                - lookback_period: Periods to look back for features (default: 20)
                - prediction_threshold: Confidence threshold (default: 0.6)
                - model_path: Path to saved model (optional)
        """
        default_params = {
            'lookback_period': 20,
            'prediction_threshold': 0.6,
            'model_path': None,
        }
        if params:
            default_params.update(params)

        super().__init__('MLStrategy', default_params)

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Load existing model if path provided
        if self.params['model_path']:
            self.load_model(self.params['model_path'])

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for ML features"""
        df = TechnicalIndicators.add_all_indicators(df)
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model.

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with feature columns
        """
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'adx', 'stoch_k', 'stoch_d',
            'bb_width', 'atr', 'volatility',
            'trend_strength', 'returns'
        ]

        # Add lagged features
        for col in ['close', 'volume', 'rsi']:
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                feature_columns.append(f'{col}_lag_{lag}')

        # Rolling statistics
        df['close_rolling_mean'] = df['close'].rolling(window=10).mean()
        df['close_rolling_std'] = df['close'].rolling(window=10).std()
        feature_columns.extend(['close_rolling_mean', 'close_rolling_std'])

        # Select and clean features
        features = df[feature_columns].copy()
        features = features.fillna(method='bfill').fillna(0)

        return features

    def create_labels(self, df: pd.DataFrame, future_periods: int = 5) -> pd.Series:
        """
        Create labels for training.
        Label = 1 (BUY) if price increases in next N periods
        Label = -1 (SELL) if price decreases in next N periods
        Label = 0 (HOLD) otherwise

        Args:
            df: DataFrame with price data
            future_periods: Periods to look ahead

        Returns:
            Series with labels
        """
        future_returns = df['close'].shift(-future_periods) / df['close'] - 1

        labels = pd.Series(0, index=df.index)
        labels[future_returns > 0.02] = 1   # BUY if >2% gain
        labels[future_returns < -0.02] = -1  # SELL if >2% loss

        return labels

    def train(self, df: pd.DataFrame, save_path: str = None):
        """
        Train the ML model.

        Args:
            df: Training data with OHLCV
            save_path: Path to save trained model
        """
        self.logger.info("Training ML model...")

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Prepare features and labels
        features = self.prepare_features(df)
        labels = self.create_labels(df)

        # Remove NaN rows
        mask = ~(features.isna().any(axis=1) | labels.isna())
        X = features[mask]
        y = labels[mask]

        # Remove last rows where we don't have future data
        X = X[:-10]
        y = y[:-10]

        self.logger.info(f"Training set size: {len(X)}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate training accuracy
        train_accuracy = self.model.score(X_scaled, y)
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.logger.info(f"Top 5 features:\n{feature_importance.head()}")

        # Save model
        if save_path:
            self.save_model(save_path)

    def save_model(self, path: str):
        """Save trained model and scaler"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'params': self.params
            }, f)

        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model and scaler"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True

            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate signal using trained ML model.

        Args:
            df: DataFrame with market data

        Returns:
            Signal (BUY, SELL, or HOLD)
        """
        if not self.validate_data(df):
            return Signal.HOLD

        if not self.is_trained:
            self.logger.warning("Model not trained, training now...")
            self.train(df)

        if len(df) < 100:
            self.logger.warning("Insufficient data for ML prediction")
            return Signal.HOLD

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Prepare features for latest data point
        features = self.prepare_features(df)
        X = features.iloc[-1:].fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Get confidence (max probability)
        confidence = max(probabilities)

        # Only act if confidence is above threshold
        if confidence < self.params['prediction_threshold']:
            return Signal.HOLD

        if prediction == 1:
            self.logger.info(f"BUY signal: ML prediction with {confidence:.2%} confidence")
            return Signal.BUY
        elif prediction == -1:
            self.logger.info(f"SELL signal: ML prediction with {confidence:.2%} confidence")
            return Signal.SELL
        else:
            return Signal.HOLD

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on prediction confidence"""
        if not self.is_trained or len(df) < 100:
            return 0.0

        df = self.calculate_indicators(df)
        features = self.prepare_features(df)
        X = features.iloc[-1:].fillna(0)
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)

        return confidence
