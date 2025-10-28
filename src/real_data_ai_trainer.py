import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

try:
    from data_collector import HistoricalDataCollector
except ImportError:
    from src.data_collector import HistoricalDataCollector

class RealDataAITrainer:
    """
    AI model training system using real market data
    Trains advanced models for price prediction and trading signals
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.data_collector = HistoricalDataCollector()

        # Model parameters
        self.sequence_length = 24  # 24 hours of data for prediction
        self.prediction_horizon = 4  # Predict 4 hours ahead
        self.feature_columns = None
        self.target_columns = ['direction_4h', 'future_return_4h']

        # Scalers and models
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.models = {}

        # Training history
        self.training_history = {}

        # Models directory
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

        print("🤖 Real data AI trainer initialized")

    def _setup_logging(self):
        """Setup training logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'ai_training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AITrainer')

    def prepare_training_data(self, lookback_days: int = 365) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from historical market data

        Args:
            lookback_days: Number of days of historical data to use

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            self.logger.info(f"Preparing training data for {self.symbol}")

            # Get processed data
            data = self.data_collector.collect_comprehensive_dataset(self.symbol, lookback_days)

            if data is None or data.empty:
                self.logger.error("No data available for training")
                return None, None, None, None

            # Select features for training
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
                'stoch_k', 'stoch_d', 'atr_14',
                'volume_sma_20', 'volume_ratio',
                'price_range', 'hour', 'day_of_week', 'month'
            ]

            # Filter available columns
            available_features = [col for col in feature_cols if col in data.columns]
            self.feature_columns = available_features

            self.logger.info(f"Using {len(available_features)} features: {available_features}")

            # Prepare features and targets
            X = data[available_features].values
            y_direction = data['direction_4h'].values
            y_return = data['future_return_4h'].values

            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            y_return = np.nan_to_num(y_return, nan=0.0)

            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)

            # Create sequences for time series prediction
            X_sequences = []
            y_sequences_direction = []
            y_sequences_return = []

            for i in range(len(X_scaled) - self.sequence_length - self.prediction_horizon):
                X_sequences.append(X_scaled[i:i+self.sequence_length])
                y_sequences_direction.append(y_direction[i+self.sequence_length+self.prediction_horizon-1])
                y_sequences_return.append(y_return[i+self.sequence_length+self.prediction_horizon-1])

            X_sequences = np.array(X_sequences)
            y_direction = np.array(y_sequences_direction)
            y_return = np.array(y_sequences_return)

            # Split into train/test
            split_idx = int(len(X_sequences) * 0.8)

            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train_direction = y_direction[:split_idx]
            y_test_direction = y_direction[split_idx:]
            y_train_return = y_return[:split_idx]
            y_test_return = y_return[split_idx:]

            self.logger.info(f"Training data shape: X={X_train.shape}, y_direction={y_train_direction.shape}, y_return={y_train_return.shape}")
            self.logger.info(f"Testing data shape: X={X_test.shape}, y_direction={y_test_direction.shape}, y_return={y_test_return.shape}")

            return X_train, X_test, {
                'direction': (y_train_direction, y_test_direction),
                'return': (y_train_return, y_test_return)
            }

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None, None, None, None

    def build_lstm_model(self, input_shape: Tuple[int, int], output_type: str = 'direction') -> keras.Model:
        """
        Build LSTM model for time series prediction

        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_type: 'direction' for classification or 'return' for regression

        Returns:
            Compiled Keras model
        """
        try:
            model = keras.Sequential([
                keras.layers.Input(shape=input_shape),

                # LSTM layers
                keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                # Dense layers
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.3),

                # Output layer
                keras.layers.Dense(1, activation='sigmoid' if output_type == 'direction' else 'linear')
            ])

            # Compile model
            if output_type == 'direction':
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', keras.metrics.AUC()]
                )
            else:
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae', 'mse']
                )

            self.logger.info(f"Built {output_type} LSTM model with input shape {input_shape}")
            return model

        except Exception as e:
            self.logger.error(f"Error building LSTM model: {e}")
            return None

    def build_gru_model(self, input_shape: Tuple[int, int], output_type: str = 'direction') -> keras.Model:
        """
        Build GRU model for time series prediction

        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_type: 'direction' for classification or 'return' for regression

        Returns:
            Compiled Keras model
        """
        try:
            model = keras.Sequential([
                keras.layers.Input(shape=input_shape),

                # GRU layers
                keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                keras.layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                # Dense layers
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.3),

                # Output layer
                keras.layers.Dense(1, activation='sigmoid' if output_type == 'direction' else 'linear')
            ])

            # Compile model
            if output_type == 'direction':
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', keras.metrics.AUC()]
                )
            else:
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae', 'mse']
                )

            self.logger.info(f"Built {output_type} GRU model with input shape {input_shape}")
            return model

        except Exception as e:
            self.logger.error(f"Error building GRU model: {e}")
            return None

    def build_cnn_lstm_model(self, input_shape: Tuple[int, int], output_type: str = 'direction') -> keras.Model:
        """
        Build CNN-LSTM hybrid model for time series prediction

        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_type: 'direction' for classification or 'return' for regression

        Returns:
            Compiled Keras model
        """
        try:
            model = keras.Sequential([
                keras.layers.Input(shape=input_shape),

                # CNN layers for feature extraction
                keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling1D(pool_size=2),

                keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling1D(pool_size=2),

                # LSTM layers
                keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
                keras.layers.BatchNormalization(),

                # Dense layers
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.3),

                # Output layer
                keras.layers.Dense(1, activation='sigmoid' if output_type == 'direction' else 'linear')
            ])

            # Compile model
            if output_type == 'direction':
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', keras.metrics.AUC()]
                )
            else:
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae', 'mse']
                )

            self.logger.info(f"Built {output_type} CNN-LSTM model with input shape {input_shape}")
            return model

        except Exception as e:
            self.logger.error(f"Error building CNN-LSTM model: {e}")
            return None

    def train_ensemble_models(self, epochs: int = 50, batch_size: int = 32, lookback_days: int = 365) -> Dict:
        """
        Train ensemble of AI models for trading predictions

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary with training results and model performance
        """
        try:
            self.logger.info("Starting ensemble model training")

            # Prepare data
            X_train, X_test, y_data = self.prepare_training_data(lookback_days)

            if X_train is None:
                return {'error': 'Failed to prepare training data'}

            # Train direction prediction models
            direction_models = {}
            direction_results = {}

            model_configs = [
                ('lstm_direction', self.build_lstm_model, 'direction'),
                ('gru_direction', self.build_gru_model, 'direction'),
                ('cnn_lstm_direction', self.build_cnn_lstm_model, 'direction')
            ]

            for model_name, model_builder, output_type in model_configs:
                self.logger.info(f"Training {model_name} model")

                # Build model
                model = model_builder(X_train.shape[1:], output_type)
                if model is None:
                    continue

                # Get target data
                y_train, y_test = y_data[output_type]

                # Add callbacks
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                ]

                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )

                # Evaluate model
                test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)

                # Predictions for detailed metrics
                y_pred = model.predict(X_test)
                if output_type == 'direction':
                    y_pred_classes = (y_pred > 0.5).astype(int)
                    report = classification_report(y_test, y_pred_classes, output_dict=True)
                else:
                    report = {'mae': np.mean(np.abs(y_test - y_pred.flatten()))}

                # Store results
                direction_models[model_name] = model
                direction_results[model_name] = {
                    'history': history.history,
                    'test_metrics': {
                        'loss': test_loss,
                        'accuracy': test_acc,
                        'auc': test_auc
                    },
                    'classification_report': report,
                    'predictions': y_pred.flatten()[:100]  # Store first 100 predictions
                }

                # Save model
                model_path = f"data/models/{model_name}_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                model.save(model_path)
                self.logger.info(f"Saved {model_name} model to {model_path}")

            # Store all models and results
            self.models = direction_models
            self.training_history = direction_results

            # Save training results
            results_path = f"data/models/training_results_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for model_name, result in direction_results.items():
                    serializable_results[model_name] = {
                        'test_metrics': result['test_metrics'],
                        'classification_report': result['classification_report'],
                        'predictions_sample': result['predictions'][:10].tolist()
                    }
                json.dump(serializable_results, f, indent=4)

            self.logger.info(f"Saved training results to {results_path}")

            return {
                'success': True,
                'models_trained': len(direction_models),
                'results': direction_results,
                'best_model': max(direction_results.keys(),
                                key=lambda x: direction_results[x]['test_metrics']['accuracy'])
            }

        except Exception as e:
            self.logger.error(f"Error training ensemble models: {e}")
            return {'error': str(e)}

    def predict_trading_signal(self, current_data: pd.DataFrame) -> Dict:
        """
        Generate trading signal using ensemble of trained models

        Args:
            current_data: Current market data (last sequence_length hours)

        Returns:
            Dictionary with prediction results
        """
        try:
            # Load models if not already loaded
            if not self.models:
                self.load_trained_models()

            # Set default feature columns if not set
            if self.feature_columns is None:
                self.feature_columns = [
                    'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns',
                    'sma_5', 'sma_10', 'sma_20', 'sma_50',
                    'ema_5', 'ema_10', 'ema_20', 'ema_50',
                    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
                    'stoch_k', 'stoch_d', 'atr_14',
                    'volume_sma_20', 'volume_ratio',
                    'price_range', 'hour', 'day_of_week', 'month'
                ]

            if not self.models:
                return {'error': 'No trained models available'}

            # Prepare input data
            if self.feature_columns is None:
                return {'error': 'Feature columns not defined'}

            # Get latest data
            available_features = [col for col in self.feature_columns if col in current_data.columns]
            X = current_data[available_features].tail(self.sequence_length).values

            if len(X) < self.sequence_length:
                return {'error': f'Insufficient data: {len(X)} < {self.sequence_length}'}

            # Scale features if scaler is available and fitted
            try:
                if self.feature_scaler is not None:
                    X_scaled = self.feature_scaler.transform(X)
                else:
                    # If no scaler, use data as-is (not ideal but allows testing)
                    X_scaled = X
            except Exception as e:
                # If scaler is not fitted or has issues, use data as-is
                print(f"Warning: Feature scaling failed ({e}), using raw data")
                X_scaled = X
            X_input = X_scaled.reshape(1, self.sequence_length, -1)

            # Get predictions from all models
            predictions = {}
            confidence_scores = []

            for model_name, model in self.models.items():
                pred = model.predict(X_input, verbose=0)[0][0]
                predictions[model_name] = pred
                confidence_scores.append(pred)

            # Ensemble prediction
            ensemble_pred = np.mean(confidence_scores)
            ensemble_confidence = np.std(confidence_scores)  # Lower std = higher agreement

            # Determine signal
            if ensemble_pred > 0.55:  # Lowered from 0.6 to 0.55
                signal = 'BUY'
                strength = 'Strong' if ensemble_pred > 0.7 else 'Moderate'
            elif ensemble_pred < 0.45:  # Adjusted symmetrically
                signal = 'SELL'
                strength = 'Strong' if ensemble_pred < 0.3 else 'Moderate'
            else:
                signal = 'HOLD'
                strength = 'Neutral'

            return {
                'signal': signal,
                'strength': strength,
                'confidence': ensemble_pred,
                'agreement': 1.0 - ensemble_confidence,  # Higher agreement = lower std
                'individual_predictions': predictions,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return {'error': str(e)}

    def load_trained_models(self, model_directory: str = "data/models") -> bool:
        """
        Load previously trained models

        Args:
            model_directory: Directory containing saved models

        Returns:
            True if models loaded successfully
        """
        try:
            if not os.path.exists(model_directory):
                self.logger.warning(f"Model directory {model_directory} does not exist")
                return False

            # Find latest model files
            model_files = [f for f in os.listdir(model_directory) if f.endswith('.h5') and self.symbol in f]

            if not model_files:
                self.logger.warning("No model files found")
                return False

            # Load latest models for each type
            loaded_models = {}
            for model_file in model_files:
                model_path = os.path.join(model_directory, model_file)
                model_name = model_file.split('_')[0] + '_' + model_file.split('_')[1]  # e.g., 'lstm_direction'

                try:
                    model = keras.models.load_model(model_path)
                    loaded_models[model_name] = model
                    self.logger.info(f"Loaded model: {model_name} from {model_file}")
                except Exception as e:
                    self.logger.error(f"Error loading model {model_file}: {e}")

            self.models = loaded_models
            self.logger.info(f"Successfully loaded {len(loaded_models)} models")

            # Load scalers if available
            scaler_file = os.path.join(model_directory, 'multi_frame_scalers.pkl')
            if os.path.exists(scaler_file):
                try:
                    import pickle
                    with open(scaler_file, 'rb') as f:
                        scaler_data = pickle.load(f)
                        if 'feature_scaler' in scaler_data:
                            self.feature_scaler = scaler_data['feature_scaler']
                            self.logger.info("Loaded feature scaler")
                        if 'target_scaler' in scaler_data:
                            self.target_scaler = scaler_data['target_scaler']
                            self.logger.info("Loaded target scaler")
                except Exception as e:
                    self.logger.warning(f"Could not load scalers: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading trained models: {e}")
            return False

    def get_model_performance_summary(self) -> Dict:
        """Get summary of model performance"""
        if not self.training_history:
            return {'error': 'No training history available'}

        summary = {
            'total_models': len(self.training_history),
            'model_performance': {}
        }

        for model_name, results in self.training_history.items():
            metrics = results['test_metrics']
            summary['model_performance'][model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'auc': metrics.get('auc', 0),
                'loss': metrics.get('loss', 0)
            }

        # Find best performing model
        if summary['model_performance']:
            best_model = max(summary['model_performance'].keys(),
                           key=lambda x: summary['model_performance'][x]['accuracy'])
            summary['best_model'] = best_model
            summary['best_accuracy'] = summary['model_performance'][best_model]['accuracy']

        return summary

    def predict(self, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading prediction from market data

        Args:
            market_data: Preprocessed market data

        Returns:
            Dictionary with prediction results or None if error
        """
        try:
            # Check if models are available
            if not self.check_models_health():
                # Return mock prediction for validation/testing purposes
                self.logger.info("No trained models available, returning mock prediction for validation")
                return {
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'price_prediction': market_data['close'].iloc[-1] if 'close' in market_data.columns else None,
                    'timestamp': datetime.now(),
                    'mock_prediction': True
                }

            # Prepare data for prediction
            prediction_data = self.prepare_prediction_data(market_data)
            if prediction_data is None:
                return None

            # Load best model
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]
            if not model_files:
                return None

            best_model_path = os.path.join(self.models_dir, model_files[0])
            model = tf.keras.models.load_model(best_model_path)

            # Make prediction
            prediction = model.predict(prediction_data, verbose=0)

            # Convert to trading signal
            if prediction.shape[1] == 1:
                # Binary classification (up/down)
                confidence = float(prediction[0][0])
                direction = 'BUY' if confidence > 0.5 else 'SELL'
                confidence = confidence if confidence > 0.5 else 1 - confidence
            else:
                # Multi-class classification
                predicted_class = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
                direction = ['SELL', 'HOLD', 'BUY'][predicted_class]

            return {
                'direction': direction,
                'confidence': confidence,
                'prediction_raw': prediction.tolist(),
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return None

    def check_models_health(self) -> bool:
        """
        Check if trained models are healthy and ready for prediction

        Returns:
            True if models are healthy, False otherwise
        """
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_dir):
                return False

            # Check if any model files exist
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]
            if not model_files:
                return False

            # Try to load the first model to check if it's valid
            model_path = os.path.join(self.models_dir, model_files[0])
            try:
                model = tf.keras.models.load_model(model_path)
                # Try a simple prediction with correct input shape
                if model.input_shape and len(model.input_shape) == 3:
                    # Use the actual input shape from the model
                    batch_size = 1
                    timesteps = model.input_shape[1] if model.input_shape[1] else 24
                    features = model.input_shape[2] if model.input_shape[2] else 32
                    dummy_input = np.random.rand(batch_size, timesteps, features).astype(np.float32)
                else:
                    # Fallback to reasonable defaults
                    dummy_input = np.random.rand(1, 24, 32).astype(np.float32)
                prediction = model.predict(dummy_input, verbose=0)
                return prediction is not None
            except Exception:
                return False

        except Exception as e:
            self.logger.error(f"Error checking models health: {e}")
            return False