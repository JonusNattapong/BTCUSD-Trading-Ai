#!/usr/bin/env python3
"""
BTCUSD Trading AI Demo Script
This script demonstrates how to use the trained AI model for BTCUSD trading predictions.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_model_and_predict():
    """
    Load the trained model and make predictions
    """
    try:
        # Load the trained model
        model_path = 'models/btcusd_lstm_model.h5'
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully!")

        # Load recent data
        data_path = 'data/btcusd_with_indicators.csv'
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df = df.dropna()

        # Features used for training
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                    'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_SMA']

        # Recreate scaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])

        # Get last 60 days for prediction
        last_sequence = scaled_data[-60:]

        # Make prediction for next day
        prediction = model.predict(last_sequence.reshape(1, -1, len(features)), verbose=0)
        
        # Create array for inverse transform (match feature dimensions)
        pred_array = np.zeros((1, len(features)))
        pred_array[0, 0] = prediction[0, 0]  # Put prediction in Close price position
        
        predicted_price = scaler.inverse_transform(pred_array)[0, 0]

        current_price = df.iloc[-1]['Close']

        print(f"Current BTC Price: ${current_price:.2f}")
        print(f"Predicted Next Day Price: ${predicted_price:.2f}")

        # Generate trading signal
        price_change = (predicted_price - current_price) / current_price * 100

        if price_change > 2:
            signal = "🟢 BUY"
        elif price_change < -2:
            signal = "🔴 SELL"
        else:
            signal = "🟡 HOLD"

        print(f"Price Change: {price_change:.2f}%")
        print(f"Trading Signal: {signal}")

        return predicted_price, signal

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run the training scripts first:")
        print("1. python src/data_collector.py")
        print("2. python src/train_model.py")
        return None, None

if __name__ == "__main__":
    print("🚀 BTCUSD Trading AI Demo")
    print("=" * 40)

    predicted_price, signal = load_model_and_predict()

    if predicted_price:
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        print("Check the data/ and models/ directories for detailed results.")