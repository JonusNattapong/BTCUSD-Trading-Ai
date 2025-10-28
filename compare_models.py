#!/usr/bin/env python3
"""
BTCUSD Trading AI Model Comparison
Compare single-frame vs multi-frame model predictions
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def compare_models():
    """
    Compare predictions from both single-frame and multi-frame models
    """
    print("🔄 BTCUSD Trading AI Model Comparison")
    print("=" * 50)

    try:
        # Load single-frame model
        import tensorflow as tf
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        # Single-frame model
        print("Loading single-frame model...")
        single_model = tf.keras.models.load_model('models/btcusd_lstm_model.h5')

        # Load data for single-frame prediction
        df = pd.read_csv('data/btcusd_with_indicators.csv', index_col=0, parse_dates=True)
        df = df.dropna()
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                    'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_SMA']

        # Recreate scaler for single-frame
        scaler_single = MinMaxScaler()
        scaled_data = scaler_single.fit_transform(df[features])
        last_sequence = scaled_data[-60:]  # Last 60 days

        # Make single-frame prediction
        single_pred = single_model.predict(last_sequence.reshape(1, -1, len(features)), verbose=0)

        # Inverse transform
        pred_array = np.zeros((1, len(features)))
        pred_array[0, 0] = single_pred[0, 0]
        single_pred_price = scaler_single.inverse_transform(pred_array)[0, 0]

        # Multi-frame model
        print("Loading multi-frame model...")
        from multi_frame_strategy import load_multi_frame_model_and_scalers, prepare_multi_frame_prediction_data, predict_multi_frame, generate_multi_frame_signal

        multi_model, multi_scalers = load_multi_frame_model_and_scalers()
        processed_data, _ = prepare_multi_frame_prediction_data()
        multi_pred = predict_multi_frame(multi_model, processed_data)

        # Generate multi-frame signal
        current_price = df.iloc[-1]['Close']
        multi_signal, multi_pred_price, multi_change = generate_multi_frame_signal(multi_pred, current_price, multi_scalers)

        # Calculate single-frame signal
        single_change = (single_pred_price - current_price) / current_price
        if single_change > 0.02:
            single_signal = "BUY"
        elif single_change < -0.02:
            single_signal = "SELL"
        else:
            single_signal = "HOLD"

        # Display comparison
        print(f"\n📊 Current BTC Price: ${current_price:.2f}")
        print(f"\n🤖 Single-Frame Model (LSTM):")
        print(f"   Predicted Price: ${single_pred_price:.2f}")
        print(f"   Price Change: {single_change:.2f}%")
        print(f"   Signal: {single_signal}")

        print(f"\n🚀 Multi-Frame Model (1H + 4H + 1D):")
        print(f"   Predicted Price: ${multi_pred_price:.2f}")
        print(f"   Price Change: {multi_change:.2f}%")
        print(f"   Signal: {multi_signal}")

        print(f"\n💡 Analysis:")
        if single_signal == multi_signal:
            print(f"   ✅ Models agree: Both predict {single_signal}")
        else:
            print(f"   ⚠️  Models disagree: Single-frame suggests {single_signal}, Multi-frame suggests {multi_signal}")
            print(f"      Multi-frame model combines multiple timeframes for potentially better accuracy")

        # Performance comparison
        print(f"\n📈 Model Capabilities:")
        print(f"   Single-Frame: Daily data analysis (60-day sequences)")
        print(f"   Multi-Frame: Cross-timeframe analysis (1H, 4H, 1D combined)")
        print(f"   Multi-Frame Advantage: Better pattern recognition across time scales")

    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        print("Make sure both models are trained and data is available.")

if __name__ == "__main__":
    import numpy as np
    compare_models()