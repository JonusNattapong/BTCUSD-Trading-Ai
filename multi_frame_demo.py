#!/usr/bin/env python3
"""
BTCUSD Multi-Frame Trading AI Demo Script
This script demonstrates the multi-timeframe AI model for BTCUSD trading predictions.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_multi_frame_demo():
    """
    Run the multi-frame trading AI demo
    """
    try:
        from multi_frame_strategy import load_multi_frame_model_and_scalers, prepare_multi_frame_prediction_data, predict_multi_frame, generate_multi_frame_signal

        print("🚀 BTCUSD Multi-Frame Trading AI Demo")
        print("=" * 50)

        # Load model and scalers
        model, scalers = load_multi_frame_model_and_scalers()
        print("✓ Multi-frame model loaded successfully!")

        # Prepare prediction data
        processed_data, _ = prepare_multi_frame_prediction_data()

        if processed_data:
            # Make prediction
            prediction = predict_multi_frame(model, processed_data)

            if prediction is not None:
                # Load current price
                import pandas as pd
                daily_data = pd.read_csv('data/btcusd_with_indicators.csv', index_col=0, parse_dates=True)
                current_price = daily_data.iloc[-1]['Close']

                # Generate signal
                signal, predicted_price, price_change = generate_multi_frame_signal(prediction, current_price, scalers)

                print(f"\n📊 Analysis Results:")
                print(f"Current BTC Price: ${current_price:.2f}")
                print(f"Predicted Price: ${predicted_price:.2f}")
                print(f"Price Change: {price_change:.2f}%")

                if signal == 'BUY':
                    print(f"🟢 Trading Signal: {signal} (Price expected to rise)")
                elif signal == 'SELL':
                    print(f"🔴 Trading Signal: {signal} (Price expected to fall)")
                else:
                    print(f"🟡 Trading Signal: {signal} (Price expected to be stable)")

                print(f"\n💡 Multi-frame advantage: Combines 1H, 4H, and 1D analysis for better predictions!")

            else:
                print("❌ Could not make prediction - insufficient data")
        else:
            print("❌ Could not prepare prediction data")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed.")
    except FileNotFoundError:
        print("❌ Multi-frame model not found. Please train the model first:")
        print("python src/train_multi_frame.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_multi_frame_demo()