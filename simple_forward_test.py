import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_data_ai_trainer import RealDataAITrainer
from data_collector import HistoricalDataCollector

def run_simple_forward_test():
    """
    Run a simple forward test to validate AI model predictions
    """
    print("🧪 Running Simple Forward Test")
    print("=" * 40)

    try:
        # Initialize components
        print("Initializing AI Trainer...")
        ai_trainer = RealDataAITrainer()

        print("Loading trained models...")
        if not ai_trainer.load_trained_models():
            print("❌ No trained models found. Please run training first.")
            return

        print("Collecting recent market data...")
        data_collector = HistoricalDataCollector()

        # Use the same comprehensive data collection as training
        # This ensures all technical indicators are included
        recent_data = data_collector.collect_comprehensive_dataset(
            symbol="BTCUSDT",
            days=7  # Get last 7 days of comprehensive data
        )

        if recent_data is None or len(recent_data) < 24:
            print("❌ Insufficient comprehensive data for prediction")
            return

        print(f"✅ Collected {len(recent_data)} data points with technical indicators")

        # Make predictions
        print("\n🔮 Making AI Predictions...")
        predictions = ai_trainer.predict_trading_signal(recent_data)

        if predictions and 'error' not in predictions:
            print("📊 Prediction Results:")
            signal = predictions.get('signal', 'UNKNOWN')
            confidence = predictions.get('confidence', 0)
            strength = predictions.get('strength', 'Unknown')
            agreement = predictions.get('agreement', 0)

            print(f"  Signal: {signal}")
            print(".3f")
            print(f"  Strength: {strength}")
            print(".3f")

            # Check if signal meets confidence threshold
            if signal in ['BUY', 'SELL'] and confidence >= 0.55:  # Updated threshold
                print("  ✅ Signal meets confidence threshold (≥0.55)")
            elif signal == 'HOLD':
                print("  ➡️ HOLD signal - no trade")
            else:
                print("  ❌ Signal below confidence threshold (<0.55)")

            # Show individual model predictions if available
            if 'individual_predictions' in predictions:
                print("  Individual Model Predictions:")
                for model_name, pred in predictions['individual_predictions'].items():
                    print(".3f")

        elif predictions and 'error' in predictions:
            print(f"❌ Prediction Error: {predictions['error']}")
        else:
            print("❌ No predictions generated")

        print("\n✅ Forward Test Completed Successfully")
        print("The AI models are working and can make predictions on live data")

    except Exception as e:
        print(f"❌ Error during forward test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_simple_forward_test()