#!/usr/bin/env python3
"""
BTCUSD AI Model Training Script
Trains the ensemble AI models for BTCUSD trading predictions
"""

import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def train_ai_models():
    """Train the AI models"""
    print("🤖 BTCUSD AI Model Training")
    print("=" * 40)

    try:
        from real_data_ai_trainer import RealDataAITrainer

        print("Initializing AI trainer...")
        ai_trainer = RealDataAITrainer()

        print("Collecting and preparing training data...")
        # The trainer will collect data internally

        print("Training ensemble models...")
        print("This may take 10-30 minutes depending on your hardware...")
        start_time = time.time()

        training_results = ai_trainer.train_ensemble_models(epochs=20, batch_size=64, lookback_days=180)

        training_time = time.time() - start_time

        if 'error' in training_results:
            print(f"❌ Training failed: {training_results['error']}")
            return False
        else:
            print("✅ AI models trained successfully!")
            print(".1f")
            print(f"   Models saved to: models/")
            print(f"   Training history: {training_results.get('models_trained', 'N/A')} models")

            # Show model performance if available
            if 'performance' in training_results:
                perf = training_results['performance']
                print("   Model Performance:")
                print(".3f")
                print(".3f")

            return True

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_training():
    """Validate that models were trained successfully"""
    print("\n🔍 Validating Training Results")
    print("-" * 30)

    try:
        from real_data_ai_trainer import RealDataAITrainer

        trainer = RealDataAITrainer()

        # Check if models are healthy
        models_healthy = trainer.check_models_health()

        if models_healthy:
            print("✅ Models are healthy and ready for prediction")

            # Test a prediction
            print("Testing prediction generation...")
            sample_data = {
                'open': [50000, 50100, 50200],
                'high': [50500, 50600, 50700],
                'low': [49500, 49600, 49700],
                'close': [50200, 50300, 50400],
                'volume': [100, 110, 120]
            }
            df = pd.DataFrame(sample_data)
            prediction = trainer.predict(df)

            if prediction:
                print("✅ Prediction generation working")
                print(f"   Sample prediction: {prediction.get('signal', 'N/A')}")
            else:
                print("❌ Prediction generation failed")

        else:
            print("❌ Models are not healthy - check training logs")

        return models_healthy

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    import pandas as pd

    # Check if we're in the right directory
    if not os.path.exists('src') or not os.path.exists('models'):
        print("❌ Error: Please run this script from the BTCUSD-Trading-Ai root directory")
        sys.exit(1)

    print(f"Training started at: {datetime.now()}")
    print()

    # Train the models
    success = train_ai_models()

    if success:
        # Validate the training
        validation_success = validate_training()

        if validation_success:
            print("\n🎉 Training and validation completed successfully!")
            print("\nNext steps:")
            print("1. Run backtesting: python src/backtesting_framework.py")
            print("2. Start paper trading: python train_and_trade.py")
            print("3. Monitor performance: check logs/ and reports/")
        else:
            print("\n⚠️  Training completed but validation failed")
            print("   Check the logs for more details")
    else:
        print("\n❌ Training failed - check the error messages above")
        sys.exit(1)