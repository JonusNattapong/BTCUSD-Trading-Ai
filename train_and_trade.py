#!/usr/bin/env python3
"""
BTCUSD AI Trading System - Training & Trading Workflow
Complete workflow for training AI models and starting live trading
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main training and trading workflow"""
    print("🚀 BTCUSD AI Trading System - Training & Trading Workflow")
    print("=" * 60)

    # Step 1: Train AI Models
    print("\n🤖 Step 1: Training AI Models")
    print("-" * 30)

    try:
        from real_data_ai_trainer import RealDataAITrainer

        print("Initializing AI trainer...")
        ai_trainer = RealDataAITrainer()

        print("Training ensemble models (this may take several minutes)...")
        training_results = ai_trainer.train_ensemble_models(epochs=20, batch_size=64)

        if 'error' in training_results:
            print(f"❌ Training failed: {training_results['error']}")
            return
        else:
            print("✅ AI models trained successfully!")
            print(f"   Models saved to: models/")
            print(f"   Training completed in: {training_results.get('training_time', 'N/A')}")

    except Exception as e:
        print(f"❌ AI training failed: {e}")
        return

    # Step 2: Run Comprehensive Backtesting
    print("\n📊 Step 2: Running Comprehensive Backtesting")
    print("-" * 40)

    try:
        from backtesting_framework import BacktestingFramework

        print("Initializing backtesting framework...")
        backtester = BacktestingFramework()

        print("Running AI model backtest...")
        backtest_results = backtester.run_ai_model_backtest(lookback_days=30, test_days=8)

        if 'error' in backtest_results:
            print(f"❌ Backtesting failed: {backtest_results['error']}")
            return
        else:
            print("✅ Backtesting completed successfully!")
            metrics = backtest_results.get('performance_metrics', {})
            print(".2f")
            print(".2f")
            print(".1f")

    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return

    # Step 3: Validate System Health
    print("\n🔍 Step 3: Final System Validation")
    print("-" * 35)

    try:
        from system_validator import SystemValidator

        print("Running final system validation...")
        validator = SystemValidator()
        validation_results = validator.run_full_validation()

        if validation_results['overall_status'] == 'passed':
            print("✅ System validation PASSED!")
        else:
            print("❌ System validation FAILED!")
            print("   Check reports/ for detailed error information")
            return

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return

    # Step 4: Start Live Trading (Paper Trading Mode)
    print("\n⚡ Step 4: Starting Live Trading (Paper Trading Mode)")
    print("-" * 50)

    try:
        from live_trading_system import LiveTradingSystem

        print("⚠️  IMPORTANT: Starting in PAPER TRADING mode")
        print("   No real money will be used - this is for testing only")
        print("   Monitor performance for 24-48 hours before considering live trading")
        print()

        # Ask user confirmation
        response = input("Do you want to start paper trading? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("Paper trading cancelled. You can start it later with:")
            print("python src/live_trading_system.py")
            return

        print("Starting live trading system...")
        trading_system = LiveTradingSystem(paper_trading=True)

        print("🚀 Live trading system started!")
        print("   Monitor logs/ for real-time activity")
        print("   Check reports/ for daily performance summaries")
        print("   Press Ctrl+C to stop trading")

        # Keep the system running
        try:
            while True:
                time.sleep(60)  # Check every minute
                # Could add health checks here

        except KeyboardInterrupt:
            print("\n🛑 Shutting down trading system...")
            trading_system.stop()
            print("✅ Trading system stopped safely")

    except Exception as e:
        print(f"❌ Live trading startup failed: {e}")
        return

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('src') or not os.path.exists('models'):
        print("❌ Error: Please run this script from the BTCUSD-Trading-Ai root directory")
        sys.exit(1)

    # Run the workflow
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()