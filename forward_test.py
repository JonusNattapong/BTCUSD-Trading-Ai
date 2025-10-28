import time
import signal
import sys
from datetime import datetime, timedelta
from src.live_trading_system import LiveTradingSystem

def signal_handler(signum, frame):
    """Handle interrupt signal to gracefully shutdown"""
    print("\n🛑 Received interrupt signal. Shutting down paper trading...")
    sys.exit(0)

def run_forward_test(duration_minutes=5):
    """
    Run forward test (paper trading) for a specified duration

    Args:
        duration_minutes: How long to run the test in minutes
    """
    print("🚀 Starting Forward Test (Paper Trading Mode)")
    print("=" * 50)
    print(f"Duration: {duration_minutes} minutes")
    print("Mode: Paper Trading (No real money)")
    print("Monitoring real-time market data and AI predictions")
    print()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create paper trading configuration
        paper_config = {
            'capital': {
                'initial_amount': 1000.0,  # Paper money
                'growth_target': 'conservative',  # Conservative for testing
                'max_daily_loss': 10.0  # Small loss limit for safety
            },
            'trading': {
                'symbol': 'BTCUSDT',
                'max_position_size': 0.05,  # Small position size for testing
                'min_trade_size': 1.0,  # Very small minimum
                'max_open_positions': 1,
                'trading_hours': {'start': '00:00', 'end': '23:59'}
            },
            'ai': {
                'model_update_frequency': 'daily',
                'prediction_threshold': 0.6,
                'confidence_required': 0.7
            },
            'risk': {
                'max_drawdown': 0.05,  # 5% max drawdown for paper trading
                'var_limit': 0.02,
                'stress_test_frequency': 'daily'
            },
            'monitoring': {
                'health_check_interval': 30,  # More frequent checks for testing
                'performance_log_interval': 60,  # Log every minute during test
                'alert_thresholds': {
                    'high_pnl': 10.0,  # Small thresholds for testing
                    'high_loss': -5.0,
                    'connection_issues': 3
                }
            },
            'paper_trading': True  # Flag to indicate paper trading mode
        }

        # Initialize live trading system with paper trading config
        print("Initializing Live Trading System (Paper Mode)...")
        trading_system = LiveTradingSystem(config=paper_config)

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        print(f"Test started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test will end at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nMonitoring for the next {} minutes...".format(duration_minutes))
        print("Press Ctrl+C to stop early")
        print("-" * 50)

        # Run the test
        while datetime.now() < end_time:
            try:
                # Check if system is still running
                if hasattr(trading_system, 'is_running') and not trading_system.is_running:
                    print("⚠️  Trading system stopped unexpectedly")
                    break

                # Small delay to prevent excessive CPU usage
                time.sleep(10)

                # Print progress update every minute
                elapsed = datetime.now() - start_time
                remaining = end_time - datetime.now()

                if elapsed.seconds % 60 < 10:  # Print roughly every minute
                    print(f"⏱️  Elapsed: {elapsed.seconds // 60}m {elapsed.seconds % 60}s | "
                          f"Remaining: {remaining.seconds // 60}m {remaining.seconds % 60}s")

            except KeyboardInterrupt:
                print("\n🛑 Test interrupted by user")
                break
            except Exception as e:
                print(f"⚠️  Error during test: {e}")
                continue

        # Test completed
        end_actual = datetime.now()
        duration_actual = end_actual - start_time

        print("\n" + "=" * 50)
        print("✅ Forward Test Completed")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended: {end_actual.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration_actual.seconds // 60}m {duration_actual.seconds % 60}s")
        print("=" * 50)

        # Generate summary report
        generate_test_summary(trading_system, start_time, end_actual)

    except Exception as e:
        print(f"❌ Error during forward test: {e}")
        import traceback
        traceback.print_exc()

def generate_test_summary(trading_system, start_time, end_time):
    """Generate a summary of the forward test results"""
    print("\n📊 Forward Test Summary")
    print("-" * 30)

    try:
        # Check if we have access to trading data
        if hasattr(trading_system, 'capital_growth_manager') and trading_system.capital_growth_manager:
            cgm = trading_system.capital_growth_manager
            print(f"Initial Capital: ${cgm.initial_capital:,.2f}")
            print(f"Current Capital: ${cgm.current_capital:,.2f}")
            print(f"Total Return: {((cgm.current_capital / cgm.initial_capital) - 1) * 100:.2f}%")

        if hasattr(trading_system, 'risk_manager') and trading_system.risk_manager:
            rm = trading_system.risk_manager
            print(f"Total Trades: {getattr(rm, 'total_trades', 'N/A')}")
            print(f"Win Rate: {getattr(rm, 'win_rate', 'N/A')}%")

        # Check logs directory for any activity
        import os
        if os.path.exists('logs'):
            log_files = [f for f in os.listdir('logs') if f.startswith('trading_')]
            if log_files:
                print(f"Log files generated: {len(log_files)}")
                # Show recent log entries
                latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join('logs', x)))
                log_path = os.path.join('logs', latest_log)
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()[-5:]  # Last 5 lines
                        if lines:
                            print("Recent log activity:")
                            for line in lines:
                                print(f"  {line.strip()}")
                except:
                    pass

        # Check reports directory
        if os.path.exists('reports'):
            report_files = [f for f in os.listdir('reports') if f.endswith('.json')]
            if report_files:
                print(f"Report files generated: {len(report_files)}")

    except Exception as e:
        print(f"Could not generate detailed summary: {e}")

    print(f"\nTest Duration: {(end_time - start_time).total_seconds() / 60:.1f} minutes")
    print("Check logs/ and reports/ directories for detailed activity logs")

if __name__ == '__main__':
    # Run forward test for 5 minutes by default
    duration = 5
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            print("Invalid duration argument, using default 5 minutes")

    run_forward_test(duration)