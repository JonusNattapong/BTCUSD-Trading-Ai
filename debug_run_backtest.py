import traceback

from src.backtesting_framework import BacktestingFramework

if __name__ == '__main__':
    try:
        bf = BacktestingFramework()
        print("BacktestingFramework initialized")
        res = bf.run_ai_model_backtest(lookback_days=200, test_days=30)
        print('Backtest result:', res)
    except Exception as e:
        print('Exception during backtest:')
        traceback.print_exc()
