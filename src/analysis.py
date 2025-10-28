import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import download_btcusd_data, add_technical_indicators
from trading_strategy import backtest_strategy

def plot_price_and_indicators(data_path='data/btcusd_with_indicators.csv'):
    """
    Plot BTCUSD price with technical indicators
    """
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Price and moving averages
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='orange')
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='red')
    ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.1, color='gray')
    ax1.set_title('BTCUSD Price and Moving Averages')
    ax1.legend()
    ax1.grid(True)
    
    # RSI
    ax2.plot(df.index, df['RSI'], color='purple')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    # MACD
    ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax3.plot(df.index, df['Signal_Line'], label='Signal Line', color='red')
    ax3.set_title('MACD Indicator')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/btcusd_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_backtest_results(results):
    """
    Plot backtest results
    """
    trades_df = pd.DataFrame(results['trades'])
    
    if not trades_df.empty:
        plt.figure(figsize=(12, 6))
        
        # Plot trade points
        buy_trades = trades_df[trades_df['type'] == 'BUY']
        sell_trades = trades_df[trades_df['type'] == 'SELL']
        
        plt.scatter(buy_trades['date'], buy_trades['price'], color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_trades['date'], sell_trades['price'], color='red', marker='v', s=100, label='Sell')
        
        plt.title('Trading Strategy Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"Backtest Summary:")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Number of Trades: {len(results['trades'])}")

if __name__ == "__main__":
    # Download and process data if not exists
    import os
    if not os.path.exists('data/btcusd_with_indicators.csv'):
        print("Downloading data...")
        data = download_btcusd_data()
        data_with_indicators = add_technical_indicators(data)
        data_with_indicators.to_csv('data/btcusd_with_indicators.csv')
    
    # Plot analysis
    print("Generating analysis plots...")
    plot_price_and_indicators()
    
    # Run and plot backtest
    print("Running backtest analysis...")
    results = backtest_strategy()
    plot_backtest_results(results)