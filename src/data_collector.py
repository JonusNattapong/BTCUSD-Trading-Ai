import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
import yfinance as yf

try:
    from real_time_data import RealTimeDataConnector
except ImportError:
    from src.real_time_data import RealTimeDataConnector

class HistoricalDataCollector:
    """
    Historical market data collection and management system
    Supports multiple data sources and comprehensive data preprocessing
    """

    def __init__(self):
        self.data_dir = "data"
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")

        # Initialize data sources
        self.binance_connector = RealTimeDataConnector()

        # Data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # Setup directories and logging
        self._setup_directories()
        self._setup_logging()

        print("📊 Historical data collector initialized")

    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/backtests", exist_ok=True)

    def _setup_logging(self):
        """Setup data collection logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'data_collection.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataCollector')

    def collect_binance_data(self, symbol: str = "BTCUSDT", interval: str = "1h",
                           start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Collect historical data from Binance

        Args:
            symbol: Trading pair symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date (YYYY-MM-DD) or None for recent data
            end_date: End date (YYYY-MM-DD) or None for current date

        Returns:
            DataFrame with historical data or None if error
        """
        try:
            # Calculate days based on dates
            if start_date and end_date:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                days = (end - start).days
            elif start_date:
                start = pd.to_datetime(start_date)
                end = pd.Timestamp.now()
                days = (end - start).days
            else:
                # Default to 30 days of recent data
                days = 30

            self.logger.info(f"Collecting {days} days of {symbol} data from Binance")

            # Calculate number of data points needed
            intervals_per_day = {
                '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
                '1h': 24, '2h': 12, '4h': 6, '6h': 4, '8h': 3, '12h': 2, '1d': 1
            }

            total_limit = days * intervals_per_day.get(interval, 24)

            # Convert dates to timestamps
            start_timestamp = None
            end_timestamp = None
            if start_date:
                start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
            if end_date:
                end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000)

            # Collect data in chunks if needed
            all_data = []
            current_start = start_timestamp

            while total_limit > 0:
                chunk_limit = min(total_limit, 1000)  # Binance max limit is 1000

                chunk_data = self.binance_connector.get_historical_data(
                    symbol, interval, chunk_limit, current_start, end_timestamp
                )

                if chunk_data is None or chunk_data.empty:
                    break

                all_data.append(chunk_data)

                # Update for next chunk
                total_limit -= len(chunk_data)
                if current_start:
                    # Move start time forward by the number of intervals we got
                    interval_ms = self._interval_to_ms(interval)
                    current_start = chunk_data.index[-1].timestamp() * 1000 + interval_ms

                # Small delay to avoid rate limits
                time.sleep(0.1)

            if not all_data:
                self.logger.error(f"No data collected for {symbol}")
                return None

            # Combine all chunks
            combined_data = pd.concat(all_data).drop_duplicates()

            # Sort by timestamp
            combined_data = combined_data.sort_index()

            # Save raw data
            filename = f"{symbol}_{interval}_{days}d_raw.csv"
            filepath = os.path.join(self.raw_data_dir, filename)
            combined_data.to_csv(filepath)

            self.logger.info(f"Saved {len(combined_data)} data points to {filepath}")
            return combined_data

        except Exception as e:
            self.logger.error(f"Error collecting Binance data for {symbol}: {e}")
            return None

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        interval_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return interval_map.get(interval, 60 * 60 * 1000)  # Default to 1h

    def collect_yahoo_finance_data(self, symbol: str = "BTC-USD", period: str = "2y",
                                 interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Collect historical data from Yahoo Finance

        Args:
            symbol: Yahoo Finance symbol (e.g., 'BTC-USD')
            period: Period to collect (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with historical data or None if error
        """
        try:
            self.logger.info(f"Collecting {period} of {symbol} data from Yahoo Finance")

            # Download data
            data = yf.download(symbol, period=period, interval=interval, progress=False)

            if data.empty:
                self.logger.error(f"No data received from Yahoo Finance for {symbol}")
                return None

            # Rename columns to match our format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            data = data[['open', 'high', 'low', 'close', 'volume']]

            # Save raw data
            filename = f"{symbol}_{period}_{interval}_yahoo_raw.csv"
            filepath = os.path.join(self.raw_data_dir, filename)
            data.to_csv(filepath)

            self.logger.info(f"Saved {len(data)} data points from Yahoo Finance to {filepath}")
            return data

        except Exception as e:
            self.logger.error(f"Error collecting Yahoo Finance data for {symbol}: {e}")
            return None

    def preprocess_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Preprocess raw market data for AI training

        Args:
            data: Raw OHLCV data
            symbol: Trading symbol

        Returns:
            Preprocessed DataFrame with technical indicators
        """
        try:
            self.logger.info(f"Preprocessing data for {symbol}")

            df = data.copy()

            # Basic data cleaning
            df = df.dropna()  # Remove NaN values
            df = df[df['volume'] > 0]  # Remove zero volume entries

            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Technical indicators
            df = self._add_technical_indicators(df)

            # Price-based features
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['gap_down'] = (df['close'].shift(1) - df['open']) / df['close'].shift(1)

            # Volume-based features
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']

            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month

            # Target variables for prediction
            df['future_return_1h'] = df['returns'].shift(-1)  # 1-hour ahead return
            df['future_return_4h'] = df['returns'].rolling(window=4).mean().shift(-4)  # 4-hour ahead return
            df['future_return_24h'] = df['returns'].rolling(window=24).mean().shift(-24)  # 24-hour ahead return

            # Price direction (for classification)
            df['direction_1h'] = np.where(df['future_return_1h'] > 0, 1, 0)
            df['direction_4h'] = np.where(df['future_return_4h'] > 0, 1, 0)
            df['direction_24h'] = np.where(df['future_return_24h'] > 0, 1, 0)

            # Remove rows with NaN from calculations
            df = df.dropna()

            self.logger.info(f"Preprocessed data shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error preprocessing data for {symbol}: {e}")
            return data

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            # RSI
            def calculate_rsi(data, window=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df['rsi_14'] = calculate_rsi(df['close'], 14)

            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # Stochastic Oscillator
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()

            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

            return df

        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df

    def collect_comprehensive_dataset(self, symbol: str = "BTCUSDT",
                                    days: int = 730) -> Optional[pd.DataFrame]:
        """
        Collect and preprocess comprehensive dataset for AI training

        Args:
            symbol: Trading symbol
            days: Number of days of historical data

        Returns:
            Preprocessed DataFrame ready for AI training
        """
        try:
            self.logger.info(f"Collecting comprehensive dataset for {symbol}")

            # For longer periods, use Yahoo Finance first
            if days > 100:
                yahoo_symbol = "BTC-USD" if symbol == "BTCUSDT" else symbol.replace("USDT", "-USD")
                period = f"{min(days//365 + 1, 5)}y"  # Max 5y for Yahoo
                raw_data = self.collect_yahoo_finance_data(yahoo_symbol, period, "1h")
                if raw_data is not None and not raw_data.empty:
                    self.logger.info(f"Using Yahoo Finance data: {len(raw_data)} points")
                else:
                    # Fallback to Binance
                    end_date = pd.Timestamp.now()
                    start_date = end_date - timedelta(days=days)
                    raw_data = self.collect_binance_data(symbol, "1h", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            else:
                # For shorter periods, use Binance
                end_date = pd.Timestamp.now()
                start_date = end_date - timedelta(days=days)
                raw_data = self.collect_binance_data(symbol, "1h", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if raw_data is None or raw_data.empty:
                self.logger.error("Unable to collect data from any source")
                return None

            # Preprocess data
            processed_data = self.preprocess_data(raw_data, symbol)

            # Save processed data
            filename = f"{symbol}_{days}d_processed.csv"
            filepath = os.path.join(self.processed_data_dir, filename)
            processed_data.to_csv(filepath)

            # Cache the data
            self.data_cache[symbol] = processed_data

            self.logger.info(f"Comprehensive dataset ready: {processed_data.shape}")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error collecting comprehensive dataset: {e}")
            return None

    def get_training_data(self, symbol: str, lookback_days: int = 365,
                         test_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and testing data splits

        Args:
            symbol: Trading symbol
            lookback_days: Days of data to use
            test_split: Fraction of data for testing

        Returns:
            Tuple of (training_data, testing_data)
        """
        if symbol not in self.data_cache:
            data = self.collect_comprehensive_dataset(symbol, lookback_days)
            if data is None:
                return None, None
        else:
            data = self.data_cache[symbol]

        # Split data
        split_idx = int(len(data) * (1 - test_split))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        self.logger.info(f"Training data: {train_data.shape}, Testing data: {test_data.shape}")
        return train_data, test_data

    def update_real_time_data(self, symbol: str = "BTCUSDT"):
        """
        Update dataset with latest real-time data

        Args:
            symbol: Trading symbol to update
        """
        try:
            # Get latest price data
            current_price = self.binance_connector.get_current_price(symbol)

            if current_price and symbol in self.data_cache:
                # Add new data point
                new_row = pd.DataFrame({
                    'open': [current_price],
                    'high': [current_price],
                    'low': [current_price],
                    'close': [current_price],
                    'volume': [0]  # Volume not available in real-time
                }, index=[datetime.now()])

                # Append to existing data
                self.data_cache[symbol] = pd.concat([self.data_cache[symbol], new_row])

                # Keep only recent data (last 2 years)
                cutoff_date = datetime.now() - timedelta(days=730)
                self.data_cache[symbol] = self.data_cache[symbol][
                    self.data_cache[symbol].index > cutoff_date
                ]

                self.logger.info(f"Updated {symbol} data with latest price: ${current_price}")

        except Exception as e:
            self.logger.error(f"Error updating real-time data for {symbol}: {e}")

    def collect_historical_data(self, symbol: str = "BTCUSDT", start_date: str = None,
                               end_date: str = None, interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Collect historical data for a symbol (main entry point)

        Args:
            symbol: Trading symbol (BTCUSDT for Binance, BTC-USD for Yahoo)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            DataFrame with historical data or None if error
        """
        try:
            # Determine data source based on symbol format
            if symbol.endswith('USDT'):
                # Binance format
                return self.collect_binance_data(symbol, interval, start_date, end_date)
            else:
                # Yahoo Finance format
                period = "2y" if start_date is None else None
                return self.collect_yahoo_finance_data(symbol, period, interval)

        except Exception as e:
            self.logger.error(f"Error collecting historical data for {symbol}: {e}")
            return None

    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        summary = {
            'raw_files': len(os.listdir(self.raw_data_dir)) if os.path.exists(self.raw_data_dir) else 0,
            'processed_files': len(os.listdir(self.processed_data_dir)) if os.path.exists(self.processed_data_dir) else 0,
            'cached_symbols': list(self.data_cache.keys()),
            'total_data_points': sum(len(df) for df in self.data_cache.values())
        }

        return summary

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        return self._add_technical_indicators(df.copy())

# Legacy functions for backward compatibility
def download_btcusd_data(start_date='2018-01-01', end_date=None, save_path='data/btcusd_data.csv'):
    """
    Legacy function - use HistoricalDataCollector instead
    """
    collector = HistoricalDataCollector()
    data = collector.collect_yahoo_finance_data("BTC-USD", "5y", "1d")

    if data is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        print(f"Data saved to {save_path}")

    return data

def add_technical_indicators(df):
    """
    Legacy function - use HistoricalDataCollector.preprocess_data instead
    """
    collector = HistoricalDataCollector()
    return collector._add_technical_indicators(df.copy())

if __name__ == "__main__":
    # Test the enhanced data collector
    collector = HistoricalDataCollector()

    print("Testing data collection...")

    # Collect comprehensive BTC data
    btc_data = collector.collect_comprehensive_dataset("BTCUSDT", 365)

    if btc_data is not None:
        print(f"✅ Collected {len(btc_data)} data points for BTCUSDT")
        print(f"Features available: {list(btc_data.columns)}")

        # Get training/test split
        train_data, test_data = collector.get_training_data("BTCUSDT", 365, 0.2)
        print(f"Training data: {train_data.shape if train_data is not None else 'None'}")
        print(f"Testing data: {test_data.shape if test_data is not None else 'None'}")

    # Show data summary
    summary = collector.get_data_summary()
    print(f"Data summary: {summary}")

    print("Data collection test complete!")