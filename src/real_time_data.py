import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
import threading
import queue

class RealTimeDataConnector:
    """
    Real-time market data connector for cryptocurrency trading
    Supports Binance API for BTCUSD and other major pairs
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.ws_base_url = "wss://stream.binance.com:9443/ws"

        # Data storage
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List] = {}
        self.order_book: Dict[str, Dict] = {}

        # WebSocket connections
        self.ws_connections = {}
        self.data_queue = queue.Queue()

        # Setup logging
        self._setup_logging()

        print("🔗 Real-time data connector initialized")

    def _setup_logging(self):
        """Setup data connector logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'data_connector.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataConnector')

    def get_current_price(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """
        Get current price for a symbol

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Current price or None if unavailable
        """
        try:
            endpoint = f"/api/v3/ticker/price"
            params = {"symbol": symbol}

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            data = response.json()
            price = float(data['price'])

            self.current_prices[symbol] = price
            return price

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str = "BTCUSDT", interval: str = "1h",
                          limit: int = 1000, start_time: int = None, end_time: int = None) -> Optional[pd.DataFrame]:
        """
        Get historical klines/candlestick data

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of data points (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            endpoint = "/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }

            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_asset_volume', 'number_of_trades',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

            df = pd.DataFrame(data, columns=columns)

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)

            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]

            self.logger.info(f"Retrieved {len(df)} historical data points for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    def start_price_stream(self, symbols: List[str] = None):
        """
        Start real-time price streaming for specified symbols

        Args:
            symbols: List of symbols to stream (default: ['btcusdt'])
        """
        if symbols is None:
            symbols = ['btcusdt']

        # Convert to lowercase for Binance streams
        stream_symbols = [symbol.lower() for symbol in symbols]

        # Create stream URL
        streams = [f"{symbol}@ticker" for symbol in stream_symbols]
        stream_url = f"{self.ws_base_url}/stream?streams={'/'.join(streams)}"

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get('stream') and '@ticker' in data['stream']:
                    ticker_data = data['data']
                    symbol = ticker_data['s'].upper()
                    price = float(ticker_data['c'])  # Close price

                    self.current_prices[symbol] = price

                    # Store in history
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []

                    self.price_history[symbol].append({
                        'timestamp': datetime.now(),
                        'price': price,
                        'volume': float(ticker_data.get('v', 0))
                    })

                    # Keep only last 1000 entries
                    if len(self.price_history[symbol]) > 1000:
                        self.price_history[symbol] = self.price_history[symbol][-1000:]

                    # Put in queue for processing
                    self.data_queue.put({
                        'type': 'price_update',
                        'symbol': symbol,
                        'price': price,
                        'timestamp': datetime.now()
                    })

            except Exception as e:
                self.logger.error(f"Error processing price stream message: {e}")

        def on_error(ws, error):
            self.logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.logger.info("WebSocket connection closed")

        def on_open(ws):
            self.logger.info(f"WebSocket connection opened for symbols: {symbols}")

        # Import websocket-client lazily so importing this module doesn't
        # fail in environments where the package is not installed.
        try:
            import websocket
        except Exception as e:
            self.logger.error(f"websocket-client is required for streaming but not available: {e}")
            return

        # Start WebSocket connection in a thread
        ws = websocket.WebSocketApp(stream_url,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close,
                                   on_open=on_open)

        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        self.ws_connections['price_stream'] = ws
        self.logger.info(f"Started price streaming for symbols: {symbols}")

    def get_24hr_stats(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """
        Get 24-hour price change statistics

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with 24hr stats or None if error
        """
        try:
            endpoint = "/api/v3/ticker/24hr"
            params = {"symbol": symbol}

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            data = response.json()

            return {
                'symbol': data['symbol'],
                'price_change': float(data['priceChange']),
                'price_change_percent': float(data['priceChangePercent']),
                'weighted_avg_price': float(data['weightedAvgPrice']),
                'prev_close_price': float(data['prevClosePrice']),
                'last_price': float(data['lastPrice']),
                'bid_price': float(data['bidPrice']),
                'ask_price': float(data['askPrice']),
                'open_price': float(data['openPrice']),
                'high_price': float(data['highPrice']),
                'low_price': float(data['lowPrice']),
                'volume': float(data['volume']),
                'quote_asset_volume': float(data['quoteAssetVolume']),
                'open_time': datetime.fromtimestamp(data['openTime'] / 1000),
                'close_time': datetime.fromtimestamp(data['closeTime'] / 1000),
                'count': int(data['count'])
            }

        except Exception as e:
            self.logger.error(f"Error getting 24hr stats for {symbol}: {e}")
            return None

    def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 100) -> Optional[Dict]:
        """
        Get order book depth

        Args:
            symbol: Trading pair symbol
            limit: Number of entries (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Order book data or None if error
        """
        try:
            endpoint = "/api/v3/depth"
            params = {"symbol": symbol, "limit": limit}

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            data = response.json()

            # Convert string prices/volumes to float
            bids = [[float(price), float(qty)] for price, qty in data['bids']]
            asks = [[float(price), float(qty)] for price, qty in data['asks']]

            order_book = {
                'last_update_id': data['lastUpdateId'],
                'bids': bids,  # [[price, quantity], ...]
                'asks': asks   # [[price, quantity], ...]
            }

            self.order_book[symbol] = order_book
            return order_book

        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    def get_recent_trades(self, symbol: str = "BTCUSDT", limit: int = 500) -> Optional[List[Dict]]:
        """
        Get recent trades

        Args:
            symbol: Trading pair symbol
            limit: Number of trades (max 1000)

        Returns:
            List of recent trades or None if error
        """
        try:
            endpoint = "/api/v3/trades"
            params = {"symbol": symbol, "limit": min(limit, 1000)}

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()

            trades = response.json()

            # Convert to more readable format
            formatted_trades = []
            for trade in trades:
                formatted_trades.append({
                    'id': trade['id'],
                    'price': float(trade['price']),
                    'qty': float(trade['qty']),
                    'quote_qty': float(trade['quoteQty']),
                    'time': datetime.fromtimestamp(trade['time'] / 1000),
                    'is_buyer_maker': trade['isBuyerMaker']
                })

            return formatted_trades

        except Exception as e:
            self.logger.error(f"Error getting recent trades for {symbol}: {e}")
            return None

    def get_price_history(self, symbol: str, hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Get price history from internal storage

        Args:
            symbol: Trading pair symbol
            hours: Hours of history to retrieve

        Returns:
            DataFrame with price history or None if no data
        """
        if symbol not in self.price_history:
            return None

        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [d for d in self.price_history[symbol] if d['timestamp'] > cutoff_time]

        if not recent_data:
            return None

        df = pd.DataFrame(recent_data)
        df.set_index('timestamp', inplace=True)

        return df

    def stop_all_streams(self):
        """Stop all WebSocket connections"""
        for name, ws in self.ws_connections.items():
            try:
                ws.close()
                self.logger.info(f"Closed {name} stream")
            except Exception as e:
                self.logger.error(f"Error closing {name} stream: {e}")

        self.ws_connections.clear()

    def get_market_status(self) -> Dict:
        """
        Get comprehensive market status

        Returns:
            Dictionary with current market information
        """
        btc_price = self.get_current_price("BTCUSDT")
        btc_stats = self.get_24hr_stats("BTCUSDT")

        return {
            'timestamp': datetime.now(),
            'btc_price': btc_price,
            'btc_24h_change': btc_stats.get('price_change_percent') if btc_stats else None,
            'btc_24h_volume': btc_stats.get('volume') if btc_stats else None,
            'active_streams': len(self.ws_connections),
            'data_queue_size': self.data_queue.qsize()
        }

    def get_recent_prices(self, symbol: str = "BTCUSDT", limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get recent price data from internal storage or fetch if not available

        Args:
            symbol: Trading pair symbol
            limit: Maximum number of price points to return

        Returns:
            DataFrame with recent price data or None if no data
        """
        try:
            if symbol not in self.price_history or not self.price_history[symbol]:
                # If no real-time data, fetch historical data as fallback
                self.logger.info(f"No real-time data for {symbol}, fetching historical data")
                historical_data = self.get_historical_data(symbol, interval='1m', limit=limit)
                if historical_data is not None:
                    # Convert historical data to the expected format
                    recent_data = []
                    for _, row in historical_data.iterrows():
                        recent_data.append({
                            'timestamp': row.name,
                            'price': row['close'],
                            'volume': row['volume']
                        })
                    self.price_history[symbol] = recent_data
                else:
                    return None

            # Get recent prices
            recent_data = self.price_history[symbol][-limit:]

            # Convert to DataFrame
            df = pd.DataFrame(recent_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error getting recent prices for {symbol}: {e}")
            return None

    def check_connection(self) -> bool:
        """
        Check if the data connector can connect to Binance API

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get a simple price to test connection
            price = self.get_current_price("BTCUSDT")
            return price is not None and price > 0
        except Exception as e:
            self.logger.error(f"Connection check failed: {e}")
            return False