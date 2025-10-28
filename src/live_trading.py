import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import threading
import requests
from typing import Dict, List, Optional, Callable, Tuple
import json
import os
from src.risk_manager import RiskManager
from src.advanced_trading_ai import AdvancedBTCTradingAI

class LiveTradingEngine:
    """
    Live Trading Engine for Small Capital Trading (100-500 USD)
    Integrates real-time data, AI predictions, and risk management
    """

    def __init__(self, ai_model, risk_manager: RiskManager):
        self.ai_model = ai_model
        self.risk_manager = risk_manager
        self.is_running = False
        self.trading_thread = None

        # Trading parameters
        self.trading_pairs = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD']
        self.check_interval = 300  # 5 minutes between checks
        self.max_trades_per_day = 8  # Limit daily trades
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Data storage
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}

        # API endpoints (for demo - would use real exchange APIs)
        self.price_apis = {
            'BTC-USD': 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
            'ETH-USD': 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd',
            'ADA-USD': 'https://api.coingecko.com/api/v3/simple/price?ids=cardano&vs_currencies=usd',
            'SOL-USD': 'https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd'
        }

        # Setup logging
        self._setup_logging()

        # Load trading configuration
        self.load_trading_config()

    def _setup_logging(self):
        """Setup live trading logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'live_trading.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('LiveTradingEngine')

    def load_trading_config(self):
        """Load trading configuration"""
        config_file = 'config/trading_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.check_interval = config.get('check_interval', self.check_interval)
                    self.max_trades_per_day = config.get('max_trades_per_day', self.max_trades_per_day)
                    self.trading_pairs = config.get('trading_pairs', self.trading_pairs)
                self.logger.info("Trading configuration loaded")
            except Exception as e:
                self.logger.error(f"Error loading trading config: {e}")

    def save_trading_config(self):
        """Save trading configuration"""
        os.makedirs('config', exist_ok=True)
        config = {
            'check_interval': self.check_interval,
            'max_trades_per_day': self.max_trades_per_day,
            'trading_pairs': self.trading_pairs,
            'last_updated': datetime.now().isoformat()
        }

        with open('config/trading_config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def start_trading(self):
        """Start the live trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine already running")
            return False

        self.logger.info("Starting live trading engine...")
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()

        self.logger.info("Live trading engine started successfully")
        return True

    def stop_trading(self):
        """Stop the live trading engine"""
        self.logger.info("Stopping live trading engine...")
        self.is_running = False

        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10)

        self.logger.info("Live trading engine stopped")

    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Trading loop started")

        while self.is_running:
            try:
                # Reset daily counters if new day
                self._check_daily_reset()

                # Update market data
                self._update_market_data()

                # Check for trading opportunities
                self._check_trading_opportunities()

                # Check stop losses and take profits
                self._check_risk_management()

                # Log status
                self._log_status()

                # Wait for next check
                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

        self.logger.info("Trading loop ended")

    def _check_daily_reset(self):
        """Reset daily counters if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trade_count = 0
            self.risk_manager.reset_daily_risk()
            self.last_reset_date = current_date
            self.logger.info(f"Daily reset completed for {current_date}")

    def _update_market_data(self):
        """Update current market prices and maintain price history"""
        for pair in self.trading_pairs:
            try:
                price = self._get_current_price(pair)
                if price:
                    self.current_prices[pair] = price

                    # Maintain price history (last 48 hours of hourly data)
                    if pair not in self.price_history:
                        self.price_history[pair] = pd.DataFrame()

                    # Add new price point
                    new_data = pd.DataFrame({
                        'timestamp': [datetime.now()],
                        'price': [price],
                        'volume': [np.random.randint(1000000, 10000000)]  # Simulated volume
                    })

                    self.price_history[pair] = pd.concat([
                        self.price_history[pair],
                        new_data
                    ]).tail(48)  # Keep last 48 hours

            except Exception as e:
                self.logger.error(f"Error updating {pair} data: {e}")

    def _get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a trading pair"""
        try:
            # In production, use real exchange APIs
            # For demo, simulate price movements
            if hasattr(self, '_last_prices') and pair in self._last_prices:
                last_price = self._last_prices[pair]
                # Simulate realistic price movement (±2% per 5 minutes)
                change = np.random.normal(0, 0.005)
                new_price = last_price * (1 + change)
                self._last_prices[pair] = new_price
                return new_price
            else:
                # Initialize with realistic prices
                base_prices = {
                    'BTC-USD': 45000,
                    'ETH-USD': 3000,
                    'ADA-USD': 0.50,
                    'SOL-USD': 100
                }
                if not hasattr(self, '_last_prices'):
                    self._last_prices = {}
                self._last_prices[pair] = base_prices.get(pair, 1000)
                return self._last_prices[pair]

        except Exception as e:
            self.logger.error(f"Error getting price for {pair}: {e}")
            return None

    def _check_trading_opportunities(self):
        """Check for trading opportunities using AI model"""
        if self.daily_trade_count >= self.max_trades_per_day:
            return  # Daily trade limit reached

        for pair in self.trading_pairs:
            try:
                if pair not in self.price_history or len(self.price_history[pair]) < 24:
                    continue  # Not enough data

                # Prepare data for AI prediction
                prediction_data = self._prepare_prediction_data(pair)
                if prediction_data is None:
                    continue

                # Get AI prediction
                signal, confidence, predicted_price = self._get_ai_prediction(prediction_data)

                if signal in ['BUY', 'SELL'] and confidence > 0.7:  # High confidence threshold
                    # Validate trade with risk manager
                    current_price = self.current_prices.get(pair, 0)
                    volatility = self._calculate_volatility(pair)

                    validation = self.risk_manager.validate_trade(
                        signal, confidence, current_price, volatility
                    )

                    if validation['approved']:
                        # Execute trade
                        trade_id = f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        success = self.risk_manager.open_position(trade_id, pair, signal, validation)

                        if success:
                            self.daily_trade_count += 1
                            self.logger.info(f"Trade executed: {trade_id} - {signal} {pair} "
                                           f"Size: ${validation['position_size_usd']:.2f}")

                            # Check if we've hit daily profit target
                            portfolio_status = self.risk_manager.get_portfolio_status()
                            if portfolio_status['daily_pnl'] >= self.risk_manager.daily_target:
                                self.logger.info(f"🎯 Daily profit target reached: ${portfolio_status['daily_pnl']:.2f}")
                                # Could implement auto-stop here

            except Exception as e:
                self.logger.error(f"Error checking opportunities for {pair}: {e}")

    def _prepare_prediction_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Prepare data for AI prediction"""
        try:
            if pair not in self.price_history:
                return None

            df = self.price_history[pair].copy()

            # Add basic indicators (simplified version)
            df['SMA_20'] = df['price'].rolling(20).mean()
            df['SMA_50'] = df['price'].rolling(50).mean()
            df['RSI'] = self._calculate_rsi(df['price'])

            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')

            return df

        except Exception as e:
            self.logger.error(f"Error preparing prediction data for {pair}: {e}")
            return None

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _get_ai_prediction(self, data: pd.DataFrame) -> Tuple[str, float, float]:
        """Get prediction from AI model"""
        try:
            # Prepare sequence for model (simplified)
            # In production, this would use the actual model prediction
            current_price = data['price'].iloc[-1]

            # Simple prediction logic for demo (replace with actual model)
            rsi = data['RSI'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]

            # Generate signal based on indicators
            if rsi < 30 and current_price > sma_20 > sma_50:
                signal = 'BUY'
                confidence = 0.8
            elif rsi > 70 and current_price < sma_20 < sma_50:
                signal = 'SELL'
                confidence = 0.8
            else:
                signal = 'HOLD'
                confidence = 0.5

            # Predict price movement
            predicted_change = np.random.normal(0, 0.02)  # ±2% prediction
            predicted_price = current_price * (1 + predicted_change)

            return signal, confidence, predicted_price

        except Exception as e:
            self.logger.error(f"Error getting AI prediction: {e}")
            return 'HOLD', 0.0, 0.0

    def _calculate_volatility(self, pair: str) -> float:
        """Calculate current volatility for risk management"""
        try:
            if pair not in self.price_history:
                return 0.02  # Default volatility

            prices = self.price_history[pair]['price']
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Annualized hourly volatility

            return min(volatility, 0.1)  # Cap at 10%

        except Exception as e:
            return 0.02

    def _check_risk_management(self):
        """Check stop losses and take profits"""
        try:
            triggered_positions = self.risk_manager.check_stop_loss_take_profit(self.current_prices)

            for position in triggered_positions:
                self.logger.info(f"Risk management triggered: {position['trade_id']} - "
                               f"P&L: ${position['pnl']:.2f}")

        except Exception as e:
            self.logger.error(f"Error in risk management check: {e}")

    def _log_status(self):
        """Log current trading status"""
        try:
            portfolio_status = self.risk_manager.get_portfolio_status()

            self.logger.info(f"Status - Capital: ${portfolio_status['current_capital']:.2f} | "
                           f"Daily P&L: ${portfolio_status['daily_pnl']:.2f} | "
                           f"Open Positions: {portfolio_status['open_positions_count']} | "
                           f"Trades Today: {self.daily_trade_count}")

            # Check if target reached
            if portfolio_status['daily_pnl'] >= self.risk_manager.daily_target:
                self.logger.info(f"🎯 DAILY TARGET ACHIEVED: ${portfolio_status['daily_pnl']:.2f}")

        except Exception as e:
            self.logger.error(f"Error logging status: {e}")

    def get_status(self) -> Dict:
        """Get current trading engine status"""
        return {
            'is_running': self.is_running,
            'daily_trades': self.daily_trade_count,
            'max_daily_trades': self.max_trades_per_day,
            'check_interval': self.check_interval,
            'trading_pairs': self.trading_pairs,
            'current_prices': self.current_prices,
            'portfolio_status': self.risk_manager.get_portfolio_status(),
            'performance_metrics': self.risk_manager.get_performance_metrics()
        }

    def emergency_stop(self, reason: str = 'manual'):
        """Emergency stop all trading activities"""
        self.logger.critical(f"EMERGENCY STOP: {reason}")
        self.stop_trading()
        self.risk_manager.emergency_stop(reason)

    def update_config(self, new_config: Dict):
        """Update trading configuration"""
        if 'check_interval' in new_config:
            self.check_interval = new_config['check_interval']
        if 'max_trades_per_day' in new_config:
            self.max_trades_per_day = new_config['max_trades_per_day']
        if 'trading_pairs' in new_config:
            self.trading_pairs = new_config['trading_pairs']

        self.save_trading_config()
        self.logger.info("Trading configuration updated")

class LiveTradingDashboard:
    """
    Real-time dashboard for monitoring live trading performance
    """

    def __init__(self, trading_engine: LiveTradingEngine):
        self.trading_engine = trading_engine

    def display_status(self):
        """Display current trading status"""
        status = self.trading_engine.get_status()

        print("\n" + "="*70)
        print("🚀 LIVE TRADING DASHBOARD - Small Capital Trading (100-500 USD)")
        print("="*70)

        print(f"Status: {'🟢 RUNNING' if status['is_running'] else '🔴 STOPPED'}")
        print(f"Daily Trades: {status['daily_trades']}/{status['max_daily_trades']}")
        print(f"Check Interval: {status['check_interval']} seconds")

        portfolio = status['portfolio_status']
        print("\n💰 PORTFOLIO STATUS:")
        print(f"Current Capital: ${portfolio['current_capital']:.2f}")
        print(f"Daily P&L: ${portfolio['daily_pnl']:.2f}")
        print(f"Open Positions: {portfolio['open_positions_count']}")
        print(f"Exposure Ratio: {portfolio['exposure_ratio']:.3f}")
        print(f"Unrealized P&L: ${portfolio['unrealized_pnl']:.2f}")

        metrics = status['performance_metrics']
        if 'total_trades' in metrics:
            print("\n📊 PERFORMANCE METRICS:")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
            print(f"Avg Win: ${metrics.get('avg_win', 0):.2f}")
            print(f"Avg Loss: ${metrics.get('avg_loss', 0):.2f}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

        print("\n💱 CURRENT PRICES:")
        for pair, price in status['current_prices'].items():
            print(f"{pair}: ${price:.2f}")

        # Check target achievement
        if portfolio['daily_pnl'] >= 3000:
            print("\n🎯 DAILY PROFIT TARGET ACHIEVED! 🎯")
            print(f"Achievement: ${portfolio['daily_pnl']:.2f}")

        print("="*70)

def main():
    """
    Main function to run live trading system
    """
    print("🚀 Advanced BTCUSD Live Trading System - Small Capital Trading (100-500 USD)")
    print("=" * 80)

    # Initialize components
    risk_manager = RiskManager(initial_capital=100000, daily_target=3000)

    # Load or create AI model (simplified for demo)
    ai_model = None  # Would load actual trained model

    # Initialize trading engine
    trading_engine = LiveTradingEngine(ai_model, risk_manager)
    dashboard = LiveTradingDashboard(trading_engine)

    # Display initial status
    dashboard.display_status()

    # Start trading
    print("\n🔄 Starting live trading engine...")
    trading_engine.start_trading()

    try:
        # Monitor trading for demonstration
        for i in range(10):  # Run for 10 cycles (50 minutes)
            time.sleep(5)  # Update every 5 seconds
            dashboard.display_status()

            # Check for target achievement
            status = trading_engine.get_status()
            if status['portfolio_status']['daily_pnl'] >= 3000:
                print("\n🎯 TARGET ACHIEVED! Stopping trading for the day.")
                break

    except KeyboardInterrupt:
        print("\n⏹️  Stopping trading engine...")
    finally:
        trading_engine.stop_trading()
        dashboard.display_status()

        print("\n✅ Live trading session completed!")
        print("📊 Check logs/live_trading.log for detailed trading history")

if __name__ == "__main__":
    main()