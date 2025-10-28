import time
import signal
import sys
from datetime import datetime, timedelta
import pandas as pd
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_data_ai_trainer import RealDataAITrainer
from data_collector import HistoricalDataCollector
from risk_manager import RiskManager
from capital_growth_manager import CapitalGrowthManager

class PaperTradingSystem:
    """
    Paper trading system for safe real-time testing
    """

    def __init__(self, initial_capital=1000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.is_running = False

        # Initialize components
        self.ai_trainer = RealDataAITrainer()
        self.data_collector = HistoricalDataCollector()
        self.risk_manager = RiskManager()
        self.capital_manager = CapitalGrowthManager(initial_capital)

        # Trading parameters
        self.check_interval = 300  # 5 minutes
        self.max_daily_trades = 8
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Position tracking
        self.current_position = None
        self.position_entry_price = None
        self.position_size = 0

        print("🧪 Paper Trading System initialized")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print("Mode: PAPER TRADING (No real money at risk)")

    def start_paper_trading(self, duration_hours=24):
        """Start paper trading for specified duration"""
        print(f"\n🚀 Starting Paper Trading for {duration_hours} hours")
        print("=" * 50)

        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        # Load models
        if not self.ai_trainer.load_trained_models():
            print("❌ No trained models found. Please run training first.")
            return

        print("✅ AI Models loaded successfully")
        print(f"Trading will run until: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to stop early\n")

        try:
            while datetime.now() < end_time and self.is_running:
                self._trading_cycle()
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n🛑 Paper trading stopped by user")

        self._generate_report()

    def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Reset daily counters if new day
            if datetime.now().date() != self.last_reset_date:
                self.daily_trade_count = 0
                self.last_reset_date = datetime.now().date()

            # Skip if max daily trades reached
            if self.daily_trade_count >= self.max_daily_trades:
                return

            # Get recent market data - try multiple sources
            recent_data = self._get_recent_market_data()
            if recent_data is None or len(recent_data) < 24:
                print("⚠️  Insufficient market data, skipping cycle")
                return

            # Get AI prediction
            prediction = self.ai_trainer.predict_trading_signal(recent_data)
            if not prediction or 'error' in prediction:
                print("⚠️  AI prediction failed, skipping cycle")
                return

            signal = prediction.get('signal', 'HOLD')
            confidence = prediction.get('confidence', 0)
            current_price = recent_data['close'].iloc[-1]

            # Debug output
            print(f"📊 Signal: {signal} | Confidence: {confidence:.3f} | Price: ${current_price:,.2f}")

            # Execute trading logic
            self._execute_trade_logic(signal, confidence, current_price)

        except Exception as e:
            print(f"⚠️  Error in trading cycle: {e}")

    def _get_recent_market_data(self):
        """Get recent market data from available sources"""
        try:
            # Try Yahoo Finance first for recent data
            import yfinance as yf
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period="2d", interval="1h")

            if data is not None and not data.empty:
                # Convert to expected format
                data = data.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                data['timestamp'] = data.index
                return data

        except Exception as e:
            print(f"⚠️  Yahoo Finance failed: {e}")

        # Generate mock data for testing when real data is unavailable
        print("📊 Using mock market data for testing")
        return self._generate_mock_market_data()

    def _generate_mock_market_data(self, hours=48):
        """Generate realistic mock BTC market data for testing"""
        import numpy as np

        # Start with realistic BTC price (around $60k-70k as of late 2024)
        base_price = 65000
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')

        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results

        # Create price series with realistic volatility
        price_changes = np.random.normal(0, 0.02, hours)  # 2% daily volatility
        prices = base_price * (1 + np.cumsum(price_changes))

        # Ensure prices stay within reasonable bounds
        prices = np.clip(prices, 30000, 150000)

        # Generate OHLC data
        data = []
        for i, price in enumerate(prices):
            # Add some intrabar volatility
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price * (1 + np.random.normal(0, 0.002))
            volume = np.random.randint(1000000, 10000000)  # Realistic volume

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': max(open_price, high),
                'low': min(open_price, low),
                'close': price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def _execute_trade_logic(self, signal, confidence, current_price):
        """Execute trading decisions"""
        timestamp = datetime.now()

        # BUY signal
        if signal == 'BUY' and self.current_position != 'LONG' and confidence >= 0.55:
            if self.current_position == 'SHORT':
                # Close short position
                pnl = (self.position_entry_price - current_price) * self.position_size
                self._record_trade('CLOSE_SHORT', current_price, pnl, timestamp)

            # Open long position
            position_value = min(self.current_capital * 0.1, self.current_capital)  # Max 10% of capital
            self.position_size = position_value / current_price
            self.current_position = 'LONG'
            self.position_entry_price = current_price

            self._record_trade('BUY', current_price, 0, timestamp)
            self.daily_trade_count += 1

            print(f"🟢 BUY: {self.position_size:.6f} BTC @ ${current_price:,.2f}")

        # SELL signal
        elif signal == 'SELL' and self.current_position != 'SHORT' and confidence >= 0.55:
            if self.current_position == 'LONG':
                # Close long position
                pnl = (current_price - self.position_entry_price) * self.position_size
                self._record_trade('CLOSE_LONG', current_price, pnl, timestamp)

            # Open short position
            position_value = min(self.current_capital * 0.1, self.current_capital)
            self.position_size = position_value / current_price
            self.current_position = 'SHORT'
            self.position_entry_price = current_price

            self._record_trade('SELL', current_price, 0, timestamp)
            self.daily_trade_count += 1

            print(f"🔴 SELL: {self.position_size:.6f} BTC @ ${current_price:,.2f}")

    def _record_trade(self, action, price, pnl, timestamp):
        """Record a trade"""
        self.current_capital += pnl

        trade = {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'pnl': pnl,
            'capital_after': self.current_capital
        }

        self.trades.append(trade)

    def _generate_report(self):
        """Generate trading performance report"""
        print("\n" + "=" * 50)
        print("📊 PAPER TRADING REPORT")
        print("=" * 50)

        if not self.trades:
            print("No trades executed during the session.")
            return

        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Final Capital: ${self.current_capital:,.2f}")
        print(f"Return: ${self.current_capital - self.initial_capital:,.2f} ({((self.current_capital/self.initial_capital)-1)*100:.2f}%)")

        # Save detailed report
        report_data = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades
            },
            'trades': self.trades
        }

        # Save to file
        os.makedirs('reports', exist_ok=True)
        filename = f"paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        import json
        with open(f"reports/{filename}", 'w') as f:
            json.dump(report_data, f, default=str, indent=2)

        print(f"\n📄 Detailed report saved to: reports/{filename}")

def main():
    """Main function for paper trading"""
    if len(sys.argv) > 1:
        try:
            hours = float(sys.argv[1])
        except:
            print("Invalid duration. Using default 24 hours.")
            hours = 24
    else:
        hours = 24

    # Initialize paper trading system
    trader = PaperTradingSystem(initial_capital=1000.0)

    # Start paper trading
    trader.start_paper_trading(duration_hours=hours)

if __name__ == '__main__':
    main()