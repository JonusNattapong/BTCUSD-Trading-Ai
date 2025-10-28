import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import threading
import signal
import sys
import os
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import json

# Import all system components
try:
    from real_time_data import RealTimeDataConnector
    from data_collector import HistoricalDataCollector
    from real_data_ai_trainer import RealDataAITrainer
    from backtesting_framework import BacktestingFramework
    from risk_manager import RiskManager
    from capital_growth_manager import CapitalGrowthManager
except ImportError:
    from src.real_time_data import RealTimeDataConnector
    from src.data_collector import HistoricalDataCollector
    from src.real_data_ai_trainer import RealDataAITrainer
    from src.backtesting_framework import BacktestingFramework
    from src.risk_manager import RiskManager
    from src.capital_growth_manager import CapitalGrowthManager

class LiveTradingSystem:
    """
    Complete live trading system integrating all components
    Production-ready BTCUSD AI trading platform
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the complete live trading system

        Args:
            config: System configuration dictionary
        """
        self.config = config or self._get_default_config()

        # System state
        self.is_running = False
        self.is_trading = False
        self.emergency_stop = False

        # Core components
        self.data_connector = None
        self.data_collector = None
        self.ai_trainer = None
        self.backtester = None
        self.risk_manager = None
        self.capital_manager = None

        # Trading state
        self.current_position = None
        self.position_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.session_start_time = None

        # Performance tracking
        self.performance_log: List[Dict] = []
        self.system_health: Dict = {}

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.threads: List[threading.Thread] = []

        # Setup logging
        self._setup_logging()

        # Initialize system
        self._initialize_system()

        print("🚀 Live trading system initialized")

    def _get_default_config(self) -> Dict:
        """Get default system configuration"""
        return {
            'capital': {
                'initial_amount': 1000.0,
                'growth_target': 'moderate',
                'max_daily_loss': 50.0
            },
            'trading': {
                'symbol': 'BTCUSDT',
                'max_position_size': 0.1,  # Max 10% of capital per trade
                'min_trade_size': 10.0,
                'max_open_positions': 1,
                'trading_hours': {'start': '00:00', 'end': '23:59'}
            },
            'ai': {
                'model_update_frequency': 'daily',
                'prediction_threshold': 0.6,
                'confidence_required': 0.7
            },
            'risk': {
                'max_drawdown': 0.15,
                'var_limit': 0.05,
                'stress_test_frequency': 'daily'
            },
            'monitoring': {
                'health_check_interval': 60,  # seconds
                'performance_log_interval': 300,  # 5 minutes
                'alert_thresholds': {
                    'high_pnl': 100.0,
                    'high_loss': -50.0,
                    'connection_issues': 5
                }
            }
        }

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logs directory
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Main system log
        logging.basicConfig(
            filename=os.path.join(log_dir, 'live_trading.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('LiveTrading')

        # Trading activity log
        self.trade_logger = logging.getLogger('TradingActivity')
        trade_handler = logging.FileHandler(os.path.join(log_dir, 'trading_activity.log'))
        trade_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)

        # Error log
        error_handler = logging.FileHandler(os.path.join(log_dir, 'errors.log'))
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s\n%(exc_info)s'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)

    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")

            # Initialize data connector
            self.data_connector = RealTimeDataConnector()
            self.logger.info("Real-time data connector initialized")

            # Initialize data collector
            self.data_collector = HistoricalDataCollector()
            self.logger.info("Historical data collector initialized")

            # Initialize AI trainer
            self.ai_trainer = RealDataAITrainer()
            self.logger.info("AI trainer initialized")

            # Initialize backtesting framework
            self.backtester = BacktestingFramework()
            self.logger.info("Backtesting framework initialized")

            # Initialize risk manager
            self.risk_manager = RiskManager(
                initial_capital=self.config['capital']['initial_amount']
            )
            self.logger.info("Risk manager initialized")

            # Initialize capital growth manager
            self.capital_manager = CapitalGrowthManager(
                initial_capital=self.config['capital']['initial_amount'],
                growth_target=self.config['capital']['growth_target']
            )
            self.logger.info("Capital growth manager initialized")

            # Load or train initial AI models
            self._initialize_ai_models()

            self.logger.info("All system components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise

    def _initialize_ai_models(self):
        """Initialize and train AI models"""
        try:
            self.logger.info("Initializing AI models...")

            # Collect recent data for training
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data

            data = self.data_collector.collect_historical_data(
                symbol='BTCUSDT',
                start_date=start_date,
                end_date=end_date,
                interval='1h'
            )

            if data is not None and len(data) > 100:
                # Train AI models
                self.ai_trainer.train_models(data)
                self.logger.info("AI models trained successfully")
            else:
                self.logger.warning("Insufficient data for AI training, using pre-trained models")

        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}")

    def start_system(self):
        """Start the live trading system"""
        try:
            if self.is_running:
                self.logger.warning("System is already running")
                return

            self.logger.info("Starting live trading system...")
            self.is_running = True
            self.session_start_time = datetime.now()

            # Start background threads
            self._start_background_threads()

            # Start real-time data streaming
            self.executor.submit(self._start_data_streaming)

            # Start health monitoring
            self.executor.submit(self._health_monitor)

            # Start performance logging
            self.executor.submit(self._performance_logger)

            self.logger.info("Live trading system started successfully")

            # Keep main thread alive
            self._main_loop()

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop_system()

    def stop_system(self):
        """Stop the live trading system"""
        try:
            self.logger.info("Stopping live trading system...")
            self.is_running = False
            self.is_trading = False

            # Stop all threads
            self.executor.shutdown(wait=True)

            # Close all positions
            if self.current_position:
                self._close_position(reason="System shutdown")

            # Save final state
            self._save_system_state()

            self.logger.info("Live trading system stopped")

        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")

    def _start_background_threads(self):
        """Start background monitoring threads"""
        try:
            # AI model update thread
            ai_thread = threading.Thread(target=self._ai_model_updater, daemon=True)
            ai_thread.start()
            self.threads.append(ai_thread)

            # Risk monitoring thread
            risk_thread = threading.Thread(target=self._risk_monitor, daemon=True)
            risk_thread.start()
            self.threads.append(risk_thread)

            # Capital management thread
            capital_thread = threading.Thread(target=self._capital_manager, daemon=True)
            capital_thread.start()
            self.threads.append(capital_thread)

        except Exception as e:
            self.logger.error(f"Failed to start background threads: {e}")

    def _main_loop(self):
        """Main trading loop"""
        try:
            self.logger.info("Entering main trading loop")

            while self.is_running and not self.emergency_stop:
                try:
                    # Check trading hours
                    if not self._is_trading_hours():
                        time.sleep(60)  # Wait 1 minute
                        continue

                    # Get latest market data
                    market_data = self._get_latest_market_data()
                    if market_data is None:
                        time.sleep(5)
                        continue

                    # Generate trading signals
                    signal = self._generate_trading_signal(market_data)

                    if signal and self.is_trading:
                        self._execute_signal(signal, market_data)

                    # Small delay to prevent excessive CPU usage
                    time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in main trading loop: {e}")
                    time.sleep(5)

        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
            self.emergency_stop = True

    def _start_data_streaming(self):
        """Start real-time data streaming"""
        try:
            self.logger.info("Starting real-time data streaming...")

            # Start price streaming
            self.data_connector.start_price_stream(
                symbol=self.config['trading']['symbol'],
                callback=self._handle_price_update
            )

        except Exception as e:
            self.logger.error(f"Failed to start data streaming: {e}")

    def _handle_price_update(self, price_data: Dict):
        """Handle real-time price updates"""
        try:
            # Update risk manager with latest price
            if self.risk_manager:
                self.risk_manager.update_market_price(price_data)

            # Check stop losses and take profits
            if self.current_position:
                self._check_position_exits(price_data)

        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")

    def _get_latest_market_data(self) -> Optional[Dict]:
        """Get latest market data for decision making"""
        try:
            # Get recent price data
            price_data = self.data_connector.get_recent_prices(
                symbol=self.config['trading']['symbol'],
                limit=100
            )

            if price_data is None or len(price_data) < 50:
                return None

            # Calculate technical indicators
            market_data = self.data_collector.calculate_technical_indicators(price_data)

            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def _generate_trading_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal using AI models"""
        try:
            if not self.ai_trainer or not hasattr(self.ai_trainer, 'predict'):
                return None

            # Prepare data for AI prediction
            prediction_data = self.ai_trainer.prepare_prediction_data(market_data)

            if prediction_data is None:
                return None

            # Get AI prediction
            prediction = self.ai_trainer.predict(prediction_data)

            if prediction is None:
                return None

            # Extract signal components
            direction = prediction.get('direction', 'hold')
            confidence = prediction.get('confidence', 0.0)
            strength = prediction.get('strength', 0.0)

            # Check confidence threshold
            if confidence < self.config['ai']['confidence_required']:
                return None

            # Generate signal
            signal = {
                'direction': direction,
                'confidence': confidence,
                'strength': strength,
                'timestamp': datetime.now(),
                'market_data': market_data.tail(1).to_dict('records')[0] if isinstance(market_data, pd.DataFrame) else market_data
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return None

    def _execute_signal(self, signal: Dict, market_data: Dict):
        """Execute trading signal"""
        try:
            direction = signal['direction']

            # Check if we should open a position
            if direction in ['buy', 'sell'] and self.current_position is None:
                self._open_position(signal, market_data)

            # Check if we should close current position
            elif direction == 'hold' and self.current_position:
                # Consider closing if signal strength is low
                if signal.get('strength', 0) < 0.3:
                    self._close_position(reason="Weak signal")

        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")

    def _open_position(self, signal: Dict, market_data: Dict):
        """Open a new trading position"""
        try:
            direction = signal['direction']
            confidence = signal['confidence']

            # Get current price
            current_price = market_data.get('close', market_data.get('price', 0))
            if current_price <= 0:
                return

            # Calculate position size using risk manager
            available_capital = self.capital_manager.current_capital

            position_size = self.risk_manager.calculate_position_size(
                capital=available_capital,
                risk_per_trade=self.config['trading']['max_position_size'],
                current_price=current_price
            )

            # Check minimum trade size
            if position_size < self.config['trading']['min_trade_size']:
                return

            # Create position record
            position = {
                'id': f"pos_{int(time.time())}",
                'direction': direction,
                'entry_price': current_price,
                'quantity': position_size / current_price,
                'size_usd': position_size,
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'open'
            }

            self.current_position = position
            self.position_history.append(position)

            # Log trade
            self.trade_logger.info(f"OPEN {direction.upper()} position: ${position_size:.2f} at ${current_price:.2f}")

            self.logger.info(f"Opened {direction} position: ${position_size:.2f}")

        except Exception as e:
            self.logger.error(f"Error opening position: {e}")

    def _close_position(self, reason: str = "Manual close"):
        """Close current position"""
        try:
            if not self.current_position:
                return

            # Get current price
            current_price = self.data_connector.get_current_price(
                self.config['trading']['symbol']
            )

            if current_price <= 0:
                return

            position = self.current_position
            entry_price = position['entry_price']
            quantity = position['quantity']

            # Calculate P&L
            if position['direction'] == 'buy':
                pnl = (current_price - entry_price) * quantity
            else:  # sell/short
                pnl = (entry_price - current_price) * quantity

            # Update position record
            position.update({
                'exit_price': current_price,
                'exit_timestamp': datetime.now(),
                'pnl': pnl,
                'exit_reason': reason,
                'status': 'closed'
            })

            # Update daily P&L
            self.daily_pnl += pnl

            # Update capital
            self.capital_manager.update_capital(
                self.capital_manager.current_capital + pnl,
                source='trading',
                metadata={
                    'position_id': position['id'],
                    'pnl': pnl,
                    'reason': reason
                }
            )

            # Log trade
            self.trade_logger.info(f"CLOSE position: P&L ${pnl:.2f} ({reason})")

            self.logger.info(f"Closed position with P&L: ${pnl:.2f}")

            # Clear current position
            self.current_position = None

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    def _check_position_exits(self, price_data: Dict):
        """Check if position should be closed based on price updates"""
        try:
            if not self.current_position:
                return

            current_price = price_data.get('price', 0)
            if current_price <= 0:
                return

            position = self.current_position
            entry_price = position['entry_price']

            # Check stop loss and take profit
            exit_reason = None

            if position['direction'] == 'buy':
                # Check stop loss
                stop_loss_price = self.risk_manager.get_stop_loss_price(position)
                if current_price <= stop_loss_price:
                    exit_reason = "Stop loss triggered"

                # Check take profit
                take_profit_price = self.risk_manager.get_take_profit_price(position)
                if current_price >= take_profit_price:
                    exit_reason = "Take profit triggered"

            else:  # sell/short
                # Check stop loss (reverse for shorts)
                stop_loss_price = self.risk_manager.get_stop_loss_price(position)
                if current_price >= stop_loss_price:
                    exit_reason = "Stop loss triggered"

                # Check take profit (reverse for shorts)
                take_profit_price = self.risk_manager.get_take_profit_price(position)
                if current_price <= take_profit_price:
                    exit_reason = "Take profit triggered"

            if exit_reason:
                self._close_position(exit_reason)

        except Exception as e:
            self.logger.error(f"Error checking position exits: {e}")

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        try:
            now = datetime.now().time()
            start_time = datetime.strptime(
                self.config['trading']['trading_hours']['start'], '%H:%M'
            ).time()
            end_time = datetime.strptime(
                self.config['trading']['trading_hours']['end'], '%H:%M'
            ).time()

            return start_time <= now <= end_time

        except Exception:
            return True  # Default to always trading if config error

    def _ai_model_updater(self):
        """Background thread for updating AI models"""
        while self.is_running:
            try:
                # Update models daily
                time.sleep(24 * 60 * 60)  # 24 hours

                if not self.is_running:
                    break

                self.logger.info("Updating AI models...")

                # Collect new data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # Last 30 days

                new_data = self.data_collector.collect_historical_data(
                    symbol='BTCUSDT',
                    start_date=start_date,
                    end_date=end_date,
                    interval='1h'
                )

                if new_data is not None and len(new_data) > 100:
                    # Update AI models with new data
                    self.ai_trainer.update_models(new_data)
                    self.logger.info("AI models updated successfully")

            except Exception as e:
                self.logger.error(f"Error updating AI models: {e}")
                time.sleep(60)  # Wait before retry

    def _risk_monitor(self):
        """Background thread for risk monitoring"""
        while self.is_running:
            try:
                time.sleep(300)  # Check every 5 minutes

                if not self.is_running:
                    break

                # Perform risk assessment
                risk_status = self.risk_manager.assess_portfolio_risk()

                # Check for emergency conditions
                if risk_status.get('emergency_stop', False):
                    self.emergency_stop = True
                    self.logger.critical("EMERGENCY STOP triggered by risk manager")

                # Daily stress test
                if datetime.now().hour == 0:  # Midnight
                    stress_results = self.risk_manager.run_stress_test()
                    if stress_results.get('breach_detected', False):
                        self.logger.warning("Stress test detected risk breaches")

            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")

    def _capital_manager(self):
        """Background thread for capital management"""
        while self.is_running:
            try:
                time.sleep(3600)  # Check every hour

                if not self.is_running:
                    break

                # Check compounding conditions
                if self.capital_manager._should_compound():
                    self.capital_manager.perform_compounding()

                # Check for withdrawals at milestones
                growth_status = self.capital_manager.get_growth_status()
                if growth_status.get('milestones_achieved', 0) > 0:
                    # Logic for milestone withdrawals handled in capital manager

                    pass

            except Exception as e:
                self.logger.error(f"Error in capital management: {e}")

    def _health_monitor(self):
        """Monitor system health"""
        while self.is_running:
            try:
                time.sleep(self.config['monitoring']['health_check_interval'])

                if not self.is_running:
                    break

                # Check data connection
                connection_ok = self.data_connector.check_connection()

                # Check AI models
                ai_ok = self.ai_trainer.check_models_health()

                # Check risk manager
                risk_ok = self.risk_manager.check_health()

                # Update health status
                self.system_health = {
                    'timestamp': datetime.now(),
                    'data_connection': connection_ok,
                    'ai_models': ai_ok,
                    'risk_manager': risk_ok,
                    'overall_health': all([connection_ok, ai_ok, risk_ok])
                }

                # Alert on issues
                if not self.system_health['overall_health']:
                    self.logger.warning(f"System health issues detected: {self.system_health}")

            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")

    def _performance_logger(self):
        """Log performance metrics periodically"""
        while self.is_running:
            try:
                time.sleep(self.config['monitoring']['performance_log_interval'])

                if not self.is_running:
                    break

                # Collect performance metrics
                performance_data = {
                    'timestamp': datetime.now(),
                    'capital': self.capital_manager.current_capital,
                    'daily_pnl': self.daily_pnl,
                    'open_positions': 1 if self.current_position else 0,
                    'total_trades': len(self.position_history),
                    'winning_trades': len([p for p in self.position_history if p.get('pnl', 0) > 0]),
                    'system_health': self.system_health
                }

                self.performance_log.append(performance_data)

                # Keep only recent logs
                if len(self.performance_log) > 1000:
                    self.performance_log = self.performance_log[-1000:]

            except Exception as e:
                self.logger.error(f"Error logging performance: {e}")

    def _save_system_state(self):
        """Save current system state"""
        try:
            state = {
                'timestamp': datetime.now(),
                'session_start': self.session_start_time,
                'current_position': self.current_position,
                'capital': self.capital_manager.current_capital,
                'daily_pnl': self.daily_pnl,
                'position_history_count': len(self.position_history),
                'performance_log_count': len(self.performance_log),
                'system_health': self.system_health
            }

            os.makedirs("state", exist_ok=True)
            with open("state/system_state.json", 'w') as f:
                json.dump(state, f, indent=4, default=str)

        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            return {
                'is_running': self.is_running,
                'is_trading': self.is_trading,
                'emergency_stop': self.emergency_stop,
                'current_position': self.current_position,
                'capital': self.capital_manager.current_capital,
                'daily_pnl': self.daily_pnl,
                'total_trades': len(self.position_history),
                'system_health': self.system_health,
                'growth_status': self.capital_manager.get_growth_status(),
                'session_runtime': str(datetime.now() - self.session_start_time) if self.session_start_time else None
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}

    def enable_trading(self):
        """Enable live trading"""
        if not self.emergency_stop:
            self.is_trading = True
            self.logger.info("Live trading enabled")

    def disable_trading(self):
        """Disable live trading"""
        self.is_trading = False
        self.logger.info("Live trading disabled")

    def emergency_shutdown(self):
        """Emergency system shutdown"""
        self.logger.critical("EMERGENCY SHUTDOWN initiated")
        self.emergency_stop = True
        self.stop_system()

def main():
    """Main entry point for the live trading system"""
    try:
        # Create system instance
        system = LiveTradingSystem()

        # Setup signal handlers
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Shutting down gracefully...")
            system.emergency_shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the system
        print("Starting BTCUSD AI Trading System...")
        print("Press Ctrl+C to stop")

        system.start_system()

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        if 'system' in locals():
            system.emergency_shutdown()
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.critical(f"Fatal system error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()