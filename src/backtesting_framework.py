import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from data_collector import HistoricalDataCollector
    from real_data_ai_trainer import RealDataAITrainer
    from risk_manager import RiskManager
except ImportError:
    from src.data_collector import HistoricalDataCollector
    from src.real_data_ai_trainer import RealDataAITrainer
    from src.risk_manager import RiskManager

class BacktestingFramework:
    """
    Comprehensive backtesting framework for trading strategies
    Tests AI models and trading strategies on historical data
    """

    def __init__(self, symbol: str = "BTCUSDT", initial_capital: float = 10000):
        self.symbol = symbol
        self.initial_capital = initial_capital

        # Components
        self.data_collector = HistoricalDataCollector()
        self.ai_trainer = RealDataAITrainer(symbol)
        self.risk_manager = RiskManager(initial_capital, daily_target=initial_capital * 0.02)  # 2% daily target

        # Backtest results
        self.backtest_results = {}
        self.performance_metrics = {}

        # Setup logging
        self._setup_logging()

        print("📊 Backtesting framework initialized")

    def _setup_logging(self):
        """Setup backtesting logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'backtesting.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Backtesting')

    def run_ai_model_backtest(self, lookback_days: int = 365, test_days: int = 90) -> Dict:
        """
        Run backtest using AI model predictions

        Args:
            lookback_days: Days of data to use for training
            test_days: Days of data to use for testing

        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info(f"Starting AI model backtest for {self.symbol}")

            # Get historical data
            data = self.data_collector.collect_comprehensive_dataset(self.symbol, lookback_days + test_days)

            if data is None or data.empty:
                print(f"No data available for backtesting - data is None or empty")
                self.logger.error(f"No data available for backtesting - data is None or empty")
                return {'error': 'No data available for backtesting'}

            print(f"Collected {len(data)} data points from {data.index[0]} to {data.index[-1]}")

            # Split data
            split_date = data.index[-1] - timedelta(days=test_days)
            train_data = data[data.index <= split_date]
            test_data = data[data.index > split_date]

            print(f"Split date: {split_date}, Train data: {len(train_data)} points, Test data: {len(test_data)} points")

            self.logger.info(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")

            # Basic validation: ensure we have enough data for training and testing
            if train_data is None or train_data.empty:
                self.logger.error("Training data is empty after split")
                return {'error': 'Training data is empty after split'}

            if test_data is None or test_data.empty:
                self.logger.error("Testing data is empty after split")
                return {'error': 'Testing data is empty after split'}

            # Train AI models on training data
            self.ai_trainer.data_collector.data_cache[self.symbol] = train_data
            training_results = self.ai_trainer.train_ensemble_models(epochs=20, batch_size=64)

            if 'error' in training_results:
                return training_results

            # Run backtest on test data
            backtest_results = self._simulate_trading(test_data)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_results)

            # Combine results
            results = {
                'backtest_type': 'ai_model_backtest',
                'symbol': self.symbol,
                'training_period': f"{train_data.index[0]} to {train_data.index[-1]}",
                'testing_period': f"{test_data.index[0]} to {test_data.index[-1]}",
                'training_results': training_results,
                'backtest_results': backtest_results,
                'performance_metrics': performance_metrics,
                'timestamp': datetime.now()
            }

            # Save results
            self._save_backtest_results(results)

            self.backtest_results = results
            self.logger.info("AI model backtest completed successfully")

            return results

        except Exception as e:
            self.logger.error(f"Error running AI model backtest: {e}")
            return {'error': str(e)}

    def _simulate_trading(self, test_data: pd.DataFrame) -> Dict:
        """
        Simulate trading using AI model predictions

        Args:
            test_data: Test data for backtesting

        Returns:
            Dictionary with trading simulation results
        """
        try:
            # Reset risk manager
            self.risk_manager = RiskManager(self.initial_capital, self.initial_capital * 0.02)

            trades = []
            portfolio_values = [self.initial_capital]

            if test_data is None or test_data.empty:
                self.logger.warning("No test data provided to _simulate_trading; returning empty results")
                return {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'final_portfolio_value': self.initial_capital,
                    'total_return': 0,
                    'trades': []
                }

            timestamps = [test_data.index[0]]

            # Process each time step
            for i in range(self.ai_trainer.sequence_length, len(test_data) - self.ai_trainer.prediction_horizon, 4):  # Every 4 hours
                # Build input window
                try:
                    current_data = test_data.iloc[i-self.ai_trainer.sequence_length:i]
                except Exception:
                    # If indexing fails (not enough history), skip this step
                    continue

                # Get AI prediction
                try:
                    signal_data = self.ai_trainer.predict_trading_signal(current_data)
                except Exception as e:
                    self.logger.error(f"Error generating prediction at index {i}: {e}")
                    continue

                if not signal_data or 'error' in signal_data:
                    continue

                signal = signal_data.get('signal')
                confidence = signal_data.get('confidence', 0)

                try:
                    current_price = float(test_data.iloc[i]['close'])
                    timestamp = test_data.index[i]
                except Exception:
                    continue

                # Check if we should execute trade
                if signal in ['BUY', 'SELL'] and confidence > 0.6:
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        signal, confidence, current_price, test_data.iloc[i]['atr_14']
                    )[0]

                    if position_size > 0:
                        # Simulate trade execution
                        try:
                            future_price = float(test_data.iloc[i+self.ai_trainer.prediction_horizon]['close'])
                        except Exception:
                            # Can't get future price (out of bounds), skip
                            continue

                        price_change = (future_price - current_price) / current_price

                        if signal == 'BUY':
                            pnl = position_size * price_change
                        else:  # SELL
                            pnl = position_size * (-price_change)

                        # Apply fees (0.1%)
                        fees = abs(pnl) * 0.001
                        pnl -= fees

                        # Update portfolio
                        self.risk_manager.current_capital += pnl

                        # Record trade
                        trade = {
                            'timestamp': timestamp,
                            'signal': signal,
                            'confidence': confidence,
                            'entry_price': current_price,
                            'exit_price': future_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'fees': fees,
                            'portfolio_value': self.risk_manager.current_capital
                        }
                        trades.append(trade)

                        portfolio_values.append(self.risk_manager.current_capital)
                        timestamps.append(timestamp)

            # Calculate summary statistics
            if trades:
                pnl_values = [trade['pnl'] for trade in trades]
                winning_trades = [t for t in trades if t['pnl'] > 0]
                losing_trades = [t for t in trades if t['pnl'] <= 0]

                results = {
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(trades),
                    'total_pnl': sum(pnl_values),
                    'average_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                    'average_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                    'largest_win': max(pnl_values) if pnl_values else 0,
                    'largest_loss': min(pnl_values) if pnl_values else 0,
                    'profit_factor': abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else float('inf'),
                    'sharpe_ratio': self._calculate_sharpe_ratio(pnl_values),
                    'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                    'final_portfolio_value': self.risk_manager.current_capital,
                    'total_return': (self.risk_manager.current_capital - self.initial_capital) / self.initial_capital,
                    'trades': trades[:100]  # Store first 100 trades for analysis
                }
            else:
                results = {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'final_portfolio_value': self.initial_capital,
                    'total_return': 0,
                    'trades': []
                }

            return results

        except Exception as e:
            self.logger.error(f"Error simulating trading: {e}")
            return {'error': str(e)}

    def _calculate_sharpe_ratio(self, pnl_values: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not pnl_values or len(pnl_values) < 2:
            return 0.0

        returns = np.array(pnl_values)
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_performance_metrics(self, backtest_results: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if 'error' in backtest_results:
                return {'error': backtest_results['error']}

            total_return = backtest_results.get('total_return', 0)
            win_rate = backtest_results.get('win_rate', 0)
            total_trades = backtest_results.get('total_trades', 0)
            max_drawdown = backtest_results.get('max_drawdown', 0)
            sharpe_ratio = backtest_results.get('sharpe_ratio', 0)

            # Risk-adjusted metrics
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')

            # Kelly Criterion approximation
            if win_rate > 0 and total_trades > 0:
                kelly_percentage = win_rate - ((1 - win_rate) / (backtest_results.get('average_win', 1) / abs(backtest_results.get('average_loss', 1))))
            else:
                kelly_percentage = 0

            # Performance rating
            if total_return > 0.5 and win_rate > 0.6 and max_drawdown < 0.2:
                rating = 'Excellent'
            elif total_return > 0.2 and win_rate > 0.55 and max_drawdown < 0.3:
                rating = 'Good'
            elif total_return > 0 and win_rate > 0.5:
                rating = 'Fair'
            elif total_return > -0.1:
                rating = 'Poor'
            else:
                rating = 'Very Poor'

            metrics = {
                'total_return_pct': total_return * 100,
                'annualized_return_pct': total_return * 100 * 365 / 90,  # Assuming 90-day test
                'win_rate_pct': win_rate * 100,
                'total_trades': total_trades,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'kelly_percentage': kelly_percentage * 100,
                'performance_rating': rating,
                'risk_adjusted_return': total_return / max_drawdown if max_drawdown > 0 else total_return,
                'profit_consistency': win_rate * (1 - max_drawdown),  # Custom metric
                'timestamp': datetime.now()
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}

    def run_benchmark_backtest(self, strategy: str = 'buy_and_hold') -> Dict:
        """
        Run benchmark backtest for comparison

        Args:
            strategy: Benchmark strategy ('buy_and_hold', 'random', etc.)

        Returns:
            Dictionary with benchmark results
        """
        try:
            self.logger.info(f"Running {strategy} benchmark backtest")

            # Get test data (last 90 days)
            data = self.data_collector.collect_comprehensive_dataset(self.symbol, 90)

            if data is None or data.empty:
                return {'error': 'No data available for benchmark'}

            if strategy == 'buy_and_hold':
                # Buy and hold strategy
                initial_price = data.iloc[0]['close']
                final_price = data.iloc[-1]['close']
                total_return = (final_price - initial_price) / initial_price

                results = {
                    'strategy': 'buy_and_hold',
                    'initial_price': initial_price,
                    'final_price': final_price,
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'period_days': len(data)
                }

            elif strategy == 'random':
                # Random trading strategy
                np.random.seed(42)
                capital = self.initial_capital
                trades = []

                for i in range(10, len(data), 24):  # Trade roughly daily
                    if np.random.random() > 0.5:  # 50% chance to trade
                        signal = np.random.choice(['BUY', 'SELL'])
                        entry_price = data.iloc[i]['close']
                        exit_price = data.iloc[i+24]['close'] if i+24 < len(data) else data.iloc[-1]['close']

                        position_size = capital * 0.1  # 10% of capital
                        pnl = position_size * ((exit_price - entry_price) / entry_price) if signal == 'BUY' else position_size * ((entry_price - exit_price) / entry_price)

                        capital += pnl
                        trades.append({
                            'signal': signal,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl
                        })

                results = {
                    'strategy': 'random',
                    'initial_capital': self.initial_capital,
                    'final_capital': capital,
                    'total_return': (capital - self.initial_capital) / self.initial_capital,
                    'total_return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
                    'total_trades': len(trades),
                    'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
                }

            else:
                return {'error': f'Unknown strategy: {strategy}'}

            return results

        except Exception as e:
            self.logger.error(f"Error running benchmark backtest: {e}")
            return {'error': str(e)}

    def compare_strategies(self, ai_results: Dict, benchmark_results: Dict) -> Dict:
        """
        Compare AI strategy with benchmark

        Args:
            ai_results: Results from AI backtest
            benchmark_results: Results from benchmark backtest

        Returns:
            Dictionary with comparison metrics
        """
        try:
            ai_return = ai_results.get('performance_metrics', {}).get('total_return_pct', 0)
            benchmark_return = benchmark_results.get('total_return_pct', 0)

            ai_win_rate = ai_results.get('backtest_results', {}).get('win_rate', 0) * 100
            ai_max_dd = ai_results.get('backtest_results', {}).get('max_drawdown', 0) * 100
            ai_sharpe = ai_results.get('backtest_results', {}).get('sharpe_ratio', 0)

            comparison = {
                'ai_return_pct': ai_return,
                'benchmark_return_pct': benchmark_return,
                'return_difference_pct': ai_return - benchmark_return,
                'ai_vs_benchmark_ratio': ai_return / benchmark_return if benchmark_return != 0 else float('inf'),
                'ai_win_rate_pct': ai_win_rate,
                'ai_max_drawdown_pct': ai_max_dd,
                'ai_sharpe_ratio': ai_sharpe,
                'recommendation': self._generate_recommendation(ai_return, benchmark_return, ai_win_rate, ai_max_dd)
            }

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}")
            return {'error': str(e)}

    def _generate_recommendation(self, ai_return: float, benchmark_return: float,
                               win_rate: float, max_dd: float) -> str:
        """Generate investment recommendation based on metrics"""
        if ai_return > benchmark_return + 10 and win_rate > 55 and max_dd < 25:
            return "Strong Buy - AI significantly outperforms benchmark with good risk control"
        elif ai_return > benchmark_return + 5 and win_rate > 50 and max_dd < 30:
            return "Buy - AI outperforms benchmark with acceptable risk"
        elif ai_return > benchmark_return and win_rate > 45:
            return "Hold - AI shows some edge over benchmark"
        elif ai_return > benchmark_return - 10:
            return "Neutral - AI performance similar to benchmark"
        else:
            return "Avoid - AI underperforms benchmark significantly"

    def _save_backtest_results(self, results: Dict):
        """Save backtest results to file"""
        try:
            os.makedirs("data/backtests", exist_ok=True)

            filename = f"backtest_results_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("data/backtests", filename)

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, default=convert_numpy)

            self.logger.info(f"Saved backtest results to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")

    def generate_backtest_report(self) -> str:
        """Generate comprehensive backtest report"""
        if not self.backtest_results:
            return "No backtest results available"

        results = self.backtest_results
        perf = results.get('performance_metrics', {})

        report = f"""
================================================================================
📊 COMPREHENSIVE BACKTEST REPORT - {self.symbol}
================================================================================

BACKTEST OVERVIEW:
- Symbol: {results['symbol']}
- Training Period: {results['training_period']}
- Testing Period: {results['testing_period']}
- Initial Capital: ${self.initial_capital:,.2f}

PERFORMANCE METRICS:
- Total Return: {perf.get('total_return_pct', 0):.2f}%
- Annualized Return: {perf.get('annualized_return_pct', 0):.2f}%
- Win Rate: {perf.get('win_rate_pct', 0):.1f}%
- Total Trades: {perf.get('total_trades', 0)}
- Max Drawdown: {perf.get('max_drawdown_pct', 0):.2f}%
- Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}
- Calmar Ratio: {perf.get('calmar_ratio', 'N/A')}
- Kelly %: {perf.get('kelly_percentage', 0):.1f}%

AI MODEL PERFORMANCE:
- Models Trained: {results['training_results'].get('models_trained', 0)}
- Best Model: {results['training_results'].get('best_model', 'N/A')}

OVERALL RATING: {perf.get('performance_rating', 'Unknown')}

================================================================================
"""

        return report

    def run_full_backtest_suite(self) -> Dict:
        """
        Run complete backtest suite including AI models and benchmarks

        Returns:
            Dictionary with complete backtest suite results
        """
        try:
            self.logger.info("Starting full backtest suite")

            # Run AI model backtest
            ai_results = self.run_ai_model_backtest()

            if 'error' in ai_results:
                return ai_results

            # Run benchmark backtests
            buy_hold_results = self.run_benchmark_backtest('buy_and_hold')
            random_results = self.run_benchmark_backtest('random')

            # Compare strategies
            buy_hold_comparison = self.compare_strategies(ai_results, buy_hold_results)
            random_comparison = self.compare_strategies(ai_results, random_results)

            # Generate comprehensive report
            report = self.generate_backtest_report()

            suite_results = {
                'ai_backtest': ai_results,
                'benchmarks': {
                    'buy_and_hold': buy_hold_results,
                    'random': random_results
                },
                'comparisons': {
                    'vs_buy_and_hold': buy_hold_comparison,
                    'vs_random': random_comparison
                },
                'report': report,
                'timestamp': datetime.now()
            }

            # Save suite results
            suite_filename = f"backtest_suite_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            suite_filepath = os.path.join("data/backtests", suite_filename)

            with open(suite_filepath, 'w') as f:
                json.dump(suite_results, f, indent=4, default=str)

            self.logger.info(f"Full backtest suite completed. Results saved to {suite_filepath}")

            return suite_results

        except Exception as e:
            self.logger.error(f"Error running full backtest suite: {e}")
            return {'error': str(e)}

    def run_backtest(self, data: pd.DataFrame = None, initial_capital: float = None,
                    commission: float = 0.001) -> Dict:
        """
        Run a comprehensive backtest (main entry point)

        Args:
            data: Historical data for backtesting
            initial_capital: Starting capital
            commission: Trading commission per trade

        Returns:
            Dictionary with backtest results
        """
        try:
            if initial_capital is not None:
                self.initial_capital = initial_capital

            if data is not None:
                # Use provided data for backtesting
                return self.run_backtest_on_data(data, initial_capital or self.initial_capital, commission)
            else:
                # Run full backtest suite
                return self.run_full_backtest_suite()

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}

    def run_backtest_on_data(self, data: pd.DataFrame, initial_capital: float = 10000,
                           commission: float = 0.001) -> Dict:
        """
        Run backtest on provided data with simple strategy

        Args:
            data: DataFrame with OHLCV data and signals
            initial_capital: Starting capital
            commission: Trading commission

        Returns:
            Dictionary with backtest results
        """
        try:
            if 'signal' not in data.columns:
                return {'error': 'Data must contain signal column'}

            # Simulate trading
            capital = initial_capital
            position = 0  # 0 = no position, 1 = long
            trades = []
            portfolio_values = []

            for i, row in data.iterrows():
                signal = row['signal']
                price = row['close']

                # Execute trades based on signal
                if signal == 1 and position == 0:  # Buy signal
                    # Calculate position size
                    position_size = capital / price
                    commission_cost = position_size * price * commission
                    capital -= commission_cost
                    position = position_size
                    trades.append({
                        'type': 'buy',
                        'price': price,
                        'size': position_size,
                        'timestamp': i,
                        'commission': commission_cost
                    })

                elif signal == -1 and position > 0:  # Sell signal
                    # Sell position
                    sale_value = position * price
                    commission_cost = sale_value * commission
                    capital += sale_value - commission_cost
                    trades.append({
                        'type': 'sell',
                        'price': price,
                        'size': position,
                        'timestamp': i,
                        'commission': commission_cost
                    })
                    position = 0

                # Track portfolio value
                portfolio_value = capital + (position * price if position > 0 else 0)
                portfolio_values.append(portfolio_value)

            # Calculate metrics
            total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
            win_trades = len([t for t in trades if t['type'] == 'sell' and t.get('profit', 0) > 0])
            total_trades = len([t for t in trades if t['type'] == 'sell'])

            return {
                'total_return': total_return,
                'final_capital': portfolio_values[-1],
                'total_trades': total_trades,
                'win_rate': win_trades / total_trades if total_trades > 0 else 0,
                'max_drawdown': 0,  # Simplified
                'sharpe_ratio': 0,  # Simplified
                'trades': trades
            }

        except Exception as e:
            self.logger.error(f"Error running backtest on data: {e}")
            return {'error': str(e)}