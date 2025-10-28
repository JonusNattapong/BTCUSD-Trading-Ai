import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PerformanceOptimizer:
    """
    Advanced Performance Optimization for 3000 Daily Profit Target
    Includes market regime detection, portfolio optimization, and dynamic strategies
    """

    def __init__(self, target_daily_profit=3000, initial_capital=100000):
        self.target_daily_profit = target_daily_profit
        self.initial_capital = initial_capital
        self.market_regime = 'normal'
        self.regime_history = []
        self.performance_history = []

        # Optimization parameters
        self.min_trades_per_day = 3
        self.max_trades_per_day = 12
        self.confidence_threshold = 0.7
        self.volatility_threshold = 0.03

        # Adaptive parameters
        self.current_risk_multiplier = 1.0
        self.current_confidence_threshold = 0.7
        self.current_position_size_multiplier = 1.0

    def detect_market_regime(self, price_data: pd.DataFrame, volume_data: pd.Series) -> str:
        """
        Detect current market regime using advanced technical analysis

        Returns:
            'bull', 'bear', 'sideways', 'high_volatility', 'low_volatility'
        """
        try:
            # Calculate multiple indicators for regime detection
            returns = price_data.pct_change()

            # Volatility regime
            volatility = returns.rolling(20).std()
            current_volatility = volatility.iloc[-1]

            # Trend regime
            sma_20 = price_data.rolling(20).mean()
            sma_50 = price_data.rolling(50).mean()
            trend_strength = (sma_20 - sma_50) / sma_50

            # Momentum regime
            rsi = self._calculate_rsi(price_data)
            macd, signal = self._calculate_macd(price_data)

            # Volume regime
            volume_sma = volume_data.rolling(20).mean()
            volume_ratio = volume_data / volume_sma

            # Regime classification
            if current_volatility > self.volatility_threshold * 1.5:
                regime = 'high_volatility'
            elif current_volatility < self.volatility_threshold * 0.5:
                regime = 'low_volatility'
            elif trend_strength.iloc[-1] > 0.02 and rsi.iloc[-1] > 60:
                regime = 'bull'
            elif trend_strength.iloc[-1] < -0.02 and rsi.iloc[-1] < 40:
                regime = 'bear'
            else:
                regime = 'sideways'

            # Store regime history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'volatility': current_volatility,
                'trend_strength': trend_strength.iloc[-1],
                'rsi': rsi.iloc[-1],
                'volume_ratio': volume_ratio.iloc[-1]
            })

            # Keep only last 100 regime detections
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]

            self.market_regime = regime
            return regime

        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return 'normal'

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def optimize_portfolio_allocation(self, assets_data: Dict[str, pd.DataFrame],
                                    current_performance: Dict) -> Dict[str, float]:
        """
        Optimize portfolio allocation based on current market conditions and performance

        Returns:
            Dict of asset allocations (0-1 scale)
        """
        try:
            # Calculate asset metrics
            asset_metrics = {}
            for asset, data in assets_data.items():
                returns = data['Close'].pct_change().dropna()

                # Risk-adjusted returns
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()

                # Volatility
                volatility = returns.std() * np.sqrt(252)

                # Recent performance (last 30 days)
                recent_returns = returns.tail(30).mean()

                asset_metrics[asset] = {
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'recent_performance': recent_returns,
                    'correlation_with_btc': self._calculate_correlation(data, assets_data.get('BTC', data))
                }

            # Adjust allocations based on market regime
            base_allocation = 1.0 / len(assets_data)  # Equal weight baseline

            allocations = {}
            for asset, metrics in asset_metrics.items():
                allocation = base_allocation

                # Adjust based on Sharpe ratio (reward good risk-adjusted returns)
                sharpe_bonus = metrics['sharpe_ratio'] * 0.1
                allocation += sharpe_bonus

                # Adjust based on recent performance
                performance_bonus = metrics['recent_performance'] * 10
                allocation += performance_bonus

                # Reduce allocation for high volatility assets in high volatility regime
                if self.market_regime == 'high_volatility' and metrics['volatility'] > 0.5:
                    allocation *= 0.7

                # Increase allocation for low correlation assets (diversification)
                if metrics['correlation_with_btc'] < 0.7:
                    allocation *= 1.2

                # Performance-based adjustment
                if current_performance.get('win_rate', 0.5) > 0.6:
                    # Increase allocation when performing well
                    allocation *= 1.1
                elif current_performance.get('win_rate', 0.5) < 0.4:
                    # Decrease allocation when performing poorly
                    allocation *= 0.8

                allocations[asset] = max(0.05, min(0.4, allocation))  # Clamp between 5% and 40%

            # Normalize to sum to 1
            total_allocation = sum(allocations.values())
            allocations = {asset: alloc / total_allocation for asset, alloc in allocations.items()}

            return allocations

        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
            # Return equal allocation as fallback
            return {asset: 1.0 / len(assets_data) for asset in assets_data.keys()}

    def _calculate_correlation(self, asset1_data: pd.DataFrame, asset2_data: pd.DataFrame) -> float:
        """Calculate correlation between two assets"""
        try:
            returns1 = asset1_data['Close'].pct_change().dropna()
            returns2 = asset2_data['Close'].pct_change().dropna()

            # Align data by date
            common_dates = returns1.index.intersection(returns2.index)
            if len(common_dates) < 10:
                return 0.5  # Default correlation

            correlation = returns1.loc[common_dates].corr(returns2.loc[common_dates])
            return correlation if not np.isnan(correlation) else 0.5

        except:
            return 0.5

    def adapt_trading_parameters(self, performance_metrics: Dict, market_regime: str) -> Dict:
        """
        Adapt trading parameters based on performance and market conditions

        Returns:
            Dict of updated parameters
        """
        try:
            # Get recent performance (last 10 trades)
            recent_win_rate = performance_metrics.get('win_rate', 0.5)
            recent_sharpe = performance_metrics.get('sharpe_ratio', 0)
            recent_max_drawdown = performance_metrics.get('max_drawdown', 0)

            # Base parameter adjustments
            new_params = {
                'confidence_threshold': self.current_confidence_threshold,
                'risk_multiplier': self.current_risk_multiplier,
                'position_size_multiplier': self.current_position_size_multiplier,
                'max_trades_per_day': 8
            }

            # Adjust based on performance
            if recent_win_rate > 0.65:
                # Performing well - increase risk slightly
                new_params['risk_multiplier'] = min(1.5, self.current_risk_multiplier * 1.1)
                new_params['position_size_multiplier'] = min(1.3, self.current_position_size_multiplier * 1.05)
                new_params['confidence_threshold'] = max(0.6, self.current_confidence_threshold * 0.98)
            elif recent_win_rate < 0.45:
                # Performing poorly - decrease risk
                new_params['risk_multiplier'] = max(0.5, self.current_risk_multiplier * 0.9)
                new_params['position_size_multiplier'] = max(0.7, self.current_position_size_multiplier * 0.95)
                new_params['confidence_threshold'] = min(0.8, self.current_confidence_threshold * 1.02)

            # Adjust based on market regime
            if market_regime == 'high_volatility':
                new_params['risk_multiplier'] *= 0.8
                new_params['max_trades_per_day'] = min(6, new_params['max_trades_per_day'])
                new_params['confidence_threshold'] = min(0.85, new_params['confidence_threshold'] * 1.05)
            elif market_regime == 'low_volatility':
                new_params['risk_multiplier'] *= 1.2
                new_params['max_trades_per_day'] = max(10, new_params['max_trades_per_day'])
                new_params['confidence_threshold'] = max(0.65, new_params['confidence_threshold'] * 0.95)
            elif market_regime == 'bull':
                new_params['position_size_multiplier'] *= 1.1
                new_params['max_trades_per_day'] = max(10, new_params['max_trades_per_day'])
            elif market_regime == 'bear':
                new_params['risk_multiplier'] *= 0.9
                new_params['confidence_threshold'] = min(0.8, new_params['confidence_threshold'] * 1.02)

            # Adjust for drawdown protection
            if recent_max_drawdown > 0.1:  # 10% drawdown
                new_params['risk_multiplier'] *= 0.7
                new_params['position_size_multiplier'] *= 0.8
                new_params['max_trades_per_day'] = min(5, new_params['max_trades_per_day'])

            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'win_rate': recent_win_rate,
                'sharpe_ratio': recent_sharpe,
                'max_drawdown': recent_max_drawdown,
                'market_regime': market_regime,
                'parameters': new_params.copy()
            })

            # Keep only last 50 performance records
            if len(self.performance_history) > 50:
                self.performance_history = self.performance_history[-50:]

            # Update current parameters
            self.current_risk_multiplier = new_params['risk_multiplier']
            self.current_confidence_threshold = new_params['confidence_threshold']
            self.current_position_size_multiplier = new_params['position_size_multiplier']

            return new_params

        except Exception as e:
            print(f"Error adapting parameters: {e}")
            return {
                'confidence_threshold': 0.7,
                'risk_multiplier': 1.0,
                'position_size_multiplier': 1.0,
                'max_trades_per_day': 8
            }

    def optimize_entry_exit_strategy(self, historical_data: pd.DataFrame,
                                   current_signal: str, confidence: float) -> Dict:
        """
        Optimize entry and exit points based on historical performance

        Returns:
            Dict with optimized entry/exit parameters
        """
        try:
            # Analyze historical trades to find optimal entry/exit patterns
            returns = historical_data['Close'].pct_change()

            # Find optimal holding periods for different market conditions
            optimal_holding = self._find_optimal_holding_period(historical_data, returns)

            # Calculate optimal stop loss based on volatility
            volatility = returns.rolling(20).std()
            optimal_stop_loss = volatility.iloc[-1] * 2  # 2 standard deviations

            # Calculate optimal take profit based on reward/risk ratio
            optimal_take_profit = optimal_stop_loss * 2.5  # 2.5:1 reward:risk ratio

            # Adjust based on current signal confidence
            if confidence > 0.8:
                # High confidence - tighter stops, higher targets
                optimal_stop_loss *= 0.8
                optimal_take_profit *= 1.2
            elif confidence < 0.6:
                # Low confidence - wider stops, lower targets
                optimal_stop_loss *= 1.2
                optimal_take_profit *= 0.8

            return {
                'optimal_holding_period': optimal_holding,
                'stop_loss_pct': optimal_stop_loss,
                'take_profit_pct': optimal_take_profit,
                'trailing_stop': optimal_stop_loss * 0.8,
                'entry_timing': self._optimize_entry_timing(historical_data, current_signal)
            }

        except Exception as e:
            print(f"Error optimizing strategy: {e}")
            return {
                'optimal_holding_period': 4,  # 4 hours default
                'stop_loss_pct': 0.03,  # 3% default
                'take_profit_pct': 0.075,  # 7.5% default
                'trailing_stop': 0.025,
                'entry_timing': 'immediate'
            }

    def _find_optimal_holding_period(self, data: pd.DataFrame, returns: pd.Series) -> int:
        """Find optimal holding period based on historical performance"""
        try:
            # Test different holding periods (1-24 hours)
            best_period = 4
            best_sharpe = -float('inf')

            for period in range(1, 25):
                # Calculate returns for this holding period
                period_returns = []
                for i in range(period, len(returns), period):
                    period_return = (1 + returns.iloc[i-period:i]).prod() - 1
                    period_returns.append(period_return)

                if period_returns:
                    sharpe = np.mean(period_returns) / np.std(period_returns) if np.std(period_returns) > 0 else 0
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_period = period

            return best_period

        except:
            return 4  # Default 4 hours

    def _optimize_entry_timing(self, data: pd.DataFrame, signal: str) -> str:
        """Optimize entry timing based on technical indicators"""
        try:
            # Simple timing optimization based on RSI and trend
            rsi = self._calculate_rsi(data['Close'])
            current_rsi = rsi.iloc[-1]

            if signal == 'BUY':
                if current_rsi < 35:
                    return 'immediate'  # Oversold condition
                elif current_rsi < 45:
                    return 'wait_for_pullback'
                else:
                    return 'wait_for_better_setup'
            else:  # SELL
                if current_rsi > 65:
                    return 'immediate'  # Overbought condition
                elif current_rsi > 55:
                    return 'wait_for_rally'
                else:
                    return 'wait_for_better_setup'

        except:
            return 'immediate'

    def predict_profit_probability(self, current_conditions: Dict,
                                 historical_performance: pd.DataFrame) -> float:
        """
        Predict probability of achieving daily profit target based on current conditions

        Returns:
            Probability (0-1) of hitting the target
        """
        try:
            # Simple probability model based on historical data
            # In production, this would use a trained ML model

            # Factors affecting probability
            win_rate = current_conditions.get('win_rate', 0.5)
            avg_win = current_conditions.get('avg_win', 100)
            avg_loss = abs(current_conditions.get('avg_loss', 100))
            daily_volatility = current_conditions.get('volatility', 0.02)

            # Required trades to hit target
            target_per_trade = self.target_daily_profit / 8  # Assuming 8 trades per day
            required_win_rate = target_per_trade / (avg_win + avg_loss)

            # Adjust for current market conditions
            if self.market_regime == 'high_volatility':
                required_win_rate *= 1.2
            elif self.market_regime == 'bull':
                required_win_rate *= 0.9

            # Calculate probability
            if required_win_rate > 1:
                probability = 0.0  # Impossible
            else:
                # Simple logistic function
                probability = 1 / (1 + np.exp(-10 * (win_rate - required_win_rate)))

            return min(0.95, max(0.05, probability))

        except Exception as e:
            print(f"Error predicting profit probability: {e}")
            return 0.5

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        try:
            if not self.performance_history:
                return {'error': 'No performance history available'}

            # Calculate key metrics
            recent_performance = self.performance_history[-10:]  # Last 10 records

            avg_win_rate = np.mean([p['win_rate'] for p in recent_performance])
            avg_sharpe = np.mean([p['sharpe_ratio'] for p in recent_performance])
            max_drawdown = max([p['max_drawdown'] for p in recent_performance])

            # Regime distribution
            regime_counts = {}
            for p in self.regime_history[-50:]:  # Last 50 regime detections
                regime = p['regime']
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Parameter evolution
            parameter_trends = {
                'risk_multiplier': [p['parameters']['risk_multiplier'] for p in recent_performance],
                'confidence_threshold': [p['parameters']['confidence_threshold'] for p in recent_performance],
                'position_size_multiplier': [p['parameters']['position_size_multiplier'] for p in recent_performance]
            }

            return {
                'average_win_rate': avg_win_rate,
                'average_sharpe_ratio': avg_sharpe,
                'maximum_drawdown': max_drawdown,
                'market_regime_distribution': regime_counts,
                'current_regime': self.market_regime,
                'parameter_trends': parameter_trends,
                'optimization_recommendations': self._generate_recommendations(avg_win_rate, max_drawdown)
            }

        except Exception as e:
            print(f"Error generating performance report: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, win_rate: float, max_drawdown: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if win_rate < 0.5:
            recommendations.append("Consider increasing confidence threshold to improve win rate")
        if max_drawdown > 0.15:
            recommendations.append("Implement stricter risk management - current drawdown too high")
        if self.market_regime == 'high_volatility':
            recommendations.append("Reduce position sizes in high volatility conditions")
        if len(self.regime_history) > 20:
            # Check if regime detection is working
            unique_regimes = len(set([r['regime'] for r in self.regime_history[-20:]]))
            if unique_regimes < 3:
                recommendations.append("Improve market regime detection - too few regime changes detected")

        return recommendations if recommendations else ["Performance is within acceptable ranges"]

    def plot_optimization_dashboard(self):
        """Create comprehensive optimization dashboard"""
        if not self.performance_history:
            print("No performance history to plot")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Performance Optimization Dashboard - 3000 Daily Profit Target', fontsize=16)

        # Performance over time
        timestamps = [p['timestamp'] for p in self.performance_history]
        win_rates = [p['win_rate'] for p in self.performance_history]
        sharpe_ratios = [p['sharpe_ratio'] for p in self.performance_history]

        axes[0, 0].plot(timestamps, win_rates, label='Win Rate', marker='o')
        axes[0, 0].axhline(y=0.6, color='r', linestyle='--', alpha=0.7, label='Target Win Rate')
        axes[0, 0].set_title('Win Rate Over Time')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].plot(timestamps, sharpe_ratios, label='Sharpe Ratio', color='green', marker='s')
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Target Sharpe')
        axes[0, 1].set_title('Sharpe Ratio Over Time')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Parameter evolution
        if self.performance_history:
            risk_mult = [p['parameters']['risk_multiplier'] for p in self.performance_history]
            conf_thresh = [p['parameters']['confidence_threshold'] for p in self.performance_history]
            pos_size_mult = [p['parameters']['position_size_multiplier'] for p in self.performance_history]

            axes[1, 0].plot(timestamps, risk_mult, label='Risk Multiplier', color='red')
            axes[1, 0].set_title('Risk Multiplier Evolution')
            axes[1, 0].tick_params(axis='x', rotation=45)

            axes[1, 1].plot(timestamps, conf_thresh, label='Confidence Threshold', color='blue')
            axes[1, 1].set_title('Confidence Threshold Evolution')
            axes[1, 1].tick_params(axis='x', rotation=45)

            axes[2, 0].plot(timestamps, pos_size_mult, label='Position Size Multiplier', color='orange')
            axes[2, 0].set_title('Position Size Multiplier Evolution')
            axes[2, 0].tick_params(axis='x', rotation=45)

        # Market regime distribution
        if self.regime_history:
            regimes = [r['regime'] for r in self.regime_history[-50:]]
            regime_counts = pd.Series(regimes).value_counts()

            axes[2, 1].bar(regime_counts.index, regime_counts.values)
            axes[2, 1].set_title('Market Regime Distribution')
            axes[2, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('models/optimization_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Optimization dashboard saved to: models/optimization_dashboard.png")