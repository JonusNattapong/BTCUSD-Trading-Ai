import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import json
import os

class RiskManager:
    """
    Advanced Risk Management System for 3000 Daily Profit Target
    """

    def __init__(self, initial_capital: float = 250, daily_target: float = 25):
        self.initial_capital = initial_capital
        self.daily_target = daily_target
        self.current_capital = initial_capital
        self.daily_risk_limit = 0.10  # 10% max daily risk for small capital
        self.max_single_trade_risk = 0.05  # 5% max per trade for small capital
        self.max_open_positions = 3  # Maximum 3 concurrent positions for small capital
        self.portfolio_heat_limit = 0.25  # 25% max portfolio concentration per asset

        # Risk tracking
        self.daily_pnl = 0
        self.daily_risk_used = 0
        self.open_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []

        # Setup logging
        self._setup_logging()

        # Load risk parameters from file if exists
        self.load_risk_config()

    def _setup_logging(self):
        """Setup risk management logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'risk_management.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('RiskManager')

    def load_risk_config(self):
        """Load risk configuration from file"""
        config_file = 'config/risk_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.daily_risk_limit = config.get('daily_risk_limit', self.daily_risk_limit)
                    self.max_single_trade_risk = config.get('max_single_trade_risk', self.max_single_trade_risk)
                    self.max_open_positions = config.get('max_open_positions', self.max_open_positions)
                    self.portfolio_heat_limit = config.get('portfolio_heat_limit', self.portfolio_heat_limit)
                self.logger.info("Risk configuration loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading risk config: {e}")

    def save_risk_config(self):
        """Save current risk configuration"""
        os.makedirs('config', exist_ok=True)
        config = {
            'daily_risk_limit': self.daily_risk_limit,
            'max_single_trade_risk': self.max_single_trade_risk,
            'max_open_positions': self.max_open_positions,
            'portfolio_heat_limit': self.portfolio_heat_limit,
            'last_updated': datetime.now().isoformat()
        }

        with open('config/risk_config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def calculate_position_size(self, signal: str, confidence: float,
                              current_price: float, volatility: float) -> Tuple[float, float, float]:
        """
        Calculate optimal position size based on risk management rules

        Returns:
            Tuple of (position_size_usd, shares, risk_amount)
        """
        # Base position size (Kelly Criterion inspired)
        portfolio_value = self.current_capital
        base_risk = portfolio_value * self.max_single_trade_risk

        # Adjust for confidence (0.3 to 1.0 multiplier)
        confidence_multiplier = 0.3 + (confidence * 0.7)
        adjusted_risk = base_risk * confidence_multiplier

        # Adjust for volatility (higher volatility = smaller position)
        vol_multiplier = 1 / (1 + volatility * 10)  # Volatility penalty
        adjusted_risk *= vol_multiplier

        # Check daily risk limit
        remaining_daily_risk = self.daily_risk_limit - self.daily_risk_used
        if remaining_daily_risk <= 0:
            self.logger.warning("Daily risk limit reached - no new positions")
            return 0, 0, 0

        adjusted_risk = min(adjusted_risk, remaining_daily_risk * portfolio_value)

        # Calculate shares
        shares = adjusted_risk / (current_price * 0.03)  # Assume 3% max adverse move

        # Convert to USD
        position_size_usd = shares * current_price

        # Check portfolio concentration limit
        if self._check_portfolio_concentration('BTC', position_size_usd):
            # Reduce position size if it exceeds concentration limit
            max_concentration = portfolio_value * self.portfolio_heat_limit
            position_size_usd = min(position_size_usd, max_concentration)
            shares = position_size_usd / current_price

        return position_size_usd, shares, adjusted_risk

    def calculate_position_size_simple(self, capital: float, risk_per_trade: float,
                                     current_price: float) -> float:
        """
        Simple position size calculation for validation

        Args:
            capital: Available capital
            risk_per_trade: Risk per trade as decimal
            current_price: Current asset price

        Returns:
            Position size in USD
        """
        risk_amount = capital * risk_per_trade
        # Assume 2% stop loss for simplicity
        stop_loss_pct = 0.02
        position_size = risk_amount / stop_loss_pct
        return position_size

    def _check_portfolio_concentration(self, asset: str, new_position_size: float) -> bool:
        """Check if new position exceeds portfolio concentration limits"""
        current_exposure = sum(pos['size_usd'] for pos in self.open_positions.values()
                              if pos['asset'] == asset)
        total_exposure = current_exposure + new_position_size
        concentration_ratio = total_exposure / self.current_capital

        return concentration_ratio > self.portfolio_heat_limit

    def calculate_dynamic_position_size(self, signal: str, confidence: float, current_price: float,
                                      volatility: float, market_regime: str = 'normal',
                                      portfolio_heat: float = 0.0) -> Tuple[float, float, float]:
        """
        Advanced dynamic position sizing with multiple factors

        Args:
            signal: Trade signal (BUY/SELL)
            confidence: AI confidence score (0-1)
            current_price: Current asset price
            volatility: Current volatility measure
            market_regime: Market regime ('bull', 'bear', 'sideways', 'high_volatility')
            portfolio_heat: Current portfolio heat (0-1)

        Returns:
            Tuple of (position_size_usd, shares, risk_amount)
        """
        try:
            portfolio_value = self.current_capital

            # Base risk calculation using Kelly Criterion with adjustments
            win_rate = self._calculate_recent_win_rate()
            avg_win = self._calculate_average_win()
            avg_loss = abs(self._calculate_average_loss())

            if avg_loss > 0 and win_rate > 0:
                # Kelly formula: (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50%
            else:
                kelly_fraction = 0.02  # Default 2%

            base_risk = portfolio_value * kelly_fraction

            # Confidence multiplier (0.2 to 2.0)
            confidence_multiplier = 0.2 + (confidence * 1.8)
            adjusted_risk = base_risk * confidence_multiplier

            # Market regime adjustments
            regime_multipliers = {
                'bull': 1.2,      # Increase position in bull markets
                'bear': 0.8,      # Reduce position in bear markets
                'sideways': 1.0,  # Normal sizing
                'high_volatility': 0.6  # Reduce position in high volatility
            }
            regime_multiplier = regime_multipliers.get(market_regime, 1.0)
            adjusted_risk *= regime_multiplier

            # Volatility adjustment (inverse relationship)
            vol_multiplier = 1 / (1 + volatility * 5)  # More conservative volatility adjustment
            adjusted_risk *= vol_multiplier

            # Portfolio heat adjustment (reduce size when portfolio is hot)
            heat_multiplier = 1 / (1 + portfolio_heat * 2)
            adjusted_risk *= heat_multiplier

            # Recent performance adjustment
            performance_multiplier = self._calculate_performance_multiplier()
            adjusted_risk *= performance_multiplier

            # Check daily risk limit
            remaining_daily_risk = self.daily_risk_limit - self.daily_risk_used
            if remaining_daily_risk <= 0:
                return 0, 0, 0

            adjusted_risk = min(adjusted_risk, remaining_daily_risk * portfolio_value)

            # Calculate stop loss distance based on volatility
            stop_loss_pct = min(0.05, volatility * 2)  # Max 5% stop loss
            risk_per_share = current_price * stop_loss_pct

            if risk_per_share <= 0:
                return 0, 0, 0

            # Calculate shares
            shares = adjusted_risk / risk_per_share

            # Convert to USD
            position_size_usd = shares * current_price

            # Final safety checks
            max_position_size = portfolio_value * 0.15  # Max 15% of portfolio per position
            position_size_usd = min(position_size_usd, max_position_size)

            # Recalculate shares
            shares = position_size_usd / current_price

            return position_size_usd, shares, adjusted_risk

        except Exception as e:
            self.logger.error(f"Error calculating dynamic position size: {e}")
            return 0, 0, 0

    def _calculate_recent_win_rate(self, lookback_trades: int = 20) -> float:
        """Calculate recent win rate"""
        if len(self.trade_history) < lookback_trades:
            return 0.5  # Default 50% win rate

        recent_trades = self.trade_history[-lookback_trades:]
        winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)

        return winning_trades / len(recent_trades)

    def _calculate_average_win(self, lookback_trades: int = 20) -> float:
        """Calculate average win amount"""
        if len(self.trade_history) < lookback_trades:
            return 0.03  # Default 3% win

        recent_trades = self.trade_history[-lookback_trades:]
        winning_trades = [trade['pnl'] for trade in recent_trades if trade.get('pnl', 0) > 0]

        return np.mean(winning_trades) if winning_trades else 0.03

    def _calculate_average_loss(self, lookback_trades: int = 20) -> float:
        """Calculate average loss amount"""
        if len(self.trade_history) < lookback_trades:
            return 0.02  # Default 2% loss

        recent_trades = self.trade_history[-lookback_trades:]
        losing_trades = [abs(trade['pnl']) for trade in recent_trades if trade.get('pnl', 0) <= 0]

        return np.mean(losing_trades) if losing_trades else 0.02

    def _calculate_performance_multiplier(self, lookback_days: int = 7) -> float:
        """Calculate performance-based multiplier"""
        if not self.trade_history:
            return 1.0

        # Calculate recent performance
        recent_trades = [trade for trade in self.trade_history
                        if (datetime.now() - trade.get('timestamp', datetime.now())).days <= lookback_days]

        if not recent_trades:
            return 1.0

        recent_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
        recent_return = recent_pnl / self.initial_capital

        # Performance multiplier (0.5 to 1.5)
        if recent_return > 0.05:  # Good recent performance
            return 1.2
        elif recent_return < -0.03:  # Poor recent performance
            return 0.7
        else:
            return 1.0

    def calculate_portfolio_var(self, confidence_level: float = 0.95,
                               lookback_days: int = 30) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio

        Args:
            confidence_level: Confidence level (0.95 = 95%)
            lookback_days: Days to look back for historical data

        Returns:
            Portfolio VaR as percentage
        """
        try:
            if len(self.trade_history) < 10:
                return self.daily_risk_limit  # Default to daily risk limit

            # Get recent PnL data
            recent_pnl = []
            for trade in self.trade_history[-100:]:  # Last 100 trades
                if 'pnl' in trade:
                    # Normalize PnL to percentage of capital at time of trade
                    capital_at_trade = trade.get('portfolio_value_before', self.initial_capital)
                    pnl_pct = trade['pnl'] / capital_at_trade
                    recent_pnl.append(pnl_pct)

            if len(recent_pnl) < 10:
                return self.daily_risk_limit

            # Calculate VaR using historical simulation
            pnl_array = np.array(recent_pnl)
            var_pct = np.percentile(pnl_array, (1 - confidence_level) * 100)

            # VaR should be positive (loss amount)
            var_pct = abs(var_pct)

            # Cap VaR at reasonable levels
            var_pct = min(var_pct, 0.20)  # Max 20% VaR

            return var_pct

        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {e}")
            return self.daily_risk_limit

    def calculate_stress_test_loss(self, scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate potential losses under different stress scenarios

        Args:
            scenarios: Dictionary of scenario names and shock percentages

        Returns:
            Dictionary of scenario losses
        """
        try:
            results = {}

            for scenario_name, shock_pct in scenarios.items():
                # Calculate potential loss based on current positions
                total_exposure = sum(pos.get('size_usd', 0) for pos in self.open_positions.values())
                potential_loss = total_exposure * shock_pct

                # Adjust for diversification (assume 20% correlation reduction)
                diversification_factor = min(0.8, 1 / max(1, len(self.open_positions)))
                adjusted_loss = potential_loss * diversification_factor

                results[scenario_name] = adjusted_loss

            return results

        except Exception as e:
            self.logger.error(f"Error calculating stress test: {e}")
            return {}

    def optimize_portfolio_allocation(self, available_assets: List[str],
                                    risk_tolerance: str = 'moderate') -> Dict[str, float]:
        """
        Optimize portfolio allocation across multiple assets

        Args:
            available_assets: List of available assets
            risk_tolerance: Risk tolerance level ('conservative', 'moderate', 'aggressive')

        Returns:
            Dictionary of asset allocations (symbol: percentage)
        """
        try:
            # Risk tolerance parameters
            risk_params = {
                'conservative': {'max_allocation': 0.20, 'diversification_factor': 0.8},
                'moderate': {'max_allocation': 0.30, 'diversification_factor': 0.9},
                'aggressive': {'max_allocation': 0.40, 'diversification_factor': 1.0}
            }

            params = risk_params.get(risk_tolerance, risk_params['moderate'])

            # Calculate current allocations
            current_allocations = {}
            total_exposure = sum(pos.get('size_usd', 0) for pos in self.open_positions.values())

            for asset in available_assets:
                asset_exposure = sum(pos.get('size_usd', 0) for pos in self.open_positions.values()
                                   if pos.get('asset') == asset)
                current_allocations[asset] = asset_exposure / self.current_capital if self.current_capital > 0 else 0

            # Optimize allocations (simple equal-weight with risk adjustments)
            num_assets = len(available_assets)
            base_allocation = 1.0 / num_assets if num_assets > 0 else 0

            optimized_allocations = {}
            for asset in available_assets:
                # Adjust based on current allocation and risk tolerance
                current = current_allocations.get(asset, 0)
                target = base_allocation * params['diversification_factor']

                # Don't exceed max allocation per asset
                target = min(target, params['max_allocation'])

                optimized_allocations[asset] = target

            # Normalize to ensure sum = 100%
            total_allocation = sum(optimized_allocations.values())
            if total_allocation > 0:
                for asset in optimized_allocations:
                    optimized_allocations[asset] /= total_allocation

            return optimized_allocations

        except Exception as e:
            self.logger.error(f"Error optimizing portfolio allocation: {e}")
            return {asset: 1.0/len(available_assets) for asset in available_assets}

    def get_portfolio_risk_metrics(self) -> Dict:
        """
        Get comprehensive portfolio risk metrics

        Returns:
            Dictionary of risk metrics
        """
        try:
            # Basic metrics
            total_exposure = sum(pos.get('size_usd', 0) for pos in self.open_positions.values())
            portfolio_heat = total_exposure / self.current_capital if self.current_capital > 0 else 0

            # Concentration metrics
            asset_exposure = {}
            for pos in self.open_positions.values():
                asset = pos.get('asset', 'unknown')
                asset_exposure[asset] = asset_exposure.get(asset, 0) + pos.get('size_usd', 0)

            max_concentration = max(asset_exposure.values()) / self.current_capital if asset_exposure else 0

            # Risk metrics
            var_95 = self.calculate_portfolio_var(0.95)

            # Performance metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Recent performance (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_trades = [trade for trade in self.trade_history
                           if trade.get('timestamp', datetime.now()) > thirty_days_ago]

            recent_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
            recent_win_rate = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0) / len(recent_trades) if recent_trades else 0

            metrics = {
                'portfolio_heat': portfolio_heat,
                'total_exposure': total_exposure,
                'max_concentration': max_concentration,
                'open_positions': len(self.open_positions),
                'var_95_pct': var_95 * 100,
                'daily_risk_used_pct': self.daily_risk_used * 100,
                'daily_risk_remaining_pct': (self.daily_risk_limit - self.daily_risk_used) * 100,
                'win_rate_pct': win_rate * 100,
                'total_trades': total_trades,
                'recent_performance_30d': recent_pnl,
                'recent_win_rate_30d_pct': recent_win_rate * 100,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'current_capital': self.current_capital,
                'capital_return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for the portfolio"""
        if len(self.trade_history) < 10:
            return 0.0

        # Get daily returns
        daily_pnls = []
        current_date = None
        daily_pnl = 0

        for trade in sorted(self.trade_history, key=lambda x: x.get('timestamp', datetime.now())):
            trade_date = trade.get('timestamp', datetime.now()).date()

            if current_date is None:
                current_date = trade_date

            if trade_date != current_date:
                if daily_pnl != 0:
                    daily_pnls.append(daily_pnl / self.current_capital)
                daily_pnl = 0
                current_date = trade_date

            daily_pnl += trade.get('pnl', 0)

        # Add final day
        if daily_pnl != 0:
            daily_pnls.append(daily_pnl / self.current_capital)

        if len(daily_pnls) < 5:
            return 0.0

        returns = np.array(daily_pnls)
        excess_returns = returns - risk_free_rate / 365  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        return np.sqrt(365) * np.mean(excess_returns) / np.std(excess_returns)

    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """
        Emergency stop all trading activities

        Args:
            reason: Reason for emergency stop
        """
        try:
            self.logger.warning(f"EMERGENCY STOP activated: {reason}")

            # Close all open positions (simulate market orders)
            closed_positions = []
            total_loss = 0

            for pos_id, position in list(self.open_positions.items()):
                # Simulate closing at current price (assume no slippage for emergency stop)
                close_price = position.get('entry_price', 0)  # Simplified assumption

                if position.get('signal') == 'BUY':
                    pnl = (close_price - position['entry_price']) * position.get('shares', 0)
                else:  # SELL position
                    pnl = (position['entry_price'] - close_price) * position.get('shares', 0)

                # Apply closing fees
                fees = abs(pnl) * 0.001
                pnl -= fees

                # Update capital
                self.current_capital += pnl
                total_loss += pnl

                # Record emergency close
                emergency_trade = {
                    'timestamp': datetime.now(),
                    'type': 'emergency_close',
                    'position_id': pos_id,
                    'pnl': pnl,
                    'fees': fees,
                    'reason': reason,
                    'portfolio_value': self.current_capital
                }

                self.trade_history.append(emergency_trade)
                closed_positions.append(pos_id)

                # Remove position
                del self.open_positions[pos_id]

            self.logger.warning(f"Emergency stop completed. Closed {len(closed_positions)} positions. Total P&L: ${total_loss:.2f}")

            return {
                'positions_closed': len(closed_positions),
                'total_pnl': total_loss,
                'final_capital': self.current_capital,
                'reason': reason
            }

        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
            return {'error': str(e)}

    def update_risk_limits_adaptive(self, market_conditions: Dict):
        """
        Adaptively update risk limits based on market conditions

        Args:
            market_conditions: Dictionary with market condition indicators
        """
        try:
            volatility = market_conditions.get('volatility', 0.5)
            market_regime = market_conditions.get('regime', 'normal')
            portfolio_performance = market_conditions.get('portfolio_return', 0)

            # Base adjustments
            if volatility > 0.8:  # High volatility
                self.daily_risk_limit = min(self.daily_risk_limit, 0.05)  # Reduce to 5%
                self.max_single_trade_risk = min(self.max_single_trade_risk, 0.02)  # Reduce to 2%
            elif volatility < 0.3:  # Low volatility
                self.daily_risk_limit = min(self.daily_risk_limit * 1.2, 0.15)  # Increase up to 15%
                self.max_single_trade_risk = min(self.max_single_trade_risk * 1.2, 0.08)  # Increase up to 8%

            # Market regime adjustments
            if market_regime == 'bear':
                self.daily_risk_limit *= 0.8  # Reduce risk in bear markets
            elif market_regime == 'bull':
                self.daily_risk_limit *= 1.1  # Slightly increase in bull markets

            # Performance-based adjustments
            if portfolio_performance < -0.05:  # Recent losses
                self.daily_risk_limit *= 0.9  # Reduce risk
            elif portfolio_performance > 0.05:  # Recent gains
                self.daily_risk_limit *= 1.05  # Slightly increase risk

            # Ensure limits stay within reasonable bounds
            self.daily_risk_limit = max(0.02, min(self.daily_risk_limit, 0.20))  # 2% to 20%
            self.max_single_trade_risk = max(0.005, min(self.max_single_trade_risk, 0.10))  # 0.5% to 10%

            self.logger.info(f"Risk limits updated - Daily: {self.daily_risk_limit:.3f}, Single Trade: {self.max_single_trade_risk:.3f}")

        except Exception as e:
            self.logger.error(f"Error updating adaptive risk limits: {e}")

    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            returns: List of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Value at Risk as a positive percentage
        """
        try:
            if not returns or len(returns) < 10:
                return 0.05  # Default 5% VaR

            returns_array = np.array(returns)

            # Calculate VaR using historical simulation
            var = np.percentile(returns_array, (1 - confidence_level) * 100)

            # Return as positive value
            return abs(var)

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.05

    def check_health(self) -> bool:
        """
        Check if risk manager is healthy and properly configured

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if basic parameters are set
            if self.current_capital <= 0:
                return False

            # Check if risk limits are reasonable
            if self.daily_risk_limit <= 0 or self.daily_risk_limit > 1:
                return False

            if self.max_single_trade_risk <= 0 or self.max_single_trade_risk > 1:
                return False

            # Check if position tracking is working
            if not isinstance(self.open_positions, dict):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking risk manager health: {e}")
            return False

    def assess_portfolio_risk(self) -> Dict:
        """
        Assess current portfolio risk (alias for get_portfolio_risk_metrics)

        Returns:
            Dictionary of risk assessment results
        """
        return self.get_portfolio_risk_metrics()

    def get_stop_loss_price(self, position: Dict) -> float:
        """
        Calculate stop loss price for a position

        Args:
            position: Position dictionary with entry_price, direction, etc.

        Returns:
            Stop loss price
        """
        try:
            entry_price = position.get('entry_price', 0)
            direction = position.get('direction', 'buy')

            # Default stop loss percentage (2%)
            stop_loss_pct = 0.02

            if direction == 'buy':
                # For long positions, stop loss is below entry
                stop_loss_price = entry_price * (1 - stop_loss_pct)
            else:
                # For short positions, stop loss is above entry
                stop_loss_price = entry_price * (1 + stop_loss_pct)

            return stop_loss_price

        except Exception as e:
            self.logger.error(f"Error calculating stop loss price: {e}")
            return 0