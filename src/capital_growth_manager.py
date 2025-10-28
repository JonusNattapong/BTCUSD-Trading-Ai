import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple

try:
    from risk_manager import RiskManager
except ImportError:
    from src.risk_manager import RiskManager

class CapitalGrowthManager:
    """
    Advanced capital growth and compounding management system
    Implements sophisticated capital allocation and growth strategies
    """

    def __init__(self, initial_capital: float = 1000, growth_target: str = 'aggressive'):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.growth_target = growth_target  # 'conservative', 'moderate', 'aggressive'

        # Growth tracking
        self.capital_history: List[Dict] = []
        self.performance_metrics: Dict = {}
        self.growth_milestones: List[Dict] = []

        # Compounding settings
        self.compounding_frequency = 'daily'  # 'daily', 'weekly', 'monthly'
        self.reinvestment_percentage = 0.80  # Reinvest 80% of profits
        self.withdrawal_threshold = initial_capital * 2  # Withdraw when doubled

        # Risk-adjusted growth parameters
        self.growth_params = self._set_growth_parameters(growth_target)

        # Performance tracking
        self.start_date = datetime.now()
        self.last_compound_date = datetime.now()

        # Setup logging
        self._setup_logging()

        # Initialize capital tracking
        self._record_capital_snapshot('initialization')

        print("💰 Capital growth manager initialized")

    def _setup_logging(self):
        """Setup capital growth logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'capital_growth.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CapitalGrowth')

    def _set_growth_parameters(self, target: str) -> Dict:
        """Set growth parameters based on target"""
        params = {
            'conservative': {
                'max_daily_risk': 0.02,      # 2% daily risk
                'target_daily_return': 0.005, # 0.5% daily target
                'max_drawdown_limit': 0.10,   # 10% max drawdown
                'reinvestment_rate': 0.70,    # 70% reinvestment
                'growth_acceleration': 1.0    # No acceleration
            },
            'moderate': {
                'max_daily_risk': 0.05,      # 5% daily risk
                'target_daily_return': 0.015, # 1.5% daily target
                'max_drawdown_limit': 0.15,   # 15% max drawdown
                'reinvestment_rate': 0.80,    # 80% reinvestment
                'growth_acceleration': 1.2    # 20% acceleration
            },
            'aggressive': {
                'max_daily_risk': 0.08,      # 8% daily risk
                'target_daily_return': 0.025, # 2.5% daily target
                'max_drawdown_limit': 0.20,   # 20% max drawdown
                'reinvestment_rate': 0.90,    # 90% reinvestment
                'growth_acceleration': 1.5    # 50% acceleration
            }
        }

        return params.get(target, params['moderate'])

    def update_capital(self, new_capital: float, source: str = 'trading',
                       metadata: Dict = None):
        """
        Update current capital with new value

        Args:
            new_capital: New capital amount
            source: Source of capital change ('trading', 'deposit', 'withdrawal')
            metadata: Additional metadata about the update
        """
        try:
            old_capital = self.current_capital
            capital_change = new_capital - old_capital
            change_percentage = (capital_change / old_capital) * 100 if old_capital > 0 else 0

            self.current_capital = new_capital

            # Record capital snapshot
            snapshot = {
                'timestamp': datetime.now(),
                'capital': new_capital,
                'change_amount': capital_change,
                'change_percentage': change_percentage,
                'source': source,
                'metadata': metadata or {}
            }

            self.capital_history.append(snapshot)
            self._record_capital_snapshot(source, metadata)

            # Check for growth milestones
            self._check_growth_milestones()

            # Auto-compound if conditions met
            if self._should_compound():
                self.perform_compounding()

            self.logger.info(f"Capital updated: ${old_capital:.2f} -> ${new_capital:.2f} ({change_percentage:.2f}%) from {source}")

        except Exception as e:
            self.logger.error(f"Error updating capital: {e}")

    def perform_compounding(self):
        """
        Perform capital compounding based on growth strategy
        """
        try:
            # Calculate compounding amount
            available_profit = self.current_capital - self.initial_capital

            if available_profit <= 0:
                return  # No profits to compound

            # Calculate reinvestment amount
            reinvestment_amount = available_profit * self.growth_params['reinvestment_rate']

            # Apply growth acceleration
            accelerated_reinvestment = reinvestment_amount * self.growth_params['growth_acceleration']

            # Cap reinvestment to prevent over-leveraging
            max_reinvestment = self.current_capital * 0.50  # Max 50% of current capital
            final_reinvestment = min(accelerated_reinvestment, max_reinvestment)

            # Update compounding tracking
            self.last_compound_date = datetime.now()

            compounding_record = {
                'timestamp': datetime.now(),
                'available_profit': available_profit,
                'reinvestment_amount': final_reinvestment,
                'reinvestment_rate': self.growth_params['reinvestment_rate'],
                'growth_acceleration': self.growth_params['growth_acceleration'],
                'capital_before_compound': self.current_capital
            }

            # The actual capital increase would come from trading profits
            # This method just tracks the compounding strategy

            self.logger.info(f"Compounding performed: ${final_reinvestment:.2f} reinvested")

            return compounding_record

        except Exception as e:
            self.logger.error(f"Error performing compounding: {e}")
            return None

    def _should_compound(self) -> bool:
        """Determine if compounding should be performed"""
        try:
            # Check time-based compounding
            time_since_last_compound = datetime.now() - self.last_compound_date

            frequency_days = {
                'daily': 1,
                'weekly': 7,
                'monthly': 30
            }

            days_threshold = frequency_days.get(self.compounding_frequency, 7)

            if time_since_last_compound.days < days_threshold:
                return False

            # Check profit threshold
            min_profit_threshold = self.initial_capital * 0.05  # 5% profit threshold
            current_profit = self.current_capital - self.initial_capital

            if current_profit < min_profit_threshold:
                return False

            # Check drawdown - don't compound during large drawdowns
            recent_drawdown = self._calculate_recent_drawdown()
            if recent_drawdown > self.growth_params['max_drawdown_limit']:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking compound conditions: {e}")
            return False

    def _calculate_recent_drawdown(self, lookback_days: int = 30) -> float:
        """Calculate recent maximum drawdown"""
        try:
            if len(self.capital_history) < 2:
                return 0.0

            # Get recent capital values
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_history = [h for h in self.capital_history if h['timestamp'] > cutoff_date]

            if len(recent_history) < 2:
                return 0.0

            capitals = [h['capital'] for h in recent_history]
            peak = max(capitals)
            current = capitals[-1]

            drawdown = (peak - current) / peak if peak > 0 else 0.0
            return drawdown

        except Exception as e:
            self.logger.error(f"Error calculating recent drawdown: {e}")
            return 0.0

    def _check_growth_milestones(self):
        """Check and record growth milestones"""
        try:
            current_multiple = self.current_capital / self.initial_capital

            milestones = [2, 5, 10, 25, 50, 100]  # Capital multiples

            for milestone in milestones:
                if current_multiple >= milestone:
                    # Check if already recorded
                    existing = [m for m in self.growth_milestones if m['milestone'] == milestone]

                    if not existing:
                        milestone_record = {
                            'timestamp': datetime.now(),
                            'milestone': milestone,
                            'capital': self.current_capital,
                            'multiple': current_multiple,
                            'days_to_achieve': (datetime.now() - self.start_date).days
                        }

                        self.growth_milestones.append(milestone_record)

                        self.logger.info(f"🎉 Growth milestone achieved: {milestone}x capital (${self.current_capital:.2f})")

                        # Consider withdrawal at major milestones
                        if milestone >= 10:  # 10x or more
                            self._consider_withdrawal(milestone)

        except Exception as e:
            self.logger.error(f"Error checking growth milestones: {e}")

    def _consider_withdrawal(self, milestone: float):
        """Consider withdrawing profits at major milestones"""
        try:
            if self.current_capital < self.withdrawal_threshold:
                return

            # Calculate withdrawal amount (25% of profits at major milestones)
            total_profit = self.current_capital - self.initial_capital
            withdrawal_amount = total_profit * 0.25

            # Cap withdrawal to maintain growth
            max_withdrawal = self.current_capital * 0.30  # Max 30% withdrawal
            withdrawal_amount = min(withdrawal_amount, max_withdrawal)

            if withdrawal_amount > 100:  # Minimum withdrawal threshold
                new_capital = self.current_capital - withdrawal_amount

                self.update_capital(new_capital, 'withdrawal', {
                    'milestone': milestone,
                    'withdrawal_amount': withdrawal_amount,
                    'reason': f'Major milestone {milestone}x achieved'
                })

                self.logger.info(f"💰 Profit withdrawal: ${withdrawal_amount:.2f} at {milestone}x milestone")

        except Exception as e:
            self.logger.error(f"Error considering withdrawal: {e}")

    def _record_capital_snapshot(self, event: str, metadata: Dict = None):
        """Record a capital snapshot for analysis"""
        try:
            snapshot = {
                'timestamp': datetime.now(),
                'capital': self.current_capital,
                'event': event,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
                'days_running': (datetime.now() - self.start_date).days,
                'metadata': metadata or {}
            }

            # Keep only last 1000 snapshots to prevent memory issues
            if len(self.capital_history) > 1000:
                self.capital_history = self.capital_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error recording capital snapshot: {e}")

    def calculate_growth_metrics(self) -> Dict:
        """
        Calculate comprehensive growth and performance metrics

        Returns:
            Dictionary with growth metrics
        """
        try:
            if not self.capital_history:
                return {}

            # Basic metrics
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            days_running = (datetime.now() - self.start_date).days

            # Time-weighted metrics
            if days_running > 0:
                annualized_return = (1 + total_return) ** (365 / days_running) - 1
                daily_return_avg = total_return / days_running
            else:
                annualized_return = 0
                daily_return_avg = 0

            # Risk metrics
            capital_values = [h['capital'] for h in self.capital_history]
            returns = np.diff(capital_values) / capital_values[:-1]

            if len(returns) > 0:
                volatility = np.std(returns) * np.sqrt(365)  # Annualized volatility
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

                # Maximum drawdown
                peak = np.maximum.accumulate(capital_values)
                drawdown = (peak - capital_values) / peak
                max_drawdown = np.max(drawdown)

                # Calmar ratio
                calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                calmar_ratio = 0

            # Growth milestones
            milestones_achieved = len(self.growth_milestones)
            highest_milestone = max([m['milestone'] for m in self.growth_milestones]) if self.growth_milestones else 1

            # Compounding efficiency
            total_compounded = sum(h.get('change_amount', 0) for h in self.capital_history
                                 if h.get('source') == 'compounding')

            metrics = {
                'current_capital': self.current_capital,
                'total_return_pct': total_return * 100,
                'annualized_return_pct': annualized_return * 100,
                'daily_return_avg_pct': daily_return_avg * 100,
                'volatility_pct': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown * 100,
                'calmar_ratio': calmar_ratio,
                'days_running': days_running,
                'milestones_achieved': milestones_achieved,
                'highest_milestone': highest_milestone,
                'total_compounded': total_compounded,
                'growth_target': self.growth_target,
                'compounding_frequency': self.compounding_frequency,
                'reinvestment_rate_pct': self.growth_params['reinvestment_rate'] * 100,
                'growth_acceleration': self.growth_params['growth_acceleration']
            }

            self.performance_metrics = metrics
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating growth metrics: {e}")
            return {}

    def get_growth_projection(self, years: int = 5) -> Dict:
        """
        Project future growth based on current performance

        Args:
            years: Number of years to project

        Returns:
            Dictionary with growth projections
        """
        try:
            metrics = self.calculate_growth_metrics()

            if not metrics:
                return {}

            # Use historical annualized return for projection
            annualized_return = metrics.get('annualized_return_pct', 0) / 100

            # Apply growth target adjustments
            projection_return = annualized_return * self.growth_params['growth_acceleration']

            # Conservative estimate (reduce by 30% for safety)
            conservative_return = projection_return * 0.7

            projections = {}
            current_capital = self.current_capital

            for year in range(1, years + 1):
                # Aggressive projection
                aggressive_capital = current_capital * (1 + projection_return) ** year

                # Conservative projection
                conservative_capital = current_capital * (1 + conservative_return) ** year

                projections[f'year_{year}'] = {
                    'aggressive_capital': aggressive_capital,
                    'conservative_capital': conservative_capital,
                    'aggressive_growth': (aggressive_capital - current_capital) / current_capital,
                    'conservative_growth': (conservative_capital - current_capital) / current_capital
                }

            return {
                'base_capital': current_capital,
                'annualized_return_pct': annualized_return * 100,
                'projection_return_pct': projection_return * 100,
                'projections': projections,
                'assumptions': {
                    'growth_acceleration': self.growth_params['growth_acceleration'],
                    'conservative_reduction': 0.7,
                    'projection_years': years
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating growth projection: {e}")
            return {}

    def optimize_growth_strategy(self, current_performance: Dict) -> Dict:
        """
        Optimize growth strategy based on current performance

        Args:
            current_performance: Current performance metrics

        Returns:
            Dictionary with optimization recommendations
        """
        try:
            recommendations = {
                'current_strategy': self.growth_target,
                'recommended_strategy': self.growth_target,
                'parameter_adjustments': {},
                'rationale': []
            }

            # Analyze performance
            sharpe_ratio = current_performance.get('sharpe_ratio', 0)
            max_drawdown = current_performance.get('max_drawdown_pct', 0) / 100
            annualized_return = current_performance.get('annualized_return_pct', 0) / 100

            # Strategy optimization logic
            if max_drawdown > 0.25:  # High drawdown
                recommendations['recommended_strategy'] = 'conservative'
                recommendations['rationale'].append("High drawdown detected - recommend conservative strategy")
                recommendations['parameter_adjustments']['max_daily_risk'] = 0.02

            elif sharpe_ratio < 0.5:  # Poor risk-adjusted returns
                if annualized_return > 0.5:  # Good returns but high risk
                    recommendations['recommended_strategy'] = 'moderate'
                    recommendations['rationale'].append("Good returns but poor Sharpe ratio - recommend moderate risk")
                else:  # Poor returns overall
                    recommendations['recommended_strategy'] = 'conservative'
                    recommendations['rationale'].append("Poor performance - recommend conservative strategy")

            elif annualized_return > 1.0 and max_drawdown < 0.15:  # Excellent performance
                recommendations['recommended_strategy'] = 'aggressive'
                recommendations['rationale'].append("Excellent risk-adjusted performance - can handle aggressive strategy")

            # Parameter adjustments
            if max_drawdown > self.growth_params['max_drawdown_limit']:
                recommendations['parameter_adjustments']['max_drawdown_limit'] = max_drawdown * 0.8
                recommendations['rationale'].append("Drawdown limit exceeded - reducing limit")

            if annualized_return < self.growth_params['target_daily_return'] * 365:
                recommendations['parameter_adjustments']['reinvestment_rate'] = min(self.growth_params['reinvestment_rate'] * 0.9, 0.95)
                recommendations['rationale'].append("Underperforming target - reducing reinvestment rate")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error optimizing growth strategy: {e}")
            return {}

    def export_growth_report(self, filename: str = None) -> str:
        """
        Export comprehensive growth report

        Args:
            filename: Optional filename for the report

        Returns:
            Path to the exported report
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"growth_report_{timestamp}.json"

            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)
            filepath = os.path.join("reports", filename)

            # Compile comprehensive report
            report = {
                'report_generated': datetime.now(),
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'growth_target': self.growth_target,
                'days_running': (datetime.now() - self.start_date).days,
                'performance_metrics': self.calculate_growth_metrics(),
                'growth_projections': self.get_growth_projection(),
                'growth_milestones': self.growth_milestones,
                'capital_history': self.capital_history[-100:],  # Last 100 entries
                'strategy_parameters': self.growth_params,
                'compounding_settings': {
                    'frequency': self.compounding_frequency,
                    'reinvestment_percentage': self.reinvestment_percentage,
                    'withdrawal_threshold': self.withdrawal_threshold
                }
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=4, default=str)

            self.logger.info(f"Growth report exported to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error exporting growth report: {e}")
            return ""

    def get_growth_status(self) -> Dict:
        """
        Get current growth status summary

        Returns:
            Dictionary with current growth status
        """
        try:
            metrics = self.calculate_growth_metrics()

            status = {
                'current_capital': self.current_capital,
                'total_growth_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
                'days_running': (datetime.now() - self.start_date).days,
                'growth_target': self.growth_target,
                'milestones_achieved': len(self.growth_milestones),
                'last_milestone': self.growth_milestones[-1] if self.growth_milestones else None,
                'performance_rating': self._get_performance_rating(metrics),
                'next_milestone': self._get_next_milestone(),
                'compounding_due': self._should_compound()
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting growth status: {e}")
            return {}

    def _get_performance_rating(self, metrics: Dict) -> str:
        """Get performance rating based on metrics"""
        try:
            sharpe = metrics.get('sharpe_ratio', 0)
            return_pct = metrics.get('annualized_return_pct', 0)
            max_dd = metrics.get('max_drawdown_pct', 0)

            if sharpe > 2.0 and return_pct > 50 and max_dd < 15:
                return 'Outstanding'
            elif sharpe > 1.5 and return_pct > 30 and max_dd < 20:
                return 'Excellent'
            elif sharpe > 1.0 and return_pct > 20 and max_dd < 25:
                return 'Very Good'
            elif sharpe > 0.5 and return_pct > 10:
                return 'Good'
            elif return_pct > 0:
                return 'Fair'
            else:
                return 'Poor'

        except Exception:
            return 'Unknown'

    def _get_next_milestone(self) -> Dict:
        """Get next growth milestone target"""
        try:
            current_multiple = self.current_capital / self.initial_capital
            milestones = [2, 5, 10, 25, 50, 100, 200, 500, 1000]

            for milestone in milestones:
                if current_multiple < milestone:
                    capital_needed = self.initial_capital * milestone
                    return {
                        'milestone': milestone,
                        'capital_needed': capital_needed,
                        'additional_needed': capital_needed - self.current_capital
                    }

            return {'milestone': 'max_achieved', 'capital_needed': None, 'additional_needed': 0}

        except Exception:
            return {'milestone': 'unknown', 'capital_needed': None, 'additional_needed': 0}