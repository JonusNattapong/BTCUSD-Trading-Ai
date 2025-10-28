import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
import traceback

# Import all system components
try:
    from real_time_data import RealTimeDataConnector
    from data_collector import HistoricalDataCollector
    from real_data_ai_trainer import RealDataAITrainer
    from backtesting_framework import BacktestingFramework
    from risk_manager import RiskManager
    from capital_growth_manager import CapitalGrowthManager
    from live_trading_system import LiveTradingSystem
except ImportError:
    # Try absolute imports if relative imports fail
    from src.real_time_data import RealTimeDataConnector
    from src.data_collector import HistoricalDataCollector
    from src.real_data_ai_trainer import RealDataAITrainer
    from src.backtesting_framework import BacktestingFramework
    from src.risk_manager import RiskManager
    from src.capital_growth_manager import CapitalGrowthManager
    from src.live_trading_system import LiveTradingSystem

class SystemValidator:
    """
    Comprehensive system validation and testing suite
    Validates all components work together correctly
    """

    def __init__(self):
        self.test_results: Dict = {}
        self.validation_errors: List[Dict] = []

        # Setup logging
        self._setup_logging()

        print("🔍 System Validator initialized")

    def _setup_logging(self):
        """Setup validation logging"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'system_validation.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemValidator')

    def run_full_validation(self) -> Dict:
        """
        Run complete system validation

        Returns:
            Dictionary with validation results
        """
        print("🚀 Starting full system validation...")

        validation_results = {
            'timestamp': datetime.now(),
            'overall_status': 'unknown',
            'component_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'errors': [],
            'recommendations': []
        }

        try:
            # Test individual components
            print("\n📦 Testing individual components...")
            validation_results['component_tests'] = self._test_components()

            # Test component integration
            print("\n🔗 Testing component integration...")
            validation_results['integration_tests'] = self._test_integration()

            # Test system performance
            print("\n⚡ Testing system performance...")
            validation_results['performance_tests'] = self._test_performance()

            # Determine overall status
            all_passed = all([
                all(result.get('status') == 'passed' for result in validation_results['component_tests'].values()),
                all(result.get('status') == 'passed' for result in validation_results['integration_tests'].values()),
                validation_results['performance_tests'].get('status') == 'passed'
            ])

            validation_results['overall_status'] = 'passed' if all_passed else 'failed'

            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(validation_results)

        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['errors'].append(str(e))

        # Save validation report
        self._save_validation_report(validation_results)

        print(f"\n✅ Validation complete. Overall status: {validation_results['overall_status']}")

        return validation_results

    def _test_components(self) -> Dict:
        """Test individual system components"""
        component_tests = {}

        # Test Real-Time Data Connector
        print("Testing Real-Time Data Connector...")
        component_tests['real_time_data'] = self._test_real_time_data()

        # Test Historical Data Collector
        print("Testing Historical Data Collector...")
        component_tests['data_collector'] = self._test_data_collector()

        # Test AI Trainer
        print("Testing AI Trainer...")
        component_tests['ai_trainer'] = self._test_ai_trainer()

        # Test Backtesting Framework
        print("Testing Backtesting Framework...")
        component_tests['backtester'] = self._test_backtester()

        # Test Risk Manager
        print("Testing Risk Manager...")
        component_tests['risk_manager'] = self._test_risk_manager()

        # Test Capital Growth Manager
        print("Testing Capital Growth Manager...")
        component_tests['capital_manager'] = self._test_capital_manager()

        return component_tests

    def _test_real_time_data(self) -> Dict:
        """Test real-time data connector"""
        try:
            connector = RealTimeDataConnector()

            # Test connection
            connection_ok = connector.check_connection()

            # Test getting current price
            price = connector.get_current_price('BTCUSDT')

            # Test getting recent prices
            prices = connector.get_recent_prices('BTCUSDT', limit=10)

            status = 'passed' if connection_ok and price > 0 and prices is not None else 'failed'

            return {
                'status': status,
                'connection': connection_ok,
                'current_price': price,
                'recent_prices_count': len(prices) if prices is not None else 0,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Real-time data test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_data_collector(self) -> Dict:
        """Test historical data collector"""
        try:
            collector = HistoricalDataCollector()

            # Test data collection
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            data = collector.collect_historical_data(
                symbol='BTCUSDT',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )

            # Test technical indicators
            if data is not None and len(data) > 50:
                data_with_indicators = collector.calculate_technical_indicators(data)
                indicators_calculated = len(data_with_indicators.columns) > len(data.columns)
            else:
                indicators_calculated = False

            status = 'passed' if data is not None and len(data) > 0 and indicators_calculated else 'failed'

            return {
                'status': status,
                'data_points': len(data) if data is not None else 0,
                'indicators_calculated': indicators_calculated,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Data collector test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_ai_trainer(self) -> Dict:
        """Test AI trainer"""
        try:
            trainer = RealDataAITrainer()

            # Test model initialization
            models_initialized = trainer.check_models_health()

            # Test with sample data
            sample_data = pd.DataFrame({
                'open': [50000, 50100, 50200],
                'high': [50500, 50600, 50700],
                'low': [49500, 49600, 49700],
                'close': [50200, 50300, 50400],
                'volume': [100, 110, 120]
            })

            prediction = trainer.predict_trading_signal(sample_data)

            status = 'passed' if models_initialized or (prediction is not None and 'error' not in prediction) else 'failed'

            return {
                'status': status,
                'models_initialized': models_initialized,
                'prediction_generated': prediction is not None and 'error' not in prediction,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"AI trainer test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_backtester(self) -> Dict:
        """Test backtesting framework"""
        try:
            backtester = BacktestingFramework()

            # Test with sample strategy
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
                'open': np.random.uniform(40000, 60000, 100),
                'high': np.random.uniform(40000, 60000, 100),
                'low': np.random.uniform(40000, 60000, 100),
                'close': np.random.uniform(40000, 60000, 100),
                'volume': np.random.uniform(50, 200, 100)
            })

            # Simple moving average crossover strategy
            sample_data['SMA_5'] = sample_data['close'].rolling(5).mean()
            sample_data['SMA_10'] = sample_data['close'].rolling(10).mean()

            signals = []
            for i in range(len(sample_data)):
                if sample_data['SMA_5'].iloc[i] > sample_data['SMA_10'].iloc[i]:
                    signals.append(1)  # Buy
                elif sample_data['SMA_5'].iloc[i] < sample_data['SMA_10'].iloc[i]:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold

            sample_data['signal'] = signals

            results = backtester.run_backtest(
                data=sample_data,
                initial_capital=1000,
                commission=0.001
            )

            status = 'passed' if results and 'total_return' in results else 'failed'

            return {
                'status': status,
                'backtest_completed': results is not None,
                'metrics_calculated': 'total_return' in results if results else False,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Backtester test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_risk_manager(self) -> Dict:
        """Test risk manager"""
        try:
            risk_manager = RiskManager(initial_capital=1000)

            # Test position sizing
            position_result = risk_manager.calculate_position_size(
                signal='BUY',
                confidence=0.8,
                current_price=50000,
                volatility=0.02
            )
            position_size_usd, shares, risk_amount = position_result

            # Test VaR calculation
            sample_returns = np.random.normal(0, 0.02, 100)
            var = risk_manager.calculate_var(sample_returns, confidence_level=0.95)

            # Test health check
            health_ok = risk_manager.check_health()

            status = 'passed' if position_size_usd > 0 and var is not None and health_ok else 'failed'

            return {
                'status': status,
                'position_size_calculated': position_size_usd > 0,
                'var_calculated': var is not None,
                'health_check_passed': health_ok,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Risk manager test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_capital_manager(self) -> Dict:
        """Test capital growth manager"""
        try:
            capital_manager = CapitalGrowthManager(initial_capital=1000)

            # Test capital update
            capital_manager.update_capital(1100, 'test')

            # Test metrics calculation
            metrics = capital_manager.calculate_growth_metrics()

            # Test status
            status_info = capital_manager.get_growth_status()

            status = 'passed' if (capital_manager.current_capital == 1100 and
                                metrics and status_info) else 'failed'

            return {
                'status': status,
                'capital_updated': capital_manager.current_capital == 1100,
                'metrics_calculated': bool(metrics),
                'status_retrieved': bool(status_info),
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Capital manager test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_integration(self) -> Dict:
        """Test component integration"""
        integration_tests = {}

        # Test data flow: Collector -> AI -> Backtester
        print("Testing data flow integration...")
        integration_tests['data_flow'] = self._test_data_flow_integration()

        # Test risk integration with trading
        print("Testing risk management integration...")
        integration_tests['risk_integration'] = self._test_risk_integration()

        # Test capital management integration
        print("Testing capital management integration...")
        integration_tests['capital_integration'] = self._test_capital_integration()

        return integration_tests

    def _test_data_flow_integration(self) -> Dict:
        """Test data flow between collector, AI, and backtester"""
        try:
            # Initialize components
            collector = HistoricalDataCollector()
            ai_trainer = RealDataAITrainer()
            backtester = BacktestingFramework()

            # Collect and preprocess data
            data = collector.collect_comprehensive_dataset('BTCUSDT', 30)

            if data is None or len(data) < 100:
                return {'status': 'failed', 'error': 'Insufficient data collected'}

            # Process with AI - use predict_trading_signal instead of predict
            prediction = ai_trainer.predict_trading_signal(data.tail(50))
            if prediction is None or 'error' in prediction:
                return {'status': 'failed', 'error': 'AI prediction failed'}

            # Test backtesting with AI signals
            # Add simple signals based on AI prediction
            test_data = data.tail(100).copy()
            signals = [prediction.get('direction', 'hold')] * len(test_data)
            test_data['signal'] = signals

            backtest_result = backtester.run_backtest(
                data=test_data,
                initial_capital=1000,
                commission=0.001
            )

            status = 'passed' if backtest_result else 'failed'

            return {
                'status': status,
                'data_collected': len(data),
                'ai_prediction': prediction is not None,
                'backtest_completed': backtest_result is not None,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Data flow integration test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_risk_integration(self) -> Dict:
        """Test risk management integration"""
        try:
            risk_manager = RiskManager(initial_capital=1000)

            # Simulate trading scenario
            current_price = 50000
            capital = 1000

            # Calculate position size
            position_result = risk_manager.calculate_position_size(
                signal='BUY',
                confidence=0.8,
                current_price=current_price,
                volatility=0.02
            )

            # Extract position size from tuple
            position_size_usd, shares, risk_amount = position_result

            # Simulate position
            position = {
                'direction': 'buy',
                'entry_price': current_price,
                'quantity': shares,
                'size_usd': position_size_usd
            }

            # Test risk assessment
            risk_assessment = risk_manager.assess_portfolio_risk()

            # Test stop loss calculation
            stop_loss = risk_manager.get_stop_loss_price(position)

            status = 'passed' if (position_size_usd > 0 and risk_assessment and stop_loss) else 'failed'

            return {
                'status': status,
                'position_size': position_size_usd,
                'risk_assessed': bool(risk_assessment),
                'stop_loss_calculated': stop_loss is not None,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Risk integration test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_capital_integration(self) -> Dict:
        """Test capital management integration"""
        try:
            capital_manager = CapitalGrowthManager(initial_capital=1000)

            # Simulate trading profits
            capital_manager.update_capital(1200, 'trading', {'trade_id': 'test_1'})

            # Test compounding
            compounding_result = capital_manager.perform_compounding()

            # Test metrics
            metrics = capital_manager.calculate_growth_metrics()

            # Test projections
            projections = capital_manager.get_growth_projection(years=1)

            status = 'passed' if (capital_manager.current_capital == 1200 and
                                metrics and projections) else 'failed'

            return {
                'status': status,
                'capital_updated': capital_manager.current_capital == 1200,
                'compounding_tested': compounding_result is not None,
                'metrics_calculated': bool(metrics),
                'projections_generated': bool(projections),
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Capital integration test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_performance(self) -> Dict:
        """Test system performance"""
        try:
            performance_results = {
                'status': 'unknown',
                'response_times': {},
                'memory_usage': {},
                'error_rates': {},
                'recommendations': []
            }

            # Test response times
            print("Testing response times...")

            # Data collection response time
            start_time = time.time()
            collector = HistoricalDataCollector()
            data = collector.collect_historical_data(
                symbol='BTCUSDT',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                interval='1h'
            )
            data_collection_time = time.time() - start_time

            # AI prediction response time
            start_time = time.time()
            ai_trainer = RealDataAITrainer()
            if data is not None and len(data) > 10:
                prediction = ai_trainer.predict(data.tail(10))
            ai_prediction_time = time.time() - start_time

            performance_results['response_times'] = {
                'data_collection': data_collection_time,
                'ai_prediction': ai_prediction_time
            }

            # Check performance thresholds
            data_threshold = 5.0  # 5 seconds max for data collection
            ai_threshold = 2.0    # 2 seconds max for AI prediction

            performance_ok = (data_collection_time < data_threshold and
                            ai_prediction_time < ai_threshold)

            if not performance_ok:
                performance_results['recommendations'].append(
                    "Performance thresholds exceeded - consider optimization"
                )

            performance_results['status'] = 'passed' if performance_ok else 'warning'

            return performance_results

        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Check component failures
        component_tests = validation_results.get('component_tests', {})
        for component, result in component_tests.items():
            if result.get('status') == 'failed':
                recommendations.append(
                    f"Fix {component} component - {result.get('error', 'Unknown error')}"
                )

        # Check integration failures
        integration_tests = validation_results.get('integration_tests', {})
        for test, result in integration_tests.items():
            if result.get('status') == 'failed':
                recommendations.append(
                    f"Fix {test} integration - {result.get('error', 'Unknown error')}"
                )

        # Check performance issues
        performance_tests = validation_results.get('performance_tests', {})
        if performance_tests.get('status') in ['failed', 'warning']:
            recommendations.extend(performance_tests.get('recommendations', []))

        # General recommendations
        if validation_results['overall_status'] == 'passed':
            recommendations.append("System validation passed - ready for production")
            recommendations.append("Monitor system performance in live environment")
            recommendations.append("Set up automated daily health checks")
        else:
            recommendations.append("Address all failed tests before production deployment")
            recommendations.append("Consider additional testing in staging environment")

        return recommendations

    def _save_validation_report(self, results: Dict):
        """Save validation report to file"""
        try:
            os.makedirs("reports", exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"system_validation_report_{timestamp}.json"

            with open(f"reports/{filename}", 'w') as f:
                json.dump(results, f, indent=4, default=str)

            print(f"📄 Validation report saved to reports/{filename}")

        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")

    def run_quick_test(self) -> bool:
        """Run a quick system health check"""
        print("🏃 Running quick system test...")

        try:
            # Test basic imports
            components = [
                RealTimeDataConnector,
                HistoricalDataCollector,
                RealDataAITrainer,
                BacktestingFramework,
                RiskManager,
                CapitalGrowthManager,
                LiveTradingSystem
            ]

            for component in components:
                try:
                    # Try to instantiate
                    if component == LiveTradingSystem:
                        # Skip full instantiation for live system
                        continue
                    instance = component()
                    print(f"✅ {component.__name__} initialized successfully")
                except Exception as e:
                    print(f"❌ {component.__name__} failed: {e}")
                    return False

            print("✅ Quick test passed - all components can be imported and initialized")
            return True

        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False

def main():
    """Main validation entry point"""
    try:
        validator = SystemValidator()

        # Run quick test first
        if not validator.run_quick_test():
            print("❌ Quick test failed - aborting full validation")
            sys.exit(1)

        # Run full validation
        results = validator.run_full_validation()

        # Print summary
        print(f"\n📊 Validation Summary:")
        print(f"Overall Status: {results['overall_status'].upper()}")

        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")

        print(f"Recommendations: {len(results['recommendations'])}")
        for rec in results['recommendations']:
            print(f"  - {rec}")

        # Exit with appropriate code
        if results['overall_status'] == 'passed':
            print("\n🎉 System is ready for production!")
            sys.exit(0)
        else:
            print("\n⚠️  System validation failed - check reports for details")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()