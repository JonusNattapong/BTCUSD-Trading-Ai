import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
import json
import os

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_data_ai_trainer import RealDataAITrainer
from data_collector import HistoricalDataCollector

class ModelOptimizer:
    """
    Optimize AI model hyperparameters and ensemble weights
    """

    def __init__(self):
        self.ai_trainer = RealDataAITrainer()
        self.data_collector = HistoricalDataCollector()

        # Optimization results
        self.optimization_results = {}

    def optimize_ensemble_weights(self):
        """Optimize weights for ensemble model"""
        print("🎯 Optimizing Ensemble Weights")
        print("=" * 40)

        # Load models
        if not self.ai_trainer.load_trained_models():
            print("❌ No trained models found")
            return

        # Get validation data
        print("📊 Collecting validation data...")
        val_data = self.data_collector.collect_comprehensive_dataset("BTCUSDT", days=14)

        if val_data is None or len(val_data) < 48:
            print("❌ Insufficient validation data")
            return

        # Test different weight combinations
        weight_combinations = self._generate_weight_combinations()

        results = []
        best_weights = None
        best_score = -np.inf

        print(f"Testing {len(weight_combinations)} weight combinations...")

        for i, weights in enumerate(weight_combinations):
            score = self._evaluate_weights(weights, val_data)
            results.append({
                'weights': weights,
                'score': score,
                'lstm_weight': weights[0],
                'gru_weight': weights[1],
                'cnn_lstm_weight': weights[2]
            })

            if score > best_score:
                best_score = score
                best_weights = weights

            if (i + 1) % 10 == 0:
                print(f"  Tested {i + 1}/{len(weight_combinations)} combinations...")

        # Save results
        self.optimization_results['ensemble_weights'] = {
            'results': results,
            'best_weights': best_weights,
            'best_score': best_score,
            'timestamp': datetime.now()
        }

        print("
✅ Ensemble weight optimization completed"        print(f"Best weights: LSTM={best_weights[0]:.2f}, GRU={best_weights[1]:.2f}, CNN-LSTM={best_weights[2]:.2f}")
        print(".4f"
        # Create visualization
        self._plot_weight_optimization(results, best_weights)

        return best_weights

    def _generate_weight_combinations(self):
        """Generate weight combinations to test"""
        weights = []
        step = 0.1

        # Generate combinations where weights sum to 1
        for lstm in np.arange(0, 1.1, step):
            for gru in np.arange(0, 1.1 - lstm, step):
                cnn_lstm = 1.0 - lstm - gru
                if cnn_lstm >= 0:
                    weights.append((round(lstm, 2), round(gru, 2), round(cnn_lstm, 2)))

        return weights

    def _evaluate_weights(self, weights, val_data):
        """Evaluate a weight combination"""
        try:
            # Get individual predictions
            predictions = self.ai_trainer.predict_trading_signal(val_data)
            if not predictions or 'individual_predictions' not in predictions:
                return -np.inf

            individual_preds = predictions['individual_predictions']

            # Calculate weighted ensemble prediction
            ensemble_pred = 0
            for i, (model_name, pred) in enumerate(individual_preds.items()):
                if i < len(weights):
                    ensemble_pred += pred * weights[i]

            # Simple scoring: reward predictions closer to 0.5 (neutral) to avoid overconfidence
            # In practice, you'd use proper validation metrics
            score = 1.0 - abs(ensemble_pred - 0.5)  # Higher score for balanced predictions

            return score

        except Exception as e:
            return -np.inf

    def optimize_prediction_thresholds(self):
        """Optimize prediction confidence thresholds"""
        print("\n📉 Optimizing Prediction Thresholds")
        print("=" * 40)

        # Get validation data
        val_data = self.data_collector.collect_comprehensive_dataset("BTCUSDT", days=14)

        if val_data is None:
            print("❌ No validation data available")
            return

        # Test different threshold combinations
        thresholds = np.arange(0.50, 0.75, 0.05)  # 0.50, 0.55, 0.60, 0.65, 0.70

        results = []

        for buy_threshold in thresholds:
            for sell_threshold in np.arange(0.25, buy_threshold, 0.05):
                # Evaluate threshold combination
                metrics = self._evaluate_thresholds(buy_threshold, sell_threshold, val_data)

                results.append({
                    'buy_threshold': buy_threshold,
                    'sell_threshold': sell_threshold,
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                    'total_pnl': metrics['total_pnl'],
                    'score': metrics['score']
                })

        # Find best combination
        best_result = max(results, key=lambda x: x['score'])

        self.optimization_results['thresholds'] = {
            'results': results,
            'best_thresholds': {
                'buy': best_result['buy_threshold'],
                'sell': best_result['sell_threshold']
            },
            'best_score': best_result['score'],
            'timestamp': datetime.now()
        }

        print("✅ Threshold optimization completed"        print(f"Best thresholds: BUY≥{best_result['buy_threshold']:.2f}, SELL≤{best_result['sell_threshold']:.2f}")
        print(".4f"
        # Create visualization
        self._plot_threshold_optimization(results, best_result)

        return best_result

    def _evaluate_thresholds(self, buy_threshold, sell_threshold, val_data):
        """Evaluate threshold combination"""
        # This is a simplified evaluation - in practice you'd use proper backtesting
        predictions = self.ai_trainer.predict_trading_signal(val_data)

        if not predictions:
            return {'win_rate': 0, 'total_trades': 0, 'total_pnl': 0, 'score': 0}

        confidence = predictions.get('confidence', 0.5)

        # Simulate trades based on thresholds
        trades = 0
        wins = 0

        if confidence >= buy_threshold:
            trades += 1
            # Assume 55% win rate for BUY signals (simplified)
            if np.random.random() < 0.55:
                wins += 1
        elif confidence <= sell_threshold:
            trades += 1
            # Assume 55% win rate for SELL signals (simplified)
            if np.random.random() < 0.55:
                wins += 1

        win_rate = wins / trades if trades > 0 else 0
        total_pnl = (wins * 10) - ((trades - wins) * 8)  # Simplified P&L

        # Score: reward win rate and reasonable trade frequency
        score = win_rate * min(trades, 5) / 5  # Prefer 1-5 trades

        return {
            'win_rate': win_rate,
            'total_trades': trades,
            'total_pnl': total_pnl,
            'score': score
        }

    def _plot_weight_optimization(self, results, best_weights):
        """Plot ensemble weight optimization results"""
        try:
            fig = plt.figure(figsize=(12, 8))

            # 3D scatter plot
            ax = fig.add_subplot(111, projection='3d')

            lstm_weights = [r['lstm_weight'] for r in results]
            gru_weights = [r['gru_weight'] for r in results]
            cnn_weights = [r['cnn_lstm_weight'] for r in results]
            scores = [r['score'] for r in results]

            scatter = ax.scatter(lstm_weights, gru_weights, cnn_weights, c=scores,
                               cmap='viridis', alpha=0.6)

            # Highlight best weights
            ax.scatter([best_weights[0]], [best_weights[1]], [best_weights[2]],
                      color='red', s=100, marker='*', label='Best Weights')

            ax.set_xlabel('LSTM Weight')
            ax.set_ylabel('GRU Weight')
            ax.set_zlabel('CNN-LSTM Weight')
            ax.set_title('Ensemble Weight Optimization')

            plt.colorbar(scatter, label='Score')
            plt.legend()

            # Save plot
            os.makedirs('optimization_results', exist_ok=True)
            plt.savefig('optimization_results/ensemble_weights_optimization.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️  Could not create weight optimization plot: {e}")

    def _plot_threshold_optimization(self, results, best_result):
        """Plot threshold optimization results"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Heatmap of scores
            buy_thresholds = sorted(list(set(r['buy_threshold'] for r in results)))
            sell_thresholds = sorted(list(set(r['sell_threshold'] for r in results)))

            score_matrix = np.zeros((len(buy_thresholds), len(sell_thresholds)))

            for result in results:
                buy_idx = buy_thresholds.index(result['buy_threshold'])
                sell_idx = sell_thresholds.index(result['sell_threshold'])
                score_matrix[buy_idx, sell_idx] = result['score']

            sns.heatmap(score_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                       xticklabels=[f'{x:.2f}' for x in sell_thresholds],
                       yticklabels=[f'{y:.2f}' for y in buy_thresholds],
                       ax=axes[0])

            axes[0].set_title('Threshold Optimization Scores')
            axes[0].set_xlabel('Sell Threshold')
            axes[0].set_ylabel('Buy Threshold')

            # Best result highlight
            best_buy_idx = buy_thresholds.index(best_result['buy_threshold'])
            best_sell_idx = sell_thresholds.index(best_result['sell_threshold'])
            axes[0].add_patch(plt.Rectangle((best_sell_idx, best_buy_idx), 1, 1,
                                          fill=False, edgecolor='red', lw=3))

            # Trade frequency vs win rate
            trade_counts = [r['total_trades'] for r in results]
            win_rates = [r['win_rate'] for r in results]
            scores = [r['score'] for r in results]

            scatter = axes[1].scatter(trade_counts, win_rates, c=scores, cmap='viridis', alpha=0.7)
            axes[1].set_xlabel('Total Trades')
            axes[1].set_ylabel('Win Rate')
            axes[1].set_title('Trade Frequency vs Win Rate')
            axes[1].grid(True, alpha=0.3)

            plt.colorbar(scatter, ax=axes[1], label='Score')

            plt.tight_layout()

            # Save plot
            os.makedirs('optimization_results', exist_ok=True)
            plt.savefig('optimization_results/threshold_optimization.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️  Could not create threshold optimization plot: {e}")

    def save_optimization_results(self):
        """Save optimization results to file"""
        if self.optimization_results:
            os.makedirs('optimization_results', exist_ok=True)
            filename = f"optimization_results/model_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(filename, 'w') as f:
                json.dump(self.optimization_results, f, default=str, indent=2)

            print(f"💾 Optimization results saved to: {filename}")

    def run_full_optimization(self):
        """Run complete model optimization"""
        print("🚀 Starting Full Model Optimization")
        print("=" * 50)

        # Optimize ensemble weights
        best_weights = self.optimize_ensemble_weights()

        # Optimize thresholds
        best_thresholds = self.optimize_prediction_thresholds()

        # Save results
        self.save_optimization_results()

        print("\n✅ Full optimization completed!")
        print("\n📋 Optimization Summary:")
        if best_weights:
            print(f"  • Best Ensemble Weights: LSTM={best_weights[0]:.2f}, GRU={best_weights[1]:.2f}, CNN-LSTM={best_weights[2]:.2f}")
        if best_thresholds:
            print(f"  • Best Thresholds: BUY≥{best_thresholds['buy_threshold']:.2f}, SELL≤{best_thresholds['sell_threshold']:.2f}")

        print("\n📊 Check optimization_results/ for detailed analysis and charts")

def main():
    """Main optimization function"""
    optimizer = ModelOptimizer()
    optimizer.run_full_optimization()

if __name__ == '__main__':
    main()