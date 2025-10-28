import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_data_ai_trainer import RealDataAITrainer
from data_collector import HistoricalDataCollector

def validate_feature_distributions():
    """
    Compare feature distributions between backtest (training) and forward test data
    """
    print("🔍 Validating Feature Distributions")
    print("=" * 50)

    try:
        # Initialize components
        ai_trainer = RealDataAITrainer()
        data_collector = HistoricalDataCollector()

        # Get training data (backtest data)
        print("📊 Collecting training data (backtest)...")
        train_data = data_collector.collect_comprehensive_dataset("BTCUSDT", days=30)

        # Get forward test data
        print("📊 Collecting forward test data...")
        forward_data = data_collector.collect_comprehensive_dataset("BTCUSDT", days=7)

        if train_data is None or forward_data is None:
            print("❌ Failed to collect data")
            return

        print(f"Training data shape: {train_data.shape}")
        print(f"Forward test data shape: {forward_data.shape}")

        # Define feature columns to compare
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
            'stoch_k', 'stoch_d', 'atr_14',
            'volume_sma_20', 'volume_ratio',
            'price_range', 'hour', 'day_of_week', 'month'
        ]

        # Filter to available columns
        available_train = [col for col in feature_cols if col in train_data.columns]
        available_forward = [col for col in feature_cols if col in forward_data.columns]

        print(f"\nAvailable features in training: {len(available_train)}")
        print(f"Available features in forward test: {len(available_forward)}")

        # Check for missing features
        missing_in_train = set(feature_cols) - set(available_train)
        missing_in_forward = set(feature_cols) - set(available_forward)

        if missing_in_train:
            print(f"⚠️  Missing in training data: {missing_in_train}")
        if missing_in_forward:
            print(f"⚠️  Missing in forward test data: {missing_in_forward}")

        # Compare distributions for common features
        common_features = list(set(available_train) & set(available_forward))
        print(f"\nCommon features to compare: {len(common_features)}")

        # Statistical comparison
        stats_comparison = []

        for feature in common_features[:10]:  # Compare first 10 features for brevity
            train_values = train_data[feature].dropna()
            forward_values = forward_data[feature].dropna()

            if len(train_values) > 0 and len(forward_values) > 0:
                train_stats = {
                    'mean': train_values.mean(),
                    'std': train_values.std(),
                    'min': train_values.min(),
                    'max': train_values.max(),
                    'count': len(train_values)
                }

                forward_stats = {
                    'mean': forward_values.mean(),
                    'std': forward_values.std(),
                    'min': forward_values.min(),
                    'max': forward_values.max(),
                    'count': len(forward_values)
                }

                # Calculate differences
                mean_diff_pct = abs(train_stats['mean'] - forward_stats['mean']) / abs(train_stats['mean']) * 100
                std_diff_pct = abs(train_stats['std'] - forward_stats['std']) / abs(train_stats['std']) * 100

                stats_comparison.append({
                    'feature': feature,
                    'train_stats': train_stats,
                    'forward_stats': forward_stats,
                    'mean_diff_pct': mean_diff_pct,
                    'std_diff_pct': std_diff_pct
                })

        # Display comparison results
        print("\n📈 Feature Distribution Comparison (Top 10 features):")
        print("-" * 80)
        print(f"{'Feature':<15} {'Train Mean':>12} {'Forward Mean':>12} {'Mean Diff %':>12} {'Std Diff %':>12}")
        print("-" * 80)

        for comp in stats_comparison:
            feature = comp['feature']
            train_mean = comp['train_stats']['mean']
            forward_mean = comp['forward_stats']['mean']
            mean_diff = comp['mean_diff_pct']
            std_diff = comp['std_diff_pct']

            print(f"{feature:<15} {train_mean:>12.4f} {forward_mean:>12.4f} {mean_diff:>12.1f}% {std_diff:>12.1f}%")

        # Overall assessment
        avg_mean_diff = np.mean([comp['mean_diff_pct'] for comp in stats_comparison])
        avg_std_diff = np.mean([comp['std_diff_pct'] for comp in stats_comparison])

        print(f"\n📊 Overall Assessment:")
        print(f"Average mean difference: {avg_mean_diff:.1f}%")
        print(f"Average std difference: {avg_std_diff:.1f}%")

        if avg_mean_diff < 10 and avg_std_diff < 20:
            print("✅ Feature distributions are CONSISTENT between training and forward test")
        elif avg_mean_diff < 25 and avg_std_diff < 40:
            print("⚠️  Feature distributions show MODERATE differences - may affect model performance")
        else:
            print("❌ Feature distributions show SIGNIFICANT differences - model may not perform well")

        # Create visualization
        create_distribution_plot(train_data, forward_data, common_features[:5])

        print("\n✅ Feature Distribution Validation Completed")

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()

def create_distribution_plot(train_data, forward_data, features_to_plot):
    """Create distribution comparison plots"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, feature in enumerate(features_to_plot):
            if i >= 5:  # Limit to 5 plots
                break

            ax = axes[i]

            # Plot distributions
            train_values = train_data[feature].dropna()
            forward_values = forward_data[feature].dropna()

            ax.hist(train_values, alpha=0.7, label='Training (Backtest)', bins=30, density=True)
            ax.hist(forward_values, alpha=0.7, label='Forward Test', bins=30, density=True)

            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(len(features_to_plot), 6):
            fig.delaxes(axes[i])

        plt.tight_layout()

        # Save plot
        os.makedirs('validation_results', exist_ok=True)
        plt.savefig('validation_results/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📊 Distribution plots saved to validation_results/feature_distributions.png")

    except Exception as e:
        print(f"⚠️  Could not create distribution plots: {e}")

if __name__ == '__main__':
    validate_feature_distributions()