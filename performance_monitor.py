import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import glob

class PerformanceMonitor:
    """
    Monitor and analyze trading system performance
    """

    def __init__(self):
        self.reports_dir = 'reports'
        self.logs_dir = 'logs'
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def generate_performance_dashboard(self):
        """Generate comprehensive performance dashboard"""
        print("📊 Generating Performance Dashboard")
        print("=" * 50)

        # Load all available reports
        reports = self._load_all_reports()

        if not reports:
            print("No performance reports found. Run some backtests or paper trading first.")
            return

        # Analyze performance metrics
        self._analyze_overall_performance(reports)
        self._analyze_trading_patterns(reports)
        self._create_performance_charts(reports)

        print("✅ Performance dashboard generated")

    def _load_all_reports(self):
        """Load all performance reports"""
        reports = []

        # Load backtest reports
        backtest_files = glob.glob('reports/backtest_*.json')
        for file in backtest_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['type'] = 'backtest'
                    data['file'] = file
                    reports.append(data)
            except:
                pass

        # Load paper trading reports
        paper_files = glob.glob('reports/paper_trading_*.json')
        for file in paper_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['type'] = 'paper_trading'
                    data['file'] = file
                    reports.append(data)
            except:
                pass

        return reports

    def _analyze_overall_performance(self, reports):
        """Analyze overall performance metrics"""
        print("\n📈 Overall Performance Analysis")
        print("-" * 30)

        backtest_reports = [r for r in reports if r['type'] == 'backtest']
        paper_reports = [r for r in reports if r['type'] == 'paper_trading']

        # Backtest performance
        if backtest_reports:
            print("Backtest Results:")
            total_backtest_return = sum(r.get('performance_metrics', {}).get('total_return_pct', 0)
                                      for r in backtest_reports) / len(backtest_reports)
            print(".2f"
            avg_win_rate = sum(r.get('performance_metrics', {}).get('win_rate_pct', 0)
                             for r in backtest_reports) / len(backtest_reports)
            print(".1f"
        # Paper trading performance
        if paper_reports:
            print("\nPaper Trading Results:")
            total_paper_return = sum(r['summary'].get('total_pnl', 0) for r in paper_reports)
            total_paper_trades = sum(r['summary'].get('total_trades', 0) for r in paper_reports)

            if total_paper_trades > 0:
                avg_win_rate_paper = sum(r['summary'].get('win_rate', 0) for r in paper_reports) / len(paper_reports)
                print(".2f"                print(f"Total Trades: {total_paper_trades}")
                print(".1f"
    def _analyze_trading_patterns(self, reports):
        """Analyze trading patterns and behaviors"""
        print("\n🎯 Trading Pattern Analysis")
        print("-" * 30)

        all_trades = []
        for report in reports:
            if 'trades' in report:
                for trade in report['trades']:
                    trade['type'] = report['type']
                    all_trades.append(trade)

        if not all_trades:
            print("No trade data available")
            return

        # Analyze trade timing
        trade_hours = [datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')).hour
                      for t in all_trades if 'timestamp' in t]

        if trade_hours:
            print(f"Most active trading hour: {pd.Series(trade_hours).mode().iloc[0]}:00")

        # Analyze trade sizes
        trade_sizes = [abs(t.get('pnl', 0)) for t in all_trades]
        if trade_sizes:
            avg_trade_size = np.mean(trade_sizes)
            print(".2f"
        # Analyze win/loss patterns
        winning_trades = [t for t in all_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in all_trades if t.get('pnl', 0) < 0]

        print(f"Winning trades: {len(winning_trades)}")
        print(f"Losing trades: {len(losing_trades)}")

        if winning_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            print(".2f"
        if losing_trades:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            print(".2f"
    def _create_performance_charts(self, reports):
        """Create performance visualization charts"""
        print("\n📊 Creating Performance Charts")

        try:
            # Create equity curve chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('BTCUSD AI Trading System - Performance Dashboard', fontsize=16)

            # Equity curve
            ax1 = axes[0, 0]
            self._plot_equity_curve(ax1, reports)
            ax1.set_title('Portfolio Equity Curve')
            ax1.grid(True, alpha=0.3)

            # Trade P&L distribution
            ax2 = axes[0, 1]
            self._plot_pnl_distribution(ax2, reports)
            ax2.set_title('Trade P&L Distribution')
            ax2.grid(True, alpha=0.3)

            # Win rate by hour
            ax3 = axes[1, 0]
            self._plot_hourly_performance(ax3, reports)
            ax3.set_title('Performance by Hour')
            ax3.grid(True, alpha=0.3)

            # Cumulative returns
            ax4 = axes[1, 1]
            self._plot_cumulative_returns(ax4, reports)
            ax4.set_title('Cumulative Returns')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save dashboard
            dashboard_file = f"reports/performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📄 Dashboard saved to: {dashboard_file}")

        except Exception as e:
            print(f"⚠️  Could not create charts: {e}")

    def _plot_equity_curve(self, ax, reports):
        """Plot equity curve"""
        for report in reports:
            if 'trades' in report and report['trades']:
                capital_values = [report.get('summary', {}).get('initial_capital', 1000)]
                timestamps = []

                for trade in report['trades']:
                    if 'capital_after' in trade:
                        capital_values.append(trade['capital_after'])
                        timestamps.append(trade.get('timestamp', ''))

                if len(capital_values) > 1:
                    ax.plot(range(len(capital_values)), capital_values,
                           label=f"{report['type']} ({len(capital_values)-1} trades)",
                           marker='o', markersize=3)

        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()

    def _plot_pnl_distribution(self, ax, reports):
        """Plot P&L distribution"""
        all_pnl = []
        for report in reports:
            if 'trades' in report:
                for trade in report['trades']:
                    if 'pnl' in trade and trade['pnl'] != 0:  # Only actual trades
                        all_pnl.append(trade['pnl'])

        if all_pnl:
            ax.hist(all_pnl, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_pnl), color='red', linestyle='--',
                      label='.2f')
            ax.set_xlabel('P&L ($)')
            ax.set_ylabel('Frequency')
            ax.legend()

    def _plot_hourly_performance(self, ax, reports):
        """Plot performance by hour"""
        hourly_returns = {}

        for report in reports:
            if 'trades' in report:
                for trade in report['trades']:
                    if 'timestamp' in trade and 'pnl' in trade:
                        try:
                            hour = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')).hour
                            if hour not in hourly_returns:
                                hourly_returns[hour] = []
                            hourly_returns[hour].append(trade['pnl'])
                        except:
                            pass

        if hourly_returns:
            hours = sorted(hourly_returns.keys())
            avg_returns = [np.mean(hourly_returns[h]) for h in hours]
            trade_counts = [len(hourly_returns[h]) for h in hours]

            ax.bar(hours, avg_returns, alpha=0.7, label='Avg P&L')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Average P&L ($)')
            ax.legend()

    def _plot_cumulative_returns(self, ax, reports):
        """Plot cumulative returns"""
        for report in reports:
            if 'trades' in report and report['trades']:
                cumulative = [0]
                for trade in report['trades']:
                    if 'pnl' in trade:
                        cumulative.append(cumulative[-1] + trade['pnl'])

                if len(cumulative) > 1:
                    ax.plot(range(len(cumulative)), cumulative,
                           label=f"{report['type']}",
                           marker='o', markersize=2)

        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.legend()

def main():
    """Main function"""
    monitor = PerformanceMonitor()
    monitor.generate_performance_dashboard()

if __name__ == '__main__':
    main()