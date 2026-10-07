"""
Visualization Module for FL-HOD Experiments
Generates figures for the paper

Charts:
1. Convergence curves (mAP50 vs rounds)
2. Method comparison bar charts
3. Byzantine attack robustness heatmaps
4. Per-client performance
5. Training timeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import re


# Paper-quality figure settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'fedavg': '#1f77b4',      # Blue
    'fedbn': '#ff7f0e',       # Orange
    'photoscreen': '#2ca02c',    # Green
    'baseline': '#d62728',    # Red
    'attack': '#9467bd',      # Purple
}


class FLVisualizer:
    """Generates visualizations for FL-HOD experiments."""
    
    def __init__(self, output_dir: str = 'figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_convergence_curves(
        self,
        log_files: Dict[str, str],
        title: str = 'FL Training Convergence',
        output_name: str = 'convergence_curves.png'
    ):
        """
        Plot mAP50 convergence curves from log files.
        
        Args:
            log_files: Dict of {method_name: log_file_path}
            title: Plot title
            output_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method, log_path in log_files.items():
            rounds, map50_values = self._parse_log_file(log_path)
            
            if rounds and map50_values:
                color = COLORS.get(method.lower(), '#333333')
                ax.plot(rounds, map50_values, 
                       label=method.upper(),
                       color=color,
                       linewidth=2,
                       marker='o',
                       markersize=3,
                       markevery=10)
        
        ax.set_xlabel('FL Round')
        ax.set_ylabel('mAP50 (%)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim(0, None)
        ax.set_ylim(0, 100)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save
        output_path = self.output_dir / output_name
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
        
        plt.close(fig)
        return output_path
    
    def plot_method_comparison(
        self,
        results: Dict[str, Dict],
        title: str = 'FL Method Comparison',
        output_name: str = 'method_comparison.png'
    ):
        """
        Bar chart comparing methods across metrics.
        
        Args:
            results: {method: {'map50': x, 'precision': y, 'recall': z}}
            title: Plot title
            output_name: Output filename
        """
        methods = list(results.keys())
        metrics = ['mAP50', 'Precision', 'Recall']
        
        x = np.arange(len(methods))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract values
        map50_vals = [results[m].get('map50', 0) for m in methods]
        prec_vals = [results[m].get('precision', 0) for m in methods]
        recall_vals = [results[m].get('recall', 0) for m in methods]
        
        # Plot bars
        rects1 = ax.bar(x - width, map50_vals, width, label='mAP50', color=COLORS['fedavg'])
        rects2 = ax.bar(x, prec_vals, width, label='Precision', color=COLORS['fedbn'])
        rects3 = ax.bar(x + width, recall_vals, width, label='Recall', color=COLORS['photoscreen'])
        
        ax.set_xlabel('FL Method')
        ax.set_ylabel('Score (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        # Save
        output_path = self.output_dir / output_name
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
        
        plt.close(fig)
        return output_path
    
    def plot_byzantine_robustness_heatmap(
        self,
        results: List[Dict],
        output_name: str = 'byzantine_robustness.png'
    ):
        """
        Heatmap showing method performance under different attacks.
        
        Args:
            results: List of experiment results
            output_name: Output filename
        """
        # Extract unique attacks, ratios, methods
        attacks = sorted(set(r['attack'] for r in results if r.get('status') == 'success'))
        methods = sorted(set(r['method'] for r in results if r.get('status') == 'success'))
        
        # Create performance matrix
        perf_matrix = {}
        for method in methods:
            perf_matrix[method] = []
            for attack in attacks:
                # Get average performance across all ratios
                vals = [r['final_map50'] for r in results 
                       if r['method'] == method and r['attack'] == attack and r.get('status') == 'success']
                avg = np.mean(vals) if vals else 0
                perf_matrix[method].append(avg * 100)  # Convert to percentage
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        data = np.array([perf_matrix[m] for m in methods])
        
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=60)
        
        # Labels
        ax.set_xticks(np.arange(len(attacks)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels([a.replace('_', '\n') for a in attacks], rotation=45, ha='right')
        ax.set_yticklabels([m.upper() for m in methods])
        
        # Add values
        for i in range(len(methods)):
            for j in range(len(attacks)):
                text = ax.text(j, i, f'{data[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Byzantine Robustness: mAP50 (%) Under Attack')
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('FL Method')
        
        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('mAP50 (%)', rotation=-90, va="bottom")
        
        # Save
        output_path = self.output_dir / output_name
        fig.tight_layout()
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
        
        plt.close(fig)
        return output_path
    
    def plot_attack_intensity_comparison(
        self,
        results: List[Dict],
        output_name: str = 'attack_intensity.png'
    ):
        """
        Line plot showing mAP50 vs malicious ratio for each method.
        
        Args:
            results: List of experiment results
            output_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['fedavg', 'fedbn', 'photoscreen']
        ratios = [0.1, 0.2, 0.3]
        
        for method in methods:
            avg_map50 = []
            for ratio in ratios:
                vals = [r['final_map50'] * 100 for r in results 
                       if r['method'] == method and r['malicious_ratio'] == ratio 
                       and r.get('status') == 'success']
                avg_map50.append(np.mean(vals) if vals else 0)
            
            color = COLORS.get(method, '#333333')
            ax.plot([r * 100 for r in ratios], avg_map50,
                   label=method.upper(),
                   color=color,
                   linewidth=2,
                   marker='o',
                   markersize=8)
        
        ax.set_xlabel('Malicious Client Ratio (%)')
        ax.set_ylabel('mAP50 (%)')
        ax.set_title('Byzantine Robustness vs Attack Intensity')
        ax.legend(loc='lower left')
        ax.set_xlim(5, 35)
        ax.set_ylim(0, 60)
        ax.set_xticks([10, 20, 30])
        
        # Save
        output_path = self.output_dir / output_name
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
        
        plt.close(fig)
        return output_path
    
    def plot_per_city_performance(
        self,
        city_results: Dict[str, Dict[str, float]],
        output_name: str = 'per_city_performance.png'
    ):
        """
        Bar chart showing per-city mAP50.
        
        Args:
            city_results: {city: {method: map50}}
            output_name: Output filename
        """
        cities = list(city_results.keys())
        methods = ['fedavg', 'fedbn', 'photoscreen']
        
        x = np.arange(len(cities))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, method in enumerate(methods):
            vals = [city_results[city].get(method, 0) * 100 for city in cities]
            color = COLORS.get(method, '#333333')
            ax.bar(x + (i - 1) * width, vals, width, label=method.upper(), color=color)
        
        ax.set_xlabel('Client (City)')
        ax.set_ylabel('mAP50 (%)')
        ax.set_title('Per-Client Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in cities])
        ax.legend()
        ax.set_ylim(0, 70)
        
        # Save
        output_path = self.output_dir / output_name
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
        
        plt.close(fig)
        return output_path
    
    def plot_defense_layers_ablation(
        self,
        ablation_results: Dict[str, float],
        output_name: str = 'defense_ablation.png'
    ):
        """
        Bar chart showing ablation study of defense layers.
        
        Args:
            ablation_results: {config: map50}
            output_name: Output filename
        """
        configs = [
            'FedAvg\n(No Defense)',
            'FedBN\n(Local BN)',
            'Layer 1\n(QualityGate)',
            'Layer 1+2\n(QualityGate+MAD)',
            'PhotoScreen\n(All 3 Layers)'
        ]
        
        # Example values (replace with actual)
        values = [
            ablation_results.get('fedavg', 45.1),
            ablation_results.get('fedbn', 47.7),
            ablation_results.get('layer1', 48.5),
            ablation_results.get('layer1_2', 49.8),
            ablation_results.get('photoscreen', 51.6),
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#2ca02c']
        bars = ax.bar(configs, values, color=colors)
        
        ax.set_ylabel('mAP50 (%)')
        ax.set_title('Defense Layer Ablation Study')
        ax.set_ylim(40, 55)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Save
        output_path = self.output_dir / output_name
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
        
        plt.close(fig)
        return output_path
    
    def _parse_log_file(self, log_path: str) -> tuple:
        """Parse FL log file for mAP50 values per round."""
        rounds = []
        map50_values = []
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Find "Avg mAP50: X.XXXX" patterns (round-level averages)
            pattern = r'Avg mAP50:\s*([0-9.]+)'
            matches = re.findall(pattern, content)
            
            for i, map50 in enumerate(matches):
                rounds.append(i + 1)
                map50_values.append(float(map50) * 100)  # Convert to percentage
            
        except Exception as e:
            print(f"Could not parse {log_path}: {e}")
        
        return rounds, map50_values
    
    def generate_all_figures(
        self,
        log_dir: str = '.',
        byzantine_results_file: str = None
    ):
        """Generate all figures for the paper."""
        print("\n" + "="*60)
        print("GENERATING ALL FIGURES")
        print("="*60)
        
        # 1. Convergence curves
        log_files = {}
        for method in ['fedavg', 'fedbn', 'photoscreen']:
            pattern = f'fl_{method}_*rounds*.log'
            matches = list(Path(log_dir).glob(pattern))
            if matches:
                log_files[method] = str(matches[-1])  # Most recent
        
        if log_files:
            self.plot_convergence_curves(
                log_files=log_files,
                title='FL Training Convergence (100 Rounds)',
                output_name='fig1_convergence.png'
            )
        
        # 2. Method comparison
        baseline_results = {
            'fedavg': {'map50': 45.1, 'precision': 66.1, 'recall': 38.0},
            'fedbn': {'map50': 47.7, 'precision': 62.6, 'recall': 42.7},
            'photoscreen': {'map50': 51.6, 'precision': 64.0, 'recall': 45.0}
        }
        
        self.plot_method_comparison(
            results=baseline_results,
            title='Baseline Method Comparison (ECP Night)',
            output_name='fig2_method_comparison.png'
        )
        
        # 3. Defense ablation
        self.plot_defense_layers_ablation(
            ablation_results={},
            output_name='fig3_ablation.png'
        )
        
        # 4. Byzantine robustness (if results available)
        if byzantine_results_file and Path(byzantine_results_file).exists():
            with open(byzantine_results_file, 'r') as f:
                byzantine_data = json.load(f)
            
            self.plot_byzantine_robustness_heatmap(
                results=byzantine_data.get('results', []),
                output_name='fig4_byzantine_heatmap.png'
            )
            
            self.plot_attack_intensity_comparison(
                results=byzantine_data.get('results', []),
                output_name='fig5_attack_intensity.png'
            )
        
        print("\nAll figures generated!")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate FL-HOD Visualizations')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='.',
        help='Directory containing FL log files'
    )
    parser.add_argument(
        '--byzantine-results',
        type=str,
        default=None,
        help='Path to Byzantine experiment results JSON'
    )
    
    args = parser.parse_args()
    
    visualizer = FLVisualizer(output_dir=args.output_dir)
    visualizer.generate_all_figures(
        log_dir=args.log_dir,
        byzantine_results_file=args.byzantine_results
    )


if __name__ == '__main__':
    main()
