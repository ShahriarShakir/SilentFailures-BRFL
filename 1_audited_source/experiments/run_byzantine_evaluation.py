"""
Byzantine Robustness Evaluation for FL-HOD
Runs 90 experiments: 10 attacks × 3 malicious ratios × 3 methods

Federated learning pipeline
"""

import flwr as fl
from pathlib import Path
import yaml
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from fl.client_yolo10s import create_yolo_client_fn
from fl.server_strategies import FedAvgStrategy, FedBNStrategy, PHOTOSCREENStrategy
from experiments.attacks.byzantine_attacks import create_attack, ATTACK_REGISTRY


# Configuration - FULL 90 EXPERIMENTS
ATTACKS = list(ATTACK_REGISTRY.keys())  # All 10 attacks
MALICIOUS_RATIOS = [0.1, 0.2, 0.3]  # 10%, 20%, 30% malicious clients
METHODS = ['fedavg', 'fedbn', 'photoscreen']  # 3 FL methods
NUM_ROUNDS = 50  # Full experiment with 50 rounds
NUM_CLIENTS = 7  # ECP cities


class ByzantineExperimentRunner:
    """Runs Byzantine robustness experiments."""
    
    def __init__(self, base_dir: Path, output_dir: Path, num_rounds: int = 50):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.num_rounds = num_rounds
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load client configs
        self.client_configs = self._load_client_configs()
        
        # Results storage
        self.results = {}
        self.results_file = self.output_dir / 'byzantine_results.json'
        
        print("\n" + "="*80)
        print("BYZANTINE ROBUSTNESS EVALUATION")
        print("="*80)
        print(f"Attacks: {len(ATTACKS)}")
        print(f"Malicious Ratios: {MALICIOUS_RATIOS}")
        print(f"Methods: {METHODS}")
        print(f"Total Experiments: {len(ATTACKS) * len(MALICIOUS_RATIOS) * len(METHODS)}")
        print(f"Output: {self.output_dir}")
        print("="*80 + "\n")
    
    def _load_client_configs(self) -> Dict[int, Dict]:
        """Load configurations for 7 ECP city clients."""
        cities = ['budapest', 'koeln', 'leipzig', 'lyon', 'prague', 'roma', 'zagreb']
        configs = {}
        
        for i, city in enumerate(cities):
            # Try different possible paths
            yaml_paths = [
                self.base_dir / f'data/yolo/client_configs/client_{city}.yaml',
                self.base_dir / f'data/client_{city}.yaml',
            ]
            
            yaml_path = None
            for p in yaml_paths:
                if p.exists():
                    yaml_path = p
                    break
            
            if yaml_path is None:
                print(f" Warning: No config found for {city}, skipping")
                continue
            
            configs[i] = {
                'data_yaml': str(yaml_path),
                'city': city,
                'model_path': 'yolov10s.pt',
                'local_epochs': 3,
                'batch_size': 4,
                'imgsz': 640,
                'device': '0'
            }
        
        return configs
    
    def _create_strategy(self, method: str):
        """Create FL strategy based on method name."""
        min_clients = len(self.client_configs)
        
        if method == 'fedavg':
            return FedAvgStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients,
                min_available_clients=min_clients
            )
        elif method == 'fedbn':
            return FedBNStrategy(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients,
                min_available_clients=min_clients
            )
        elif method == 'photoscreen':
            iara_config = self.base_dir / 'data/bn_statistics/iara_config.json'
            return PHOTOSCREENStrategy(
                iara_config_path=str(iara_config),
                enable_layer1=True,
                enable_layer2=True,
                enable_layer3=True,
                mad_threshold=3.0,
                cosine_threshold=0.5,
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients,
                min_available_clients=min_clients
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run_single_experiment(
        self,
        attack_name: str,
        malicious_ratio: float,
        method: str
    ) -> Dict:
        """
        Run a single Byzantine experiment.
        
        Args:
            attack_name: Name of attack to apply
            malicious_ratio: Fraction of malicious clients
            method: FL method (fedavg, fedbn, photoscreen)
            
        Returns:
            Experiment results dict
        """
        exp_id = f"{attack_name}_{int(malicious_ratio*100)}pct_{method}"
        
        print("\n" + "-"*60)
        print(f"EXPERIMENT: {exp_id}")
        print("-"*60)
        print(f"Attack: {attack_name}")
        print(f"Malicious Ratio: {malicious_ratio*100:.0f}%")
        print(f"Method: {method.upper()}")
        print("-"*60)
        
        start_time = time.time()
        
        # Create attack instance
        attack = create_attack(attack_name, attack_ratio=malicious_ratio, intensity=1.0)
        
        # Determine malicious clients
        num_malicious = int(len(self.client_configs) * malicious_ratio)
        malicious_clients = list(range(num_malicious))
        
        print(f"Malicious clients: {malicious_clients} ({num_malicious}/{len(self.client_configs)})")
        
        # Create strategy
        strategy = self._create_strategy(method)
        
        # Create client function with attack support
        fedbn = method in ['fedbn', 'photoscreen']
        extract_bn = method == 'photoscreen'
        
        client_fn = create_yolo_client_fn(
            client_configs=self.client_configs,
            fedbn=fedbn,
            extract_bn_stats=extract_bn,
            attack=attack,
            malicious_clients=malicious_clients
        )
        
        # Run FL simulation
        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(self.client_configs),
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
                ray_init_args={"num_cpus": 14, "num_gpus": 1},
                client_resources={"num_cpus": 2, "num_gpus": 0.14}
            )
            
            # Extract metrics
            final_map50 = self._extract_final_map50(history)
            map50_history = self._extract_map50_history(history)
            final_metrics = self._extract_final_all_metrics(history)
            
            result = {
                'experiment_id': exp_id,
                'attack': attack_name,
                'malicious_ratio': malicious_ratio,
                'method': method,
                'final_map50': final_map50,
                'final_metrics': final_metrics,  # All metrics: mAP50, mAP50-95, precision, recall
                'map50_history': map50_history,
                'num_rounds': self.num_rounds,
                'duration_seconds': time.time() - start_time,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            result = {
                'experiment_id': exp_id,
                'attack': attack_name,
                'malicious_ratio': malicious_ratio,
                'method': method,
                'error': str(e),
                'status': 'failed'
            }
        
        duration = time.time() - start_time
        print(f"\nExperiment complete in {duration/60:.1f} minutes")
        
        if result.get('status') == 'success':
            fm = result.get('final_metrics', {})
            print(f"   Final mAP50: {result['final_map50']:.4f} | mAP50-95: {fm.get('mAP50_95', 0):.4f}")
            print(f"   Precision: {fm.get('precision', 0):.4f} | Recall: {fm.get('recall', 0):.4f}")
        
        return result
    
    def _extract_final_map50(self, history) -> float:
        """Extract final mAP50 from history."""
        try:
            # Try metrics_distributed_fit first (Flower stores fit metrics here)
            if hasattr(history, 'metrics_distributed_fit'):
                metrics = history.metrics_distributed_fit
                if metrics and 'avg_map50' in metrics:
                    return float(metrics['avg_map50'][-1][1])
            
            # Try metrics_distributed (evaluation metrics)
            if hasattr(history, 'metrics_distributed'):
                metrics = history.metrics_distributed
                if metrics and 'avg_map50' in metrics:
                    return float(metrics['avg_map50'][-1][1])
            
            # Try centralized metrics
            if hasattr(history, 'metrics_centralized'):
                metrics = history.metrics_centralized
                if metrics and 'avg_map50' in metrics:
                    return float(metrics['avg_map50'][-1][1])
            
            # Try losses_distributed as proxy (1 - loss = mAP50)
            if hasattr(history, 'losses_distributed') and history.losses_distributed:
                losses = history.losses_distributed
                if losses:
                    # Loss is (1 - mAP50), so mAP50 = (1 - loss)
                    final_loss = losses[-1][1]
                    if 0 < final_loss < 1:
                        return 1.0 - final_loss
            
            return 0.0
        except Exception as e:
            print(f"Warning: Could not extract mAP50: {e}")
            return 0.0
    
    def _extract_map50_history(self, history) -> List[float]:
        """Extract mAP50 history from all rounds."""
        try:
            # Try metrics_distributed_fit first
            if hasattr(history, 'metrics_distributed_fit'):
                metrics = history.metrics_distributed_fit
                if metrics and 'avg_map50' in metrics:
                    return [float(m[1]) for m in metrics['avg_map50']]
            
            # Try metrics_distributed
            if hasattr(history, 'metrics_distributed'):
                metrics = history.metrics_distributed
                if metrics and 'avg_map50' in metrics:
                    return [float(m[1]) for m in metrics['avg_map50']]
            
            # Try losses as proxy
            if hasattr(history, 'losses_distributed') and history.losses_distributed:
                losses = history.losses_distributed
                return [1.0 - float(m[1]) if 0 < m[1] < 1 else 0.0 for m in losses]
            
            return []
        except Exception as e:
            print(f"Warning: Could not extract mAP50 history: {e}")
            return []
    
    def _extract_final_all_metrics(self, history) -> dict:
        """Extract all final metrics from history (mAP50, mAP50-95, precision, recall)."""
        metrics_result = {
            'mAP50': 0.0,
            'mAP50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        try:
            # Try metrics_distributed_fit first (Flower stores fit metrics here)
            if hasattr(history, 'metrics_distributed_fit'):
                metrics = history.metrics_distributed_fit
                if metrics:
                    if 'avg_map50' in metrics and metrics['avg_map50']:
                        metrics_result['mAP50'] = float(metrics['avg_map50'][-1][1])
                    if 'avg_map50_95' in metrics and metrics['avg_map50_95']:
                        metrics_result['mAP50_95'] = float(metrics['avg_map50_95'][-1][1])
                    if 'avg_precision' in metrics and metrics['avg_precision']:
                        metrics_result['precision'] = float(metrics['avg_precision'][-1][1])
                    if 'avg_recall' in metrics and metrics['avg_recall']:
                        metrics_result['recall'] = float(metrics['avg_recall'][-1][1])
        except Exception as e:
            print(f"Warning: Could not extract all metrics: {e}")
        
        return metrics_result
    
    def run_all_experiments(
        self,
        attacks: List[str] = None,
        ratios: List[float] = None,
        methods: List[str] = None
    ):
        """
        Run all Byzantine experiments.
        
        Args:
            attacks: List of attacks (default: all)
            ratios: List of malicious ratios (default: all)
            methods: List of FL methods (default: all)
        """
        attacks = attacks or ATTACKS
        ratios = ratios or MALICIOUS_RATIOS
        methods = methods or METHODS
        
        total_experiments = len(attacks) * len(ratios) * len(methods)
        completed = 0
        
        print(f"\nStarting {total_experiments} Byzantine experiments...\n")
        
        all_results = []
        
        for attack in attacks:
            for ratio in ratios:
                for method in methods:
                    completed += 1
                    print(f"\n[{completed}/{total_experiments}]")
                    
                    result = self.run_single_experiment(attack, ratio, method)
                    all_results.append(result)
                    
                    # Save intermediate results
                    self._save_results(all_results)
        
        # Final summary
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: List[Dict]):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'num_experiments': len(results),
                'results': results
            }, f, indent=2)
        print(f"Results saved to {self.results_file}")
    
    def _print_summary(self, results: List[Dict]):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("BYZANTINE EXPERIMENT SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\nResults by Method:")
            for method in METHODS:
                method_results = [r for r in successful if r['method'] == method]
                if method_results:
                    avg_map50 = np.mean([r['final_map50'] for r in method_results])
                    print(f"  {method.upper()}: Avg mAP50 = {avg_map50:.4f}")
            
            print("\nResults by Attack:")
            for attack in ATTACKS:
                attack_results = [r for r in successful if r['attack'] == attack]
                if attack_results:
                    for method in METHODS:
                        method_attack = [r for r in attack_results if r['method'] == method]
                        if method_attack:
                            avg_map50 = np.mean([r['final_map50'] for r in method_attack])
                            print(f"  {attack} + {method}: {avg_map50:.4f}")
        
        print("\n" + "="*80)


def run_quick_validation(num_rounds: int = 5):
    """Run quick validation with 1 attack, 1 ratio, all methods."""
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'results' / 'byzantine_validation'
    
    runner = ByzantineExperimentRunner(
        base_dir=base_dir,
        output_dir=output_dir,
        num_rounds=num_rounds
    )
    
    # Quick test: 1 attack, 1 ratio, all methods
    results = runner.run_all_experiments(
        attacks=['sign_flipping'],
        ratios=[0.2],
        methods=['fedavg', 'fedbn', 'photoscreen']
    )
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Byzantine Robustness Evaluation')
    parser.add_argument(
        '--rounds', type=int, default=50,  # Full experiments with 50 rounds
        help='Number of FL rounds per experiment'
    )
    parser.add_argument(
        '--attacks', nargs='+', default=None,
        help='Attacks to run (default: all 10)'
    )
    parser.add_argument(
        '--ratios', nargs='+', type=float, default=None,
        help='Malicious ratios (default: 0.1, 0.2, 0.3)'
    )
    parser.add_argument(
        '--methods', nargs='+', default=None,
        choices=['fedavg', 'fedbn', 'photoscreen'],
        help='FL methods (default: all)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick validation run (5 rounds, 1 attack, all methods)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_validation(num_rounds=5)
        return
    
    base_dir = Path(__file__).parent.parent
    
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_dir / 'results' / f'byzantine_{timestamp}'
    
    runner = ByzantineExperimentRunner(
        base_dir=base_dir,
        output_dir=output_dir,
        num_rounds=args.rounds
    )
    
    runner.run_all_experiments(
        attacks=args.attacks,
        ratios=args.ratios,
        methods=args.methods
    )


if __name__ == '__main__':
    main()
