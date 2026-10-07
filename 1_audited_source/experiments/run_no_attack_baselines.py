"""
No-Attack Baseline Experiments for FL-HOD
Runs 3 experiments: FedAvg, FedBN, PhotoScreen with NO Byzantine attack.

Uses IDENTICAL configuration to the 90 Byzantine experiments:
- 50 rounds, 3 local epochs, batch_size=4, imgsz=640
- AdamW optimizer, lr0=0.001
- 7 ECP city clients, yolov10s.pt base weights
- FP16 training

This produces the "No Attack" row for Tables 3-4 in the paper,
enabling fair comparison with the Byzantine experiments.
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


METHODS = ['fedavg', 'fedbn', 'photoscreen']
NUM_ROUNDS = 50
NUM_CLIENTS = 7


class NoAttackBaselineRunner:
    """Runs no-attack baseline experiments with identical config to Byzantine runs."""
    
    def __init__(self, base_dir: Path, output_dir: Path, num_rounds: int = 50):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.num_rounds = num_rounds
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client_configs = self._load_client_configs()
        self.results = {}
        self.results_file = self.output_dir / 'no_attack_results.json'
        
        print("\n" + "=" * 80)
        print("NO-ATTACK BASELINE EXPERIMENTS")
        print("=" * 80)
        print(f"Methods: {METHODS}")
        print(f"Rounds: {self.num_rounds}")
        print(f"Clients: {len(self.client_configs)}")
        print(f"Attack: NONE (clean baseline)")
        print(f"Output: {self.output_dir}")
        print("=" * 80 + "\n")
    
    def _load_client_configs(self) -> Dict[int, Dict]:
        """Load configurations for 7 ECP city clients — IDENTICAL to Byzantine runs."""
        cities = ['budapest', 'koeln', 'leipzig', 'lyon', 'prague', 'roma', 'zagreb']
        configs = {}
        
        for i, city in enumerate(cities):
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
                print(f"Warning: No config found for {city}, skipping")
                continue
            
            # EXACT same config as run_byzantine_evaluation.py line 91-98
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
        """Create FL strategy — IDENTICAL to Byzantine runs."""
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
    
    def run_single_experiment(self, method: str) -> Dict:
        """Run a single no-attack experiment."""
        exp_id = f"no_attack_{method}"
        
        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {exp_id}")
        print("=" * 60)
        print(f"Method: {method.upper()}")
        print(f"Attack: NONE")
        print(f"Rounds: {self.num_rounds}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create strategy
        strategy = self._create_strategy(method)
        
        # Create client function — NO attack, NO malicious clients
        fedbn = method in ['fedbn', 'photoscreen']
        extract_bn = method == 'photoscreen'
        
        client_fn = create_yolo_client_fn(
            client_configs=self.client_configs,
            fedbn=fedbn,
            extract_bn_stats=extract_bn,
            attack=None,              # NO ATTACK
            malicious_clients=[]       # NO MALICIOUS CLIENTS
        )
        
        # Run FL simulation — IDENTICAL Ray config to Byzantine runs
        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(self.client_configs),
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
                ray_init_args={"num_cpus": 14, "num_gpus": 1},
                client_resources={"num_cpus": 2, "num_gpus": 0.14}
            )
            
            # Extract metrics — same extraction as Byzantine runs
            final_map50 = self._extract_final_map50(history)
            map50_history = self._extract_map50_history(history)
            final_metrics = self._extract_final_all_metrics(history)
            
            result = {
                'experiment_id': exp_id,
                'attack': 'no_attack',
                'malicious_ratio': 0.0,
                'method': method,
                'final_map50': final_map50,
                'final_metrics': final_metrics,
                'map50_history': map50_history,
                'num_rounds': self.num_rounds,
                'duration_seconds': time.time() - start_time,
                'status': 'success'
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                'experiment_id': exp_id,
                'attack': 'no_attack',
                'malicious_ratio': 0.0,
                'method': method,
                'error': str(e),
                'status': 'failed',
                'duration_seconds': time.time() - start_time
            }
        
        duration = time.time() - start_time
        print(f"\n{'' if result['status'] == 'success' else ''} {exp_id} complete in {duration/60:.1f} minutes")
        
        if result.get('status') == 'success':
            fm = result.get('final_metrics', {})
            print(f"   Final mAP50: {result['final_map50']:.4f}")
            print(f"   mAP50-95: {fm.get('mAP50_95', 0):.4f}")
            print(f"   Precision: {fm.get('precision', 0):.4f}")
            print(f"   Recall: {fm.get('recall', 0):.4f}")
        
        return result
    
    def _extract_final_map50(self, history) -> float:
        """Extract final mAP50 from history."""
        try:
            if hasattr(history, 'metrics_distributed_fit'):
                metrics = history.metrics_distributed_fit
                if metrics and 'avg_map50' in metrics:
                    return float(metrics['avg_map50'][-1][1])
            if hasattr(history, 'metrics_distributed'):
                metrics = history.metrics_distributed
                if metrics and 'avg_map50' in metrics:
                    return float(metrics['avg_map50'][-1][1])
            if hasattr(history, 'metrics_centralized'):
                metrics = history.metrics_centralized
                if metrics and 'avg_map50' in metrics:
                    return float(metrics['avg_map50'][-1][1])
            if hasattr(history, 'losses_distributed') and history.losses_distributed:
                losses = history.losses_distributed
                if losses:
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
            if hasattr(history, 'metrics_distributed_fit'):
                metrics = history.metrics_distributed_fit
                if metrics and 'avg_map50' in metrics:
                    return [float(m[1]) for m in metrics['avg_map50']]
            if hasattr(history, 'metrics_distributed'):
                metrics = history.metrics_distributed
                if metrics and 'avg_map50' in metrics:
                    return [float(m[1]) for m in metrics['avg_map50']]
            if hasattr(history, 'losses_distributed') and history.losses_distributed:
                losses = history.losses_distributed
                return [1.0 - float(m[1]) if 0 < m[1] < 1 else 0.0 for m in losses]
            return []
        except Exception as e:
            print(f"Warning: Could not extract mAP50 history: {e}")
            return []
    
    def _extract_final_all_metrics(self, history) -> dict:
        """Extract all final metrics from history."""
        metrics_result = {
            'mAP50': 0.0,
            'mAP50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        try:
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
    
    def run_all(self):
        """Run all 3 no-attack baseline experiments."""
        all_results = []
        
        for i, method in enumerate(METHODS):
            print(f"\n[{i+1}/3] Running {method.upper()} no-attack baseline...")
            result = self.run_single_experiment(method)
            all_results.append(result)
            
            # Save after each experiment
            self._save_results(all_results)
        
        self._print_summary(all_results)
        return all_results
    
    def _save_results(self, results: List[Dict]):
        """Save results to JSON."""
        with open(self.results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'description': 'No-attack baselines with identical config to Byzantine experiments',
                'config': {
                    'num_rounds': self.num_rounds,
                    'local_epochs': 3,
                    'batch_size': 4,
                    'imgsz': 640,
                    'optimizer': 'AdamW',
                    'lr0': 0.001,
                    'model': 'yolov10s.pt',
                    'num_clients': len(self.client_configs),
                    'cities': [c['city'] for c in self.client_configs.values()],
                    'attack': 'none',
                    'malicious_ratio': 0.0
                },
                'num_experiments': len(results),
                'results': results
            }, f, indent=2)
        print(f"Results saved to {self.results_file}")
    
    def _print_summary(self, results: List[Dict]):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("NO-ATTACK BASELINE SUMMARY")
        print("=" * 80)
        
        for r in results:
            status = "" if r['status'] == 'success' else ""
            method = r['method'].upper()
            if r['status'] == 'success':
                fm = r.get('final_metrics', {})
                print(f"  {status} {method}: mAP50={r['final_map50']:.4f} | "
                      f"mAP50-95={fm.get('mAP50_95', 0):.4f} | "
                      f"P={fm.get('precision', 0):.4f} | "
                      f"R={fm.get('recall', 0):.4f} | "
                      f"Time={r['duration_seconds']/60:.1f}min")
            else:
                print(f"  {status} {method}: FAILED - {r.get('error', 'Unknown')}")
        
        print("=" * 80)


def main():
    base_dir = Path(__file__).parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / 'results' / f'no_attack_baselines_{timestamp}'
    
    runner = NoAttackBaselineRunner(
        base_dir=base_dir,
        output_dir=output_dir,
        num_rounds=NUM_ROUNDS
    )
    
    runner.run_all()


if __name__ == '__main__':
    main()
