"""
Multi-Seed No-Attack Baseline Experiments for FL-HOD
=====================================================
Runs 5 seeds × 3 methods = 15 experiments to get mean±std for clean baselines.

Tracks per-experiment:
  - mAP50, mAP50-95, Precision, Recall
  - Training time (total + per-round average)
  - GPU memory peak
  - Per-round timing

Uses IDENTICAL configuration to the 90 Byzantine experiments.
"""

import flwr as fl
from pathlib import Path
import yaml
import sys
import json
import time
import os
import random
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from fl.client_yolo10s import create_yolo_client_fn
from fl.server_strategies import FedAvgStrategy, FedBNStrategy, PHOTOSCREENStrategy


METHODS = ['fedavg', 'fedbn', 'photoscreen']
SEEDS = [42, 123, 256, 512, 1024]
NUM_ROUNDS = 50
NUM_CLIENTS = 7


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility variation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Don't set deterministic=True — we WANT variation between seeds
    # but we want each seed to be reproducible
    print(f"  Global seed set to {seed}")


def get_gpu_stats():
    """Get current GPU memory stats."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            'gpu_mem_allocated_gb': round(allocated, 3),
            'gpu_mem_reserved_gb': round(reserved, 3),
            'gpu_mem_peak_gb': round(max_allocated, 3)
        }
    return {}


class MultiSeedRunner:
    """Runs no-attack experiments across multiple seeds."""
    
    def __init__(self, base_dir: Path, output_dir: Path):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client_configs = self._load_client_configs()
        self.results_file = self.output_dir / 'multiseed_results.json'
        
        print("\n" + "=" * 80)
        print("MULTI-SEED NO-ATTACK BASELINE EXPERIMENTS")
        print("=" * 80)
        print(f"Methods:  {METHODS}")
        print(f"Seeds:    {SEEDS}")
        print(f"Total:    {len(METHODS) * len(SEEDS)} experiments")
        print(f"Rounds:   {NUM_ROUNDS}")
        print(f"Clients:  {len(self.client_configs)}")
        print(f"Attack:   NONE (clean baseline)")
        print(f"Output:   {self.output_dir}")
        print(f"Est time: ~{len(METHODS) * len(SEEDS) * 85 / 60:.0f} hours")
        print("=" * 80 + "\n")
    
    def _load_client_configs(self) -> Dict[int, Dict]:
        """Load configs for 7 ECP city clients — IDENTICAL to Byzantine runs."""
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
                print(f"Warning: No config found for {city}")
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
        """Create FL strategy — IDENTICAL to Byzantine runs."""
        min_clients = len(self.client_configs)
        if method == 'fedavg':
            return FedAvgStrategy(
                fraction_fit=1.0, fraction_evaluate=1.0,
                min_fit_clients=min_clients, min_evaluate_clients=min_clients,
                min_available_clients=min_clients
            )
        elif method == 'fedbn':
            return FedBNStrategy(
                fraction_fit=1.0, fraction_evaluate=1.0,
                min_fit_clients=min_clients, min_evaluate_clients=min_clients,
                min_available_clients=min_clients
            )
        elif method == 'photoscreen':
            iara_config = self.base_dir / 'data/bn_statistics/iara_config.json'
            return PHOTOSCREENStrategy(
                iara_config_path=str(iara_config),
                enable_layer1=True, enable_layer2=True, enable_layer3=True,
                mad_threshold=3.0, cosine_threshold=0.5,
                fraction_fit=1.0, fraction_evaluate=1.0,
                min_fit_clients=min_clients, min_evaluate_clients=min_clients,
                min_available_clients=min_clients
            )
    
    def run_single(self, method: str, seed: int, exp_num: int, total: int) -> Dict:
        """Run a single no-attack experiment with a specific seed."""
        exp_id = f"no_attack_{method}_seed{seed}"
        
        print("\n" + "=" * 70)
        print(f"[{exp_num}/{total}] EXPERIMENT: {exp_id}")
        print(f"  Method: {method.upper()} | Seed: {seed} | Rounds: {NUM_ROUNDS}")
        print("=" * 70)
        
        # Set seed BEFORE creating anything
        set_global_seed(seed)
        
        # Reset GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        strategy = self._create_strategy(method)
        fedbn = method in ['fedbn', 'photoscreen']
        extract_bn = method == 'photoscreen'
        
        client_fn = create_yolo_client_fn(
            client_configs=self.client_configs,
            fedbn=fedbn,
            extract_bn_stats=extract_bn,
            attack=None,
            malicious_clients=[]
        )
        
        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(self.client_configs),
                config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
                strategy=strategy,
                ray_init_args={"num_cpus": 14, "num_gpus": 1},
                client_resources={"num_cpus": 2, "num_gpus": 0.14}
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract metrics
            final_map50 = self._extract_metric(history, 'avg_map50')
            final_map50_95 = self._extract_metric(history, 'avg_map50_95')
            final_precision = self._extract_metric(history, 'avg_precision')
            final_recall = self._extract_metric(history, 'avg_recall')
            map50_history = self._extract_history(history, 'avg_map50')
            
            # GPU stats
            gpu_stats = get_gpu_stats()
            
            result = {
                'experiment_id': exp_id,
                'method': method,
                'seed': seed,
                'attack': 'none',
                'malicious_ratio': 0.0,
                'final_map50': final_map50,
                'final_metrics': {
                    'mAP50': final_map50,
                    'mAP50_95': final_map50_95,
                    'precision': final_precision,
                    'recall': final_recall
                },
                'map50_history': map50_history,
                'num_rounds': NUM_ROUNDS,
                'timing': {
                    'total_seconds': duration,
                    'total_minutes': round(duration / 60, 2),
                    'avg_seconds_per_round': round(duration / NUM_ROUNDS, 2),
                },
                'gpu': gpu_stats,
                'status': 'success'
            }
            
            print(f"\n  {exp_id} complete in {duration/60:.1f} min")
            print(f"     mAP50={final_map50:.4f} | mAP50-95={final_map50_95:.4f} | P={final_precision:.4f} | R={final_recall:.4f}")
            print(f"     Time: {duration/60:.1f} min ({duration/NUM_ROUNDS:.1f} s/round) | GPU peak: {gpu_stats.get('gpu_mem_peak_gb', 'N/A')} GB")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                'experiment_id': exp_id,
                'method': method,
                'seed': seed,
                'attack': 'none',
                'error': str(e),
                'status': 'failed',
                'timing': {'total_seconds': time.time() - start_time},
            }
            print(f"\n  {exp_id} FAILED: {e}")
        
        return result
    
    def _extract_metric(self, history, key: str) -> float:
        """Extract final value of a metric from history."""
        for attr in ['metrics_distributed_fit', 'metrics_distributed', 'metrics_centralized']:
            if hasattr(history, attr):
                metrics = getattr(history, attr)
                if metrics and key in metrics and metrics[key]:
                    return float(metrics[key][-1][1])
        return 0.0
    
    def _extract_history(self, history, key: str) -> List[float]:
        """Extract full history of a metric."""
        for attr in ['metrics_distributed_fit', 'metrics_distributed', 'metrics_centralized']:
            if hasattr(history, attr):
                metrics = getattr(history, attr)
                if metrics and key in metrics and metrics[key]:
                    return [float(m[1]) for m in metrics[key]]
        return []
    
    def run_all(self):
        """Run all 15 experiments (3 methods × 5 seeds)."""
        all_results = []
        total = len(METHODS) * len(SEEDS)
        exp_num = 0
        
        overall_start = time.time()
        
        for method in METHODS:
            for seed in SEEDS:
                exp_num += 1
                result = self.run_single(method, seed, exp_num, total)
                all_results.append(result)
                self._save_results(all_results, overall_start)
                
                # Print running summary after each method completes a seed
                self._print_running_stats(all_results)
        
        self._print_final_summary(all_results, overall_start)
        return all_results
    
    def _save_results(self, results: List[Dict], overall_start: float):
        """Save results to JSON after each experiment."""
        elapsed = time.time() - overall_start
        with open(self.results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'description': 'Multi-seed no-attack baselines (5 seeds × 3 methods)',
                'config': {
                    'num_rounds': NUM_ROUNDS,
                    'local_epochs': 3,
                    'batch_size': 4,
                    'imgsz': 640,
                    'optimizer': 'AdamW',
                    'lr0': 0.001,
                    'model': 'yolov10s.pt',
                    'num_clients': NUM_CLIENTS,
                    'cities': [c['city'] for c in self.client_configs.values()],
                    'seeds': SEEDS,
                    'attack': 'none',
                },
                'elapsed_seconds': round(elapsed, 1),
                'num_experiments_completed': len(results),
                'num_experiments_total': len(METHODS) * len(SEEDS),
                'results': results
            }, f, indent=2)
        print(f"  Saved ({len(results)}/{len(METHODS)*len(SEEDS)}) to {self.results_file}")
    
    def _print_running_stats(self, results: List[Dict]):
        """Print running statistics for completed methods."""
        successful = [r for r in results if r['status'] == 'success']
        for method in METHODS:
            method_results = [r for r in successful if r['method'] == method]
            if len(method_results) >= 2:
                maps = [r['final_map50'] * 100 for r in method_results]
                times = [r['timing']['total_minutes'] for r in method_results]
                print(f"\n  {method.upper()} ({len(method_results)} seeds): "
                      f"mAP50 = {np.mean(maps):.2f}% ± {np.std(maps):.2f}% | "
                      f"Time = {np.mean(times):.1f} ± {np.std(times):.1f} min")
    
    def _print_final_summary(self, results: List[Dict], overall_start: float):
        """Print comprehensive final summary."""
        total_time = time.time() - overall_start
        successful = [r for r in results if r['status'] == 'success']
        
        print("\n" + "=" * 80)
        print("MULTI-SEED NO-ATTACK BASELINE — FINAL SUMMARY")
        print("=" * 80)
        print(f"Total experiments: {len(results)} ({len(successful)} success, {len(results)-len(successful)} failed)")
        print(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.0f} min)")
        print()
        
        print(f"{'Method':<12} {'Seeds':>6} {'mAP50 (%)':>16} {'mAP50-95 (%)':>16} {'Prec (%)':>14} {'Recall (%)':>14} {'Time (min)':>14}")
        print("-" * 100)
        
        for method in METHODS:
            mr = [r for r in successful if r['method'] == method]
            if not mr:
                print(f"  {method.upper():<10} {'N/A':>6}")
                continue
            
            maps = [r['final_map50'] * 100 for r in mr]
            map95s = [r['final_metrics']['mAP50_95'] * 100 for r in mr]
            precs = [r['final_metrics']['precision'] * 100 for r in mr]
            recs = [r['final_metrics']['recall'] * 100 for r in mr]
            times = [r['timing']['total_minutes'] for r in mr]
            
            print(f"  {method.upper():<10} {len(mr):>5}  "
                  f"{np.mean(maps):>6.2f} ± {np.std(maps):.2f}  "
                  f"{np.mean(map95s):>6.2f} ± {np.std(map95s):.2f}  "
                  f"{np.mean(precs):>5.2f} ± {np.std(precs):.2f}  "
                  f"{np.mean(recs):>5.2f} ± {np.std(recs):.2f}  "
                  f"{np.mean(times):>5.1f} ± {np.std(times):.1f}")
        
        # Computational cost comparison
        print("\n" + "-" * 80)
        print("COMPUTATIONAL COST COMPARISON")
        print("-" * 80)
        
        for method in METHODS:
            mr = [r for r in successful if r['method'] == method]
            if not mr:
                continue
            times = [r['timing']['total_seconds'] for r in mr]
            per_round = [r['timing']['avg_seconds_per_round'] for r in mr]
            gpu_peaks = [r['gpu'].get('gpu_mem_peak_gb', 0) for r in mr if r.get('gpu')]
            
            print(f"  {method.upper():<12}: "
                  f"Total={np.mean(times)/60:.1f}±{np.std(times)/60:.1f} min | "
                  f"Per-round={np.mean(per_round):.1f}±{np.std(per_round):.1f} s | "
                  f"GPU peak={np.mean(gpu_peaks):.2f} GB" if gpu_peaks else
                  f"  {method.upper():<12}: "
                  f"Total={np.mean(times)/60:.1f}±{np.std(times)/60:.1f} min | "
                  f"Per-round={np.mean(per_round):.1f}±{np.std(per_round):.1f} s")
        
        # Defense overhead
        fedavg_times = [r['timing']['total_seconds'] for r in successful if r['method'] == 'fedavg']
        photoscreen_times = [r['timing']['total_seconds'] for r in successful if r['method'] == 'photoscreen']
        if fedavg_times and photoscreen_times:
            overhead = np.mean(photoscreen_times) - np.mean(fedavg_times)
            overhead_pct = overhead / np.mean(fedavg_times) * 100
            print(f"\n  PhotoScreen defense overhead vs FedAvg: {overhead:.1f}s ({overhead_pct:+.2f}%)")
        
        print("=" * 80)


def main():
    base_dir = Path(__file__).parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / 'results' / f'no_attack_multiseed_{timestamp}'
    
    runner = MultiSeedRunner(base_dir=base_dir, output_dir=output_dir)
    runner.run_all()


if __name__ == '__main__':
    main()
