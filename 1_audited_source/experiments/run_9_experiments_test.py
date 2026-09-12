"""
9-Experiment Test Run for Byzantine Robustness Evaluation
Tests 3 attacks × 1 malicious ratio × 3 methods = 9 experiments

Purpose: Validate the cache fix before full 90-experiment run
Created: December 21, 2025
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
import subprocess

sys.path.append(str(Path(__file__).parent.parent))
from fl.client_yolo10s import create_yolo_client_fn
from fl.server_strategies import FedAvgStrategy, FedBNStrategy, PHOTOSCREENStrategy
from experiments.attacks.byzantine_attacks import create_attack, ATTACK_REGISTRY


# REDUCED Configuration for Quick Test
ATTACKS = ['gaussian_noise', 'label_flip', 'sign_flip']  # Only 3 attacks
MALICIOUS_RATIOS = [0.2]  # Only 20% malicious (2 out of 7 clients)
METHODS = ['fedavg', 'fedbn', 'photoscreen']  # All 3 FL methods
NUM_ROUNDS = 10  # Reduced rounds for testing
NUM_CLIENTS = 7  # ECP cities


def clean_cache_files():
    """Clean all cache files before running experiments."""
    import shutil
    
    print("\nCleaning cache files...")
    
    # 1. Delete shared cache files
    cache_patterns = [
        Path("/ANON/fl_hod/data/yolo/labels/ecp/train.cache"),
        Path("/ANON/fl_hod/data/yolo/labels/ecp/val.cache"),
    ]
    for cache in cache_patterns:
        if cache.exists():
            cache.unlink()
            print(f"   Deleted: {cache}")
    
    # 2. Delete worker cache directories in /tmp
    import glob
    for d in glob.glob("/tmp/yolo_cache_*"):
        shutil.rmtree(d, ignore_errors=True)
        print(f"   Deleted: {d}")
    
    for d in glob.glob("/tmp/fl_yolo_*"):
        shutil.rmtree(d, ignore_errors=True)
        print(f"   Deleted: {d}")
    
    print("Cache cleanup complete\n")


class ByzantineExperimentRunner:
    """Runs Byzantine robustness experiments."""
    
    def __init__(self, base_dir: Path, output_dir: Path, num_rounds: int = 10):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.num_rounds = num_rounds
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load client configs
        self.client_configs = self._load_client_configs()
        
        # Results storage
        self.results = {}
        self.results_file = self.output_dir / 'byzantine_9test_results.json'
        
        print("\n" + "="*80)
        print("BYZANTINE ROBUSTNESS - 9 EXPERIMENT TEST RUN")
        print("="*80)
        print(f"Attacks: {ATTACKS}")
        print(f"Malicious Ratios: {MALICIOUS_RATIOS}")
        print(f"Methods: {METHODS}")
        print(f"Total Experiments: {len(ATTACKS) * len(MALICIOUS_RATIOS) * len(METHODS)}")
        print(f"Rounds per experiment: {self.num_rounds}")
        print(f"Output: {self.output_dir}")
        print("="*80 + "\n")
    
    def _load_client_configs(self) -> Dict[int, Dict]:
        """Load configurations for 7 ECP city clients."""
        cities = ['budapest', 'koeln', 'leipzig', 'lyon', 'prague', 'roma', 'zagreb']
        configs = {}
        
        for i, city in enumerate(cities):
            yaml_path = self.base_dir / f'data/yolo/client_configs/client_{city}.yaml'
            
            if not yaml_path.exists():
                print(f" Warning: No config found for {city}")
                continue
            
            configs[i] = {
                'data_yaml': str(yaml_path),
                'city': city,
                'model_path': 'yolov10s.pt',
                'local_epochs': 3,  # Keep original epochs
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
                min_available_clients=min_clients,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients
            )
        elif method == 'fedbn':
            return FedBNStrategy(
                min_available_clients=min_clients,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients
            )
        elif method == 'photoscreen':
            return PHOTOSCREENStrategy(
                min_available_clients=min_clients,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run_single_experiment(
        self, 
        attack_type: str,
        malicious_ratio: float,
        method: str,
        exp_id: int
    ) -> Dict:
        """Run a single Byzantine experiment."""
        
        exp_name = f"exp{exp_id:02d}_{attack_type}_{int(malicious_ratio*100)}pct_{method}"
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_id}/9: {exp_name}")
        print(f"{'='*60}")
        
        # Clean cache before each experiment
        clean_cache_files()
        
        start_time = time.time()
        
        # Determine which clients are malicious
        num_malicious = int(len(self.client_configs) * malicious_ratio)
        malicious_clients = list(range(num_malicious))
        
        print(f"Attack: {attack_type}")
        print(f"Malicious clients: {malicious_clients} ({int(malicious_ratio*100)}%)")
        print(f"Method: {method}")
        print(f"Rounds: {self.num_rounds}")
        
        try:
            # Create strategy
            strategy = self._create_strategy(method)
            
            # Create client function with attack configuration
            def client_fn(cid: str):
                client_id = int(cid)
                config = self.client_configs.get(client_id)
                
                if config is None:
                    raise ValueError(f"No config for client {client_id}")
                
                # Configure attack for malicious clients
                is_malicious = client_id in malicious_clients
                attack_config = None
                
                if is_malicious:
                    attack_config = {
                        'type': attack_type,
                        'strength': 0.5,  # 50% attack strength
                        'target_class': 0  # Pedestrian class
                    }
                
                return create_yolo_client_fn(
                    data_yaml=config['data_yaml'],
                    model_path=config['model_path'],
                    client_id=client_id,
                    local_epochs=config['local_epochs'],
                    batch_size=config['batch_size'],
                    imgsz=config['imgsz'],
                    device=config['device'],
                    attack_config=attack_config
                )
            
            # Run simulation
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(self.client_configs),
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
                ray_init_args={"num_cpus": 14, "num_gpus": 1},
                client_resources={"num_cpus": 2, "num_gpus": 0.14}
            )
            
            # Extract results
            elapsed = time.time() - start_time
            
            # Get final metrics
            final_metrics = {}
            if history.metrics_distributed_fit:
                last_round = max(history.metrics_distributed_fit.keys())
                final_metrics = history.metrics_distributed_fit[last_round]
            
            result = {
                'experiment': exp_name,
                'attack': attack_type,
                'malicious_ratio': malicious_ratio,
                'method': method,
                'num_rounds': self.num_rounds,
                'malicious_clients': malicious_clients,
                'elapsed_time': elapsed,
                'success': True,
                'final_metrics': final_metrics,
                'error': None
            }
            
            # Calculate average mAP50 from final round
            if history.metrics_distributed_fit:
                mAPs = []
                for r, metrics in history.metrics_distributed_fit.items():
                    if isinstance(metrics, dict) and 'mAP50' in metrics:
                        mAPs.append(metrics['mAP50'])
                if mAPs:
                    result['avg_mAP50'] = np.mean(mAPs)
                    result['final_mAP50'] = mAPs[-1] if mAPs else 0
            
            print(f"\nExperiment {exp_id} COMPLETED in {elapsed/60:.1f} minutes")
            if 'final_mAP50' in result:
                print(f"   Final mAP50: {result['final_mAP50']:.4f}")
            
            return result
            
        except Exception as e:
            import traceback
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\nExperiment {exp_id} FAILED after {elapsed/60:.1f} minutes")
            print(f"   Error: {error_msg}")
            traceback.print_exc()
            
            return {
                'experiment': exp_name,
                'attack': attack_type,
                'malicious_ratio': malicious_ratio,
                'method': method,
                'success': False,
                'elapsed_time': elapsed,
                'error': error_msg
            }
    
    def run_all_experiments(self):
        """Run all 9 test experiments."""
        
        exp_id = 0
        total = len(ATTACKS) * len(MALICIOUS_RATIOS) * len(METHODS)
        
        print(f"\nStarting {total} Byzantine robustness experiments...\n")
        
        for attack in ATTACKS:
            for ratio in MALICIOUS_RATIOS:
                for method in METHODS:
                    exp_id += 1
                    
                    result = self.run_single_experiment(
                        attack_type=attack,
                        malicious_ratio=ratio,
                        method=method,
                        exp_id=exp_id
                    )
                    
                    # Store result
                    key = f"{attack}_{int(ratio*100)}pct_{method}"
                    self.results[key] = result
                    
                    # Save after each experiment
                    self._save_results()
                    
                    # Status update
                    success_count = sum(1 for r in self.results.values() if r.get('success', False))
                    fail_count = sum(1 for r in self.results.values() if not r.get('success', True))
                    print(f"\nProgress: {exp_id}/{total} | {success_count} success | {fail_count} failed")
        
        # Final summary
        self._print_summary()
    
    def _save_results(self):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def _print_summary(self):
        """Print summary of all experiments."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        success = sum(1 for r in self.results.values() if r.get('success', False))
        failed = len(self.results) - success
        
        print(f"\nTotal: {len(self.results)} experiments")
        print(f"Success: {success}")
        print(f"Failed: {failed}")
        
        if success > 0:
            print("\nResults by Method:")
            for method in METHODS:
                method_results = [r for r in self.results.values() 
                                  if r.get('method') == method and r.get('success')]
                if method_results:
                    avg_mAP = np.mean([r.get('final_mAP50', 0) for r in method_results])
                    print(f"   {method.upper()}: mAP50 = {avg_mAP:.4f}")
        
        if failed > 0:
            print("\nFailed Experiments:")
            for key, r in self.results.items():
                if not r.get('success', True):
                    print(f"   - {key}: {r.get('error', 'Unknown error')}")
        
        print(f"\nResults saved to: {self.results_file}")
        print("="*80)


def main():
    """Main entry point."""
    
    base_dir = Path("/ANON/fl_hod")
    output_dir = base_dir / "experiments" / "results" / "9test_run"
    
    # Initial cleanup
    clean_cache_files()
    
    # Create runner
    runner = ByzantineExperimentRunner(
        base_dir=base_dir,
        output_dir=output_dir,
        num_rounds=NUM_ROUNDS
    )
    
    # Run experiments
    runner.run_all_experiments()
    
    print("\n9-experiment test run completed!")
    print("If all experiments passed, you can run the full 90-experiment evaluation.")


if __name__ == "__main__":
    main()
