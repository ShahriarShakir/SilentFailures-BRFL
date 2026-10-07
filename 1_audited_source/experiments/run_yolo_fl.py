"""
Federated Learning Experiment Runner for YOLOv10-S
Runs FedAvg, FedBN, PhotoScreen baselines on 7 ECP clients.

Federated learning pipeline
"""

import flwr as fl
from pathlib import Path
import yaml
import sys
from typing import Dict, List
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from fl.client_yolo10s import create_yolo_client_fn
from fl.server_strategies import FedAvgStrategy, FedBNStrategy, PHOTOSCREENStrategy


def load_client_configs(base_dir: Path, cities: list = None) -> Dict[int, Dict]:
    """
    Load configurations for ECP city clients.
    
    Args:
        base_dir: Repository root.
        cities: Optional list of city slugs (e.g., ['budapest', 'koeln_a', ...]).
                Defaults to the original 7 ECP-night cities. Pass slugs like
                'budapest_a' / 'budapest_b' for the 14-client split.
    
    Returns:
        {client_id: {'data_yaml': ..., 'city': ...}}
    """
    if cities is None:
        cities = ['budapest', 'koeln', 'leipzig', 'lyon', 'prague', 'roma', 'zagreb']
    configs = {}
    
    for i, city in enumerate(cities):
        yaml_path = base_dir / f'data/yolo/client_configs/client_{city}.yaml'
        
        if not yaml_path.exists():
            print(f" Warning: {yaml_path} not found, skipping {city}")
            continue
        
        configs[i] = {
            'data_yaml': str(yaml_path),
            'city': city,
            'model_path': 'yolov10s.pt',
            'local_epochs': 3,  # Reduced from 5 for faster iteration
            'batch_size': 4,  # Reduced from 12 to fit memory
            'imgsz': 640,  # Reduced from 1280 to fit memory
            'device': '0'
        }
    
    return configs


def run_fedavg_baseline(
    client_configs: Dict[int, Dict],
    num_rounds: int = 100,
    min_fit_clients: int = 7,
    min_available_clients: int = 7
):
    """
    Run FedAvg baseline (no Byzantine defense).
    
    Args:
        client_configs: Client configurations
        num_rounds: FL rounds
        min_fit_clients: Min clients per round
        min_available_clients: Min total clients
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: FedAvg Baseline")
    print("="*80)
    print(f"Clients: {len(client_configs)} ECP cities")
    print(f"Rounds: {num_rounds}")
    print(f"Strategy: Simple weighted averaging")
    print("="*80 + "\n")
    
    # Create strategy
    strategy = FedAvgStrategy(
        fraction_fit=1.0,  # Use all clients
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients
    )
    
    # Create client function
    client_fn = create_yolo_client_fn(
        client_configs=client_configs,
        fedbn=False,
        extract_bn_stats=False  # No QualityGate for FedAvg
    )
    
    # Run FL simulation
    print("Starting FL simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_configs),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 14, "num_gpus": 1},
        client_resources={"num_cpus": 2, "num_gpus": 0.14}  # ~2 clients in parallel max
    )
    
    print("\nFedAvg experiment complete!")
    print(f"Final metrics: {history}")
    
    return history


def run_fedbn_baseline(
    client_configs: Dict[int, Dict],
    num_rounds: int = 100,
    min_fit_clients: int = 7,
    min_available_clients: int = 7
):
    """
    Run FedBN baseline (local BN, aggregate other params).
    
    Args:
        client_configs: Client configurations
        num_rounds: FL rounds
        min_fit_clients: Min clients per round
        min_available_clients: Min total clients
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: FedBN Baseline")
    print("="*80)
    print(f"Clients: {len(client_configs)} ECP cities")
    print(f"Rounds: {num_rounds}")
    print(f"Strategy: Aggregate non-BN params only")
    print("="*80 + "\n")
    
    # Create strategy
    strategy = FedBNStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients
    )
    
    # Create client function with FedBN enabled
    client_fn = create_yolo_client_fn(
        client_configs=client_configs,
        fedbn=True,  # Enable FedBN
        extract_bn_stats=False
    )
    
    # Run FL simulation
    print("Starting FL simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_configs),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 14, "num_gpus": 1},
        client_resources={"num_cpus": 2, "num_gpus": 0.14}
    )
    
    print("\nFedBN experiment complete!")
    print(f"Final metrics: {history}")
    
    return history


def run_photoscreen(
    client_configs: Dict[int, Dict],
    num_rounds: int = 100,
    min_fit_clients: int = 7,
    min_available_clients: int = 7,
    iara_config_path: str = 'data/bn_statistics/iara_config.json'
):
    """
    Run PhotoScreen (three-layer Byzantine defense).
    
    Args:
        client_configs: Client configurations
        num_rounds: FL rounds
        min_fit_clients: Min clients per round
        min_available_clients: Min total clients
        iara_config_path: Path to QualityGate configuration
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: PhotoScreen")
    print("="*80)
    print(f"Clients: {len(client_configs)} ECP cities")
    print(f"Rounds: {num_rounds}")
    print(f"Strategy: Three-layer Byzantine defense")
    print("  Layer 1: QualityGate quality scoring")
    print("  Layer 2: Gradient MAD filtering")
    print("  Layer 3: Semantic consistency")
    print("="*80 + "\n")
    
    # Create PhotoScreen strategy
    strategy = PHOTOSCREENStrategy(
        iara_config_path=iara_config_path,
        enable_layer1=True,
        enable_layer2=True,
        enable_layer3=True,
        mad_threshold=3.0,
        cosine_threshold=0.5,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients
    )
    
    # Create client function with QualityGate enabled
    client_fn = create_yolo_client_fn(
        client_configs=client_configs,
        fedbn=True,  # FedBN + PhotoScreen
        extract_bn_stats=True  # Enable BN stats for QualityGate
    )
    
    # Run FL simulation
    print("Starting FL simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_configs),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 14, "num_gpus": 1},
        client_resources={"num_cpus": 2, "num_gpus": 0.14}
    )
    
    print("\nPhotoScreen experiment complete!")
    print(f"Final metrics: {history}")
    
    return history


def run_all_baselines(
    num_rounds: int = 100,
    experiments: List[str] = None
):
    """
    Run all FL baseline experiments.
    
    Args:
        num_rounds: Number of FL rounds
        experiments: List of experiments to run ['fedavg', 'fedbn', 'photoscreen']
    """
    if experiments is None:
        experiments = ['fedavg', 'fedbn', 'photoscreen']
    
    base_dir = Path(__file__).parent.parent
    
    # Load client configurations
    print("\n" + "="*80)
    print("FEDERATED LEARNING BASELINE EXPERIMENTS")
    print("YOLOv10-S on 7 ECP Night Cities")
    print("="*80 + "\n")
    
    client_configs = load_client_configs(base_dir)
    
    print(f"Loaded {len(client_configs)} client configurations:")
    for cid, config in client_configs.items():
        print(f"  Client {cid}: {config['city']}")
    print()
    
    results = {}
    
    # Run experiments
    if 'fedavg' in experiments:
        results['fedavg'] = run_fedavg_baseline(
            client_configs=client_configs,
            num_rounds=num_rounds
        )
    
    if 'fedbn' in experiments:
        results['fedbn'] = run_fedbn_baseline(
            client_configs=client_configs,
            num_rounds=num_rounds
        )
    
    if 'photoscreen' in experiments:
        iara_config = base_dir / 'data/bn_statistics/iara_config.json'
        results['photoscreen'] = run_photoscreen(
            client_configs=client_configs,
            num_rounds=num_rounds,
            iara_config_path=str(iara_config)
        )
    
    # Summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    
    for exp_name, history in results.items():
        print(f"\n{exp_name.upper()}:")
        # Extract final metrics (placeholder - actual history structure varies)
        print(f"  Status: Complete")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FL Experiments for YOLOv10-S')
    parser.add_argument(
        '--rounds',
        type=int,
        default=100,
        help='Number of FL rounds'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=['fedavg', 'fedbn', 'photoscreen'],
        choices=['fedavg', 'fedbn', 'photoscreen'],
        help='Experiments to run'
    )
    
    args = parser.parse_args()
    
    run_all_baselines(
        num_rounds=args.rounds,
        experiments=args.experiments
    )


if __name__ == '__main__':
    main()
