"""
FedBN Strategy Implementation
Batch Normalization layer personalization for handling non-IID data.

Reference: FedBN: Federated Learning on Non-IID Features via Local Batch Normalization
          (ICLR 2021)
"""

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class FedBNStrategy(fl.server.strategy.FedAvg):
    """
    FedBN: Federated Learning with Batch Normalization Personalization.
    
    Key idea: Keep BN layers local (client-specific), only aggregate non-BN parameters.
    This helps handle heterogeneous data distributions (e.g., different illumination conditions).
    
    In our case:
    - Each ECP client (Budapest, Koeln, etc.) has different lighting conditions
    - BN statistics (mean, variance) capture these local distributions
    - Only feature extraction weights are shared globally
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("\n" + "="*60)
        print("FL STRATEGY: FedBN")
        print("="*60)
        print("BN layers stay local (client-specific)")
        print("Only non-BN parameters aggregated")
        print("Better handling of illumination heterogeneity")
        print("="*60 + "\n")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates (FedBN variant).
        
        Note: BN parameter filtering happens client-side in get_parameters().
        Server only receives non-BN parameters.
        """
        
        if not results:
            return None, {}
        
        # Convert results to weights (already filtered by client)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate (weighted average by num_samples)
        aggregated_weights = self._weighted_average(weights_results)
        
        # Collect metrics
        metrics = self._collect_metrics(results)
        
        print(f"\n[Round {server_round}] FedBN aggregated {len(results)} clients")
        print(f"  Avg mAP50: {metrics.get('avg_map50', 0):.4f}")
        print(f"  BN layers: Client-specific (not aggregated)")
        
        return ndarrays_to_parameters(aggregated_weights), metrics
    
    def _weighted_average(
        self,
        weights_results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Compute weighted average of non-BN parameters."""
        
        # Calculate total samples
        num_examples_total = sum(num_examples for _, num_examples in weights_results)
        
        # Initialize aggregated weights
        aggregated = [
            np.zeros_like(weights[0])
            for weights, _ in weights_results[:1]
        ]
        
        # Weighted sum
        for i in range(len(aggregated)):
            for weights, num_examples in weights_results:
                aggregated[i] += weights[i] * (num_examples / num_examples_total)
        
        return aggregated
    
    def _collect_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict:
        """Collect and average metrics from clients."""
        
        map50_list = []
        bn_mu_list = []
        bn_sigma_list = []
        
        for _, fit_res in results:
            metrics = fit_res.metrics
            
            if 'map50' in metrics:
                map50_list.append(metrics['map50'])
            
            # BN statistics (for monitoring heterogeneity)
            if 'bn_mu' in metrics:
                bn_mu_list.append(metrics['bn_mu'])
            if 'bn_sigma' in metrics:
                bn_sigma_list.append(metrics['bn_sigma'])
        
        metrics_dict = {
            'avg_map50': np.mean(map50_list) if map50_list else 0.0,
            'num_clients': len(results)
        }
        
        # Add BN statistics variance (measure of heterogeneity)
        if bn_mu_list:
            metrics_dict['bn_mu_variance'] = np.var(bn_mu_list)
        if bn_sigma_list:
            metrics_dict['bn_sigma_variance'] = np.var(bn_sigma_list)
        
        return metrics_dict


class FedBNWithMonitoring(FedBNStrategy):
    """
    FedBN with enhanced monitoring of BN statistics.
    Useful for analyzing illumination heterogeneity across clients.
    """
    
    def __init__(self, log_dir: str = 'fl_experiments/fedbn_logs', **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.bn_stats_history = []
        
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"BN statistics logging enabled: {log_dir}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with BN statistics logging."""
        
        # Log BN statistics before aggregation
        self._log_bn_statistics(server_round, results)
        
        # Standard FedBN aggregation
        return super().aggregate_fit(server_round, results, failures)
    
    def _log_bn_statistics(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]]
    ):
        """Log BN statistics to track illumination heterogeneity."""
        
        import json
        
        round_stats = {
            'round': server_round,
            'clients': []
        }
        
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            
            client_stat = {
                'client_id': metrics.get('client_id', 'unknown'),
                'bn_mu': float(metrics.get('bn_mu', 0)),
                'bn_sigma': float(metrics.get('bn_sigma', 0)),
                'map50': float(metrics.get('map50', 0)),
                'num_samples': fit_res.num_examples
            }
            
            round_stats['clients'].append(client_stat)
        
        self.bn_stats_history.append(round_stats)
        
        # Save to file
        log_file = f"{self.log_dir}/round_{server_round}_bn_stats.json"
        with open(log_file, 'w') as f:
            json.dump(round_stats, f, indent=2)
        
        # Print summary
        bn_mus = [c['bn_mu'] for c in round_stats['clients']]
        bn_sigmas = [c['bn_sigma'] for c in round_stats['clients']]
        
        if bn_mus:
            print(f"\n  BN Statistics Variance:")
            print(f"     μ range: [{min(bn_mus):.4f}, {max(bn_mus):.4f}]")
            print(f"     σ range: [{min(bn_sigmas):.4f}, {max(bn_sigmas):.4f}]")
            print(f"     Heterogeneity: {np.std(bn_mus):.4f} (μ), {np.std(bn_sigmas):.4f} (σ)")
