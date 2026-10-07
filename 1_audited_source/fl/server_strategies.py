"""
Federated Learning Server Strategies
Implements: FedAvg, FedProx, FedBN, PhotoScreen

Supports both YOLOv10-S and RT-DETR-L architectures.
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
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from fl.utils.quality_scoring import IARAQualityScorer


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Standard FedAvg aggregation.
    Baseline for comparison.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("\n" + "="*60)
        print("FL STRATEGY: FedAvg (Baseline)")
        print("="*60)
        print("Simple weighted averaging by num_samples\n")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates with FedAvg."""
        
        if not results:
            return None, {}
        
        # Convert results to weights
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate (weighted average by num_samples)
        aggregated_weights = self._weighted_average(weights_results)
        
        # Collect metrics
        metrics = self._collect_metrics(results)
        
        print(f"\n[Round {server_round}] FedAvg aggregated {len(results)} clients")
        print(f"  Avg mAP50: {metrics.get('avg_map50', 0):.4f} | mAP50-95: {metrics.get('avg_map50_95', 0):.4f}")
        print(f"  Avg Precision: {metrics.get('avg_precision', 0):.4f} | Recall: {metrics.get('avg_recall', 0):.4f}")
        
        return ndarrays_to_parameters(aggregated_weights), metrics
    
    def _weighted_average(
        self,
        weights_results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Compute weighted average of parameters."""
        
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
        map50_95_list = []
        precision_list = []
        recall_list = []
        
        for _, fit_res in results:
            metrics = fit_res.metrics
            if 'map50' in metrics:
                map50_list.append(metrics['map50'])
            if 'map50_95' in metrics:
                map50_95_list.append(metrics['map50_95'])
            if 'precision' in metrics:
                precision_list.append(metrics['precision'])
            if 'recall' in metrics:
                recall_list.append(metrics['recall'])
        
        return {
            'avg_map50': np.mean(map50_list) if map50_list else 0.0,
            'avg_map50_95': np.mean(map50_95_list) if map50_95_list else 0.0,
            'avg_precision': np.mean(precision_list) if precision_list else 0.0,
            'avg_recall': np.mean(recall_list) if recall_list else 0.0,
            'num_clients': len(results)
        }


class FedBNStrategy(fl.server.strategy.FedAvg):
    """
    FedBN: Federated Learning with Batch Normalization Personalization.
    
    Key idea: Keep BN layers local (client-specific), only aggregate non-BN parameters.
    This helps handle heterogeneous data distributions (e.g., different illumination conditions).
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
        """Aggregate client updates (FedBN variant)."""
        
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
        
        num_examples_total = sum(num_examples for _, num_examples in weights_results)
        
        aggregated = [
            np.zeros_like(weights[0])
            for weights, _ in weights_results[:1]
        ]
        
        for i in range(len(aggregated)):
            for weights, num_examples in weights_results:
                aggregated[i] += weights[i] * (num_examples / num_examples_total)
        
        return aggregated
    
    def _collect_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict:
        """Collect and average metrics from clients."""
        
        map50_list = []
        map50_95_list = []
        precision_list = []
        recall_list = []
        bn_mu_list = []
        bn_sigma_list = []
        
        for _, fit_res in results:
            metrics = fit_res.metrics
            
            if 'map50' in metrics:
                map50_list.append(metrics['map50'])
            if 'map50_95' in metrics:
                map50_95_list.append(metrics['map50_95'])
            if 'precision' in metrics:
                precision_list.append(metrics['precision'])
            if 'recall' in metrics:
                recall_list.append(metrics['recall'])
            if 'bn_mu' in metrics:
                bn_mu_list.append(metrics['bn_mu'])
            if 'bn_sigma' in metrics:
                bn_sigma_list.append(metrics['bn_sigma'])
        
        metrics_dict = {
            'avg_map50': np.mean(map50_list) if map50_list else 0.0,
            'avg_map50_95': np.mean(map50_95_list) if map50_95_list else 0.0,
            'avg_precision': np.mean(precision_list) if precision_list else 0.0,
            'avg_recall': np.mean(recall_list) if recall_list else 0.0,
            'num_clients': len(results)
        }
        
        # Add BN statistics variance (measure of heterogeneity)
        if bn_mu_list:
            metrics_dict['bn_mu_variance'] = np.var(bn_mu_list)
        if bn_sigma_list:
            metrics_dict['bn_sigma_variance'] = np.var(bn_sigma_list)
        
        return metrics_dict


class PHOTOSCREENStrategy(fl.server.strategy.FedAvg):
    """
    PhotoScreen: Illumination-Aware Byzantine-Resilient Aggregation.
    
    Three-layer defense:
    1. QualityGate quality scoring (BN statistics)
    2. Gradient magnitude filtering (MAD)
    3. Semantic consistency check (cosine similarity)
    """
    
    def __init__(
        self,
        iara_config_path: str = 'data/bn_statistics/iara_config.json',
        enable_layer1: bool = True,  # QualityGate quality scoring
        enable_layer2: bool = True,  # Gradient MAD filtering
        enable_layer3: bool = True,  # Semantic check
        mad_threshold: float = 3.0,
        cosine_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.enable_layer1 = enable_layer1
        self.enable_layer2 = enable_layer2
        self.enable_layer3 = enable_layer3
        self.mad_threshold = mad_threshold
        self.cosine_threshold = cosine_threshold
        
        # Initialize QualityGate scorer
        if enable_layer1:
            self.iara_scorer = IARAQualityScorer(config_path=iara_config_path)
        else:
            self.iara_scorer = None
        
        print("\n" + "="*60)
        print("FL STRATEGY: PhotoScreen")
        print("="*60)
        print(f"Layer 1 (QualityGate Quality): {'' if enable_layer1 else ''}")
        print(f"Layer 2 (Gradient MAD): {'' if enable_layer2 else ''}")
        print(f"Layer 3 (Semantic Check): {'' if enable_layer3 else ''}")
        print("="*60 + "\n")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with PhotoScreen three-layer defense."""
        
        if not results:
            return None, {}
        
        print(f"\n[Round {server_round}] PhotoScreen processing {len(results)} clients...")
        
        # Extract client data
        client_weights = []
        client_samples = []
        client_stats = {}
        client_ids = []
        
        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples
            metrics = fit_res.metrics
            
            client_id = metrics.get('client_id', len(client_ids))
            
            client_weights.append(weights)
            client_samples.append(num_samples)
            client_ids.append(client_id)
            
            # Extract BN stats for QualityGate
            client_stats[client_id] = {
                'mu': metrics.get('bn_mu', 0.29),
                'sigma': metrics.get('bn_sigma', 0.19)
            }
        
        # Layer 1: QualityGate Quality Scoring
        if self.enable_layer1 and self.iara_scorer:
            print("   Layer 1: QualityGate Quality Filtering...")
            
            valid_clients, quality_weights = self.iara_scorer.filter_clients(client_stats)
            
            # Filter results
            filtered_weights = [client_weights[i] for i in range(len(client_ids)) if client_ids[i] in valid_clients]
            filtered_samples = [client_samples[i] for i in range(len(client_ids)) if client_ids[i] in valid_clients]
            
            print(f"     Retained: {len(valid_clients)}/{len(client_ids)} clients")
        else:
            filtered_weights = client_weights
            filtered_samples = client_samples
            quality_weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        
        # Layer 2: Gradient MAD Filtering
        if self.enable_layer2:
            print("   Layer 2: Gradient Magnitude Filtering...")
            filtered_weights, filtered_samples = self._filter_by_mad(
                filtered_weights, 
                filtered_samples,
                self.mad_threshold
            )
        
        # Layer 3: Semantic Consistency
        if self.enable_layer3:
            print("   Layer 3: Semantic Consistency Check...")
            filtered_weights, filtered_samples = self._filter_by_cosine(
                filtered_weights,
                filtered_samples,
                self.cosine_threshold
            )
        
        # Final aggregation
        weights_results = list(zip(filtered_weights, filtered_samples))
        aggregated_weights = self._weighted_average(weights_results)
        
        # Collect metrics
        metrics = self._collect_metrics(results)
        metrics['num_filtered'] = len(client_ids) - len(filtered_weights)
        
        print(f"  Aggregation complete: {len(filtered_weights)}/{len(client_ids)} clients")
        print(f"     Avg mAP50: {metrics.get('avg_map50', 0):.4f}")
        
        return ndarrays_to_parameters(aggregated_weights), metrics
    
    def _filter_by_mad(
        self,
        weights_list: List[List[np.ndarray]],
        samples_list: List[int],
        threshold: float = 3.0
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        """Filter clients by gradient magnitude (MAD)."""
        
        # Compute norms
        norms = []
        for weights in weights_list:
            norm = np.sqrt(sum(np.sum(w**2) for w in weights))
            norms.append(norm)
        
        norms = np.array(norms)
        
        # MAD-based outlier detection
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))
        
        # Filter outliers
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
        
        valid_indices = [
            i for i, norm in enumerate(norms)
            if lower_bound <= norm <= upper_bound
        ]
        
        filtered_weights = [weights_list[i] for i in valid_indices]
        filtered_samples = [samples_list[i] for i in valid_indices]
        
        filtered_count = len(weights_list) - len(filtered_weights)
        if filtered_count > 0:
            print(f"     Filtered {filtered_count} outliers (MAD threshold={threshold})")
        
        return filtered_weights, filtered_samples
    
    def _filter_by_cosine(
        self,
        weights_list: List[List[np.ndarray]],
        samples_list: List[int],
        threshold: float = 0.5
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        """Filter by cosine similarity to mean update."""
        
        if len(weights_list) < 2:
            return weights_list, samples_list
        
        # Flatten weights for cosine similarity
        flat_weights = [
            np.concatenate([w.flatten() for w in weights])
            for weights in weights_list
        ]

        # Clip to minimum common length: detection heads may vary in shape
        # across per-city clients; backbone params are homogeneous and sufficient
        # for cosine comparison.
        min_len = min(len(f) for f in flat_weights)
        flat_weights_cmp = [f[:min_len] for f in flat_weights]

        # Compute mean update
        flat_array = np.stack(flat_weights_cmp)
        mean_update = flat_array.mean(axis=0)

        # Cosine similarities
        similarities = []
        for flat_w in flat_weights_cmp:
            sim = np.dot(flat_w, mean_update) / (
                np.linalg.norm(flat_w) * np.linalg.norm(mean_update) + 1e-8
            )
            similarities.append(sim)
        
        # Filter by threshold
        valid_indices = [
            i for i, sim in enumerate(similarities)
            if sim >= threshold
        ]
        
        filtered_weights = [weights_list[i] for i in valid_indices]
        filtered_samples = [samples_list[i] for i in valid_indices]
        
        filtered_count = len(weights_list) - len(filtered_weights)
        if filtered_count > 0:
            print(f"     Filtered {filtered_count} semantically inconsistent clients")
        
        return filtered_weights, filtered_samples
    
    def _weighted_average(
        self,
        weights_results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Compute weighted average."""
        
        num_examples_total = sum(num_examples for _, num_examples in weights_results)
        
        aggregated = [
            np.zeros_like(weights[0])
            for weights, _ in weights_results[:1]
        ]
        
        for i in range(len(aggregated)):
            for weights, num_examples in weights_results:
                aggregated[i] += weights[i] * (num_examples / num_examples_total)
        
        return aggregated
    
    def _collect_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict:
        """Collect metrics."""
        
        map50_list = []
        map50_95_list = []
        precision_list = []
        recall_list = []
        
        for _, fit_res in results:
            metrics = fit_res.metrics
            if 'map50' in metrics:
                map50_list.append(metrics['map50'])
            if 'map50_95' in metrics:
                map50_95_list.append(metrics['map50_95'])
            if 'precision' in metrics:
                precision_list.append(metrics['precision'])
            if 'recall' in metrics:
                recall_list.append(metrics['recall'])
        
        return {
            'avg_map50': np.mean(map50_list) if map50_list else 0.0,
            'avg_map50_95': np.mean(map50_95_list) if map50_95_list else 0.0,
            'avg_precision': np.mean(precision_list) if precision_list else 0.0,
            'avg_recall': np.mean(recall_list) if recall_list else 0.0,
            'num_clients': len(results)
        }
