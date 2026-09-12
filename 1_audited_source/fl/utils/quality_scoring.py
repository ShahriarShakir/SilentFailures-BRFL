"""
QualityGate Quality Scoring Module
Illumination-Aware Robust Aggregation for Byzantine-resilient FL

Based on BN statistics as proxy for illumination quality.
Reference values from profiling: μ_ref=0.2942, σ_ref=0.1888
"""

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json


class IARAQualityScorer:
    """
    Computes client quality scores based on illumination statistics.
    
    Quality score (Eq. 3 from paper):
    q_k = sigmoid(β*(μ_k - μ_ref)) * min(σ_ref / (σ_k + ε), 2.0)
    
    Higher quality → Higher aggregation weight
    Lower quality → Downweighted (potentially malicious)
    """
    
    def __init__(
        self,
        mu_ref: float = 0.2942,
        sigma_ref: float = 0.1888,
        beta: float = 0.1,
        q_min: float = 0.2,
        config_path: str = None
    ):
        """
        Initialize QualityGate quality scorer.
        
        Args:
            mu_ref: Reference mean illumination (from profiling)
            sigma_ref: Reference std illumination
            beta: Sigmoid steepness parameter
            q_min: Minimum quality threshold (filter below this)
            config_path: Optional path to config JSON
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.mu_ref = config.get('mu_ref', mu_ref)
                self.sigma_ref = config.get('sigma_ref', sigma_ref)
                self.beta = config.get('beta', beta)
                self.q_min = config.get('q_min', q_min)
        else:
            self.mu_ref = mu_ref
            self.sigma_ref = sigma_ref
            self.beta = beta
            self.q_min = q_min
        
        print(f"QualityGate Quality Scorer initialized:")
        print(f"  μ_ref: {self.mu_ref:.4f}")
        print(f"  σ_ref: {self.sigma_ref:.4f}")
        print(f"  β: {self.beta:.4f}")
        print(f"  q_min: {self.q_min:.4f}")
    
    def compute_quality(
        self,
        mu: float,
        sigma: float
    ) -> float:
        """
        Compute quality score for single client.
        
        Args:
            mu: Client's mean BN statistic (brightness proxy)
            sigma: Client's std BN statistic (noise proxy)
            
        Returns:
            Quality score in [0, 2.0) — values above q_min (0.2) are accepted
        """
        # Brightness component (sigmoid)
        brightness_score = 1.0 / (1.0 + np.exp(-self.beta * (mu - self.mu_ref)))
        
        # Noise component (inverse ratio)
        noise_score = self.sigma_ref / (sigma + 1e-6)
        
        # Combined quality
        quality = brightness_score * min(noise_score, 2.0)  # Cap at 2x
        
        return float(quality)
    
    def compute_batch_quality(
        self,
        client_stats: Dict[int, Dict[str, float]]
    ) -> Dict[int, float]:
        """
        Compute quality scores for all clients.
        
        Args:
            client_stats: {client_id: {'mu': ..., 'sigma': ...}}
            
        Returns:
            {client_id: quality_score}
        """
        qualities = {}
        
        for cid, stats in client_stats.items():
            mu = stats.get('mu', self.mu_ref)
            sigma = stats.get('sigma', self.sigma_ref)
            qualities[cid] = self.compute_quality(mu, sigma)
        
        return qualities
    
    def filter_clients(
        self,
        client_stats: Dict[int, Dict[str, float]],
        return_weights: bool = True
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        Filter clients below quality threshold and compute weights.
        
        Args:
            client_stats: {client_id: {'mu': ..., 'sigma': ...}}
            return_weights: If True, return normalized weights
            
        Returns:
            (valid_client_ids, weights_dict)
        """
        # Compute qualities
        qualities = self.compute_batch_quality(client_stats)
        
        # Filter by threshold
        valid_clients = [
            cid for cid, q in qualities.items()
            if q >= self.q_min
        ]
        
        if not valid_clients:
            print(f" WARNING: No clients above q_min={self.q_min}! Using all.")
            valid_clients = list(qualities.keys())
        
        # Compute weights (normalized qualities)
        if return_weights:
            valid_qualities = {cid: qualities[cid] for cid in valid_clients}
            total_quality = sum(valid_qualities.values())
            weights = {
                cid: q / total_quality
                for cid, q in valid_qualities.items()
            }
        else:
            weights = {cid: 1.0 / len(valid_clients) for cid in valid_clients}
        
        # Log filtering
        filtered_count = len(qualities) - len(valid_clients)
        if filtered_count > 0:
            print(f"QualityGate filtered {filtered_count}/{len(qualities)} clients")
            filtered_ids = [cid for cid in qualities if cid not in valid_clients]
            for cid in filtered_ids:
                print(f"   Client {cid}: q={qualities[cid]:.4f} < {self.q_min}")
        
        return valid_clients, weights
    
    def get_aggregation_weights(
        self,
        client_stats: Dict[int, Dict[str, float]],
        num_samples: Dict[int, int] = None
    ) -> Dict[int, float]:
        """
        Get final aggregation weights (QualityGate quality × sample size).
        
        Args:
            client_stats: {client_id: {'mu': ..., 'sigma': ...}}
            num_samples: Optional {client_id: num_training_samples}
            
        Returns:
            {client_id: aggregation_weight}
        """
        # Filter and get quality weights
        valid_clients, quality_weights = self.filter_clients(client_stats)
        
        # If no sample sizes provided, use quality weights only
        if num_samples is None:
            return quality_weights
        
        # Combine with sample size (FedAvg-style)
        combined_weights = {}
        total_weighted_samples = 0.0
        
        for cid in valid_clients:
            samples = num_samples.get(cid, 1)
            combined_weights[cid] = quality_weights[cid] * samples
            total_weighted_samples += combined_weights[cid]
        
        # Normalize
        if total_weighted_samples > 0:
            combined_weights = {
                cid: w / total_weighted_samples
                for cid, w in combined_weights.items()
            }
        
        return combined_weights


def compute_bn_statistics(model, data_loader, device='cuda', max_batches=10):
    """
    Extract BN statistics from model during forward pass.
    
    Args:
        model: PyTorch model with BN layers
        data_loader: DataLoader for computing statistics
        device: Device to run on
        max_batches: Max batches to process
        
    Returns:
        {'mu': mean_brightness, 'sigma': std_brightness}
    """
    import torch
    
    model.eval()
    bn_means = []
    bn_stds = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            
            # Forward pass
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)
            
            # Extract BN statistics (simplified: use image statistics as proxy)
            # For full BN extraction, would need hooks on BN layers
            mean = images.mean().item()
            std = images.std().item()
            
            bn_means.append(mean)
            bn_stds.append(std)
    
    return {
        'mu': float(np.mean(bn_means)),
        'sigma': float(np.mean(bn_stds))
    }


# Singleton instance
_iara_scorer = None

def get_iara_scorer(config_path='data/bn_statistics/iara_config.json'):
    """Get or create QualityGate quality scorer singleton."""
    global _iara_scorer
    if _iara_scorer is None:
        _iara_scorer = IARAQualityScorer(config_path=config_path)
    return _iara_scorer
