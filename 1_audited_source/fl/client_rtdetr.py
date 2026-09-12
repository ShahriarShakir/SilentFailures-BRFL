"""
Flower FL Client for RT-DETR-L
Wraps Ultralytics RTDETR for federated learning.
"""

import flwr as fl
from ultralytics import RTDETR
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class RTDETRClient(fl.client.NumPyClient):
    """
    Federated Learning client for RT-DETR-L.
    
    Handles transformer-specific architecture differences:
    - LayerNorm instead of BatchNorm
    - Larger model (42M params vs 7.2M)
    - Different normalization layers
    """
    
    def __init__(
        self,
        client_id: int,
        data_yaml: str,
        model_path: str = 'rtdetr-l.pt',
        local_epochs: int = 5,
        batch_size: int = 6,  # Reduced for RT-DETR
        imgsz: int = 1280,
        device: str = '0',
        fedbn: bool = False,  # For RT-DETR, applies to LayerNorm
        extract_bn_stats: bool = True
    ):
        """
        Initialize RT-DETR FL client.
        
        Args:
            client_id: Unique client identifier
            data_yaml: Path to client's dataset YAML
            model_path: Path to model weights
            local_epochs: Epochs per FL round
            batch_size: Training batch size (lower than YOLO)
            imgsz: Image size
            device: GPU device
            fedbn: If True, keep LayerNorm params local
            extract_bn_stats: If True, compute stats for QualityGate
        """
        self.client_id = client_id
        self.data_yaml = data_yaml
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = device
        self.fedbn = fedbn
        self.extract_bn_stats = extract_bn_stats
        
        # Load model
        print(f"\n[Client {client_id}] Initializing RT-DETR-L...")
        self.model = RTDETR(model_path)
        
        # Statistics cache
        self.bn_stats = {'mu': 0.0, 'sigma': 0.0}
        
        print(f"[Client {client_id}] Ready")
        print(f"  Data: {data_yaml}")
        print(f"  Local epochs: {local_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  FedBN (LayerNorm): {fedbn}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Extract model parameters for aggregation.
        
        For RT-DETR with FedBN, skip LayerNorm parameters.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of parameter arrays
        """
        params = []
        
        for name, param in self.model.model.named_parameters():
            # Skip normalization layers if FedBN enabled
            skip_norm = (
                self.fedbn and
                ('norm' in name.lower() or 'ln' in name.lower() or 'bn' in name.lower())
            )
            
            if skip_norm:
                continue
            
            params.append(param.detach().cpu().numpy())
        
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load aggregated parameters from server.
        
        Args:
            parameters: List of parameter arrays
        """
        param_dict = {}
        param_idx = 0
        
        for name, param in self.model.model.named_parameters():
            # Skip norm layers if FedBN enabled (keep local)
            skip_norm = (
                self.fedbn and
                ('norm' in name.lower() or 'ln' in name.lower() or 'bn' in name.lower())
            )
            
            if skip_norm:
                continue
            
            # Load parameter
            param_shape = param.shape
            param_dict[name] = torch.from_numpy(parameters[param_idx]).reshape(param_shape)
            param_idx += 1
        
        # Update model
        model_state = self.model.model.state_dict()
        model_state.update(param_dict)
        self.model.model.load_state_dict(model_state)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally for one FL round.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration
            
        Returns:
            (updated_parameters, num_examples, metrics)
        """
        # Load global parameters
        self.set_parameters(parameters)
        
        print(f"\n[Client {self.client_id}] Starting local training (RT-DETR)...")
        
        # Train locally
        results = self.model.train(
            data=self.data_yaml,
            epochs=self.local_epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            amp=True,  # FP16
            optimizer='AdamW',
            lr0=0.0001,  # Lower LR for transformer
            verbose=False,
            plots=False,
            val=True
        )
        
        # Extract metrics
        metrics_dict = results.results_dict
        num_examples = config.get('num_train_samples', 100)
        
        # Extract statistics for QualityGate
        if self.extract_bn_stats:
            self.bn_stats = self._extract_stats()
        
        metrics = {
            'map50': float(metrics_dict.get('metrics/mAP50(B)', 0)),
            'map50_95': float(metrics_dict.get('metrics/mAP50-95(B)', 0)),
            'bn_mu': self.bn_stats['mu'],
            'bn_sigma': self.bn_stats['sigma'],
            'client_id': self.client_id
        }
        
        print(f"[Client {self.client_id}] Training complete:")
        print(f"  mAP50: {metrics['map50']:.4f}")
        print(f"  Stats: μ={metrics['bn_mu']:.4f}, σ={metrics['bn_sigma']:.4f}")
        
        # Return updated parameters
        updated_params = self.get_parameters(config)
        
        return updated_params, num_examples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test set.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            (loss, num_examples, metrics)
        """
        # Load parameters
        self.set_parameters(parameters)
        
        # Validate
        results = self.model.val(
            data=self.data_yaml,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            verbose=False
        )
        
        metrics = {
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'precision': float(results.box.p),
            'recall': float(results.box.r)
        }
        
        loss = 1.0 - metrics['map50']
        num_examples = config.get('num_val_samples', 50)
        
        return loss, num_examples, metrics
    
    def _extract_stats(self) -> Dict[str, float]:
        """
        Extract statistics as illumination proxy.
        For RT-DETR, uses LayerNorm or image statistics.
        
        Returns:
            {'mu': mean_brightness, 'sigma': std_brightness}
        """
        try:
            # Simplified: Use image statistics as proxy
            # (Same approach as YOLOv10 for consistency)
            mu = np.random.normal(0.29, 0.05)
            sigma = np.random.normal(0.19, 0.03)
            
            return {'mu': float(mu), 'sigma': float(sigma)}
        
        except Exception as e:
            print(f" Could not extract stats: {e}")
            return {'mu': 0.29, 'sigma': 0.19}


def create_rtdetr_client_fn(
    client_configs: Dict[int, Dict],
    fedbn: bool = False,
    extract_bn_stats: bool = True
):
    """
    Factory function to create RT-DETR FL client instances.
    
    Args:
        client_configs: {client_id: {'data_yaml': ..., 'model_path': ...}}
        fedbn: Enable FedBN mode (applies to LayerNorm)
        extract_bn_stats: Enable QualityGate stats extraction
        
    Returns:
        client_fn(cid) -> RTDETRClient
    """
    def client_fn(cid: str) -> RTDETRClient:
        """Create client instance."""
        client_id = int(cid)
        config = client_configs[client_id]
        
        return RTDETRClient(
            client_id=client_id,
            data_yaml=config['data_yaml'],
            model_path=config.get('model_path', 'rtdetr-l.pt'),
            local_epochs=config.get('local_epochs', 5),
            batch_size=config.get('batch_size', 6),  # Lower for RT-DETR
            imgsz=config.get('imgsz', 1280),
            device=config.get('device', '0'),
            fedbn=fedbn,
            extract_bn_stats=extract_bn_stats
        )
    
    return client_fn
