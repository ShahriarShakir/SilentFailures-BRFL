"""
Flower FL Client for YOLOv10-S
Wraps Ultralytics YOLO for federated learning.
"""

import flwr as fl
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import shutil
import tempfile
import os
import threading
import fcntl  # For file locking on Linux

# Global lock for model file operations
_model_copy_lock = threading.Lock()

# Fix: Monkey-patch Ultralytics to make cache fully per-worker
# 
# ROOT CAUSE ANALYSIS (Jan 8, 2026 - Post-vacation discovery):
# 1. YOLO's get_labels() ALWAYS tries to load/create label cache files (train.cache, val.cache)
# 2. The cache=False parameter only controls IMAGE caching, NOT label caching!
# 3. Multiple Ray workers access the SAME cache file simultaneously
# 4. Race condition: Worker A writes cache, Worker B reads partial data → _pickle.UnpicklingError
#
# PREVIOUS FIX v2 FLAW (Dec 21, 2025):
# - We patched ultralytics.data.utils.load_dataset_cache_file
# - BUT ultralytics.data.dataset IMPORTS this function at MODULE LOAD TIME
# - The dataset module keeps a reference to the ORIGINAL function!
# - Our patch only affects new imports of data_utils, not the already-imported reference
#
# SOLUTION (v3):
# - Patch BOTH ultralytics.data.utils AND ultralytics.data.dataset modules
# - Each module has its own reference that needs to be patched
# - Use file locking + per-worker /tmp isolation
def _patch_ultralytics_cache():
    """Patch Ultralytics cache in ALL modules that import it."""
    try:
        from ultralytics.data import utils as data_utils
        from ultralytics.data import dataset as dataset_module
        import numpy as np
        import fcntl
        
        def per_worker_cache_path(path):
            """Convert cache path to per-worker unique path in /tmp."""
            path = Path(path)
            # Use /tmp for per-worker caches - completely isolated from shared data dir
            worker_cache_dir = Path(f"/tmp/yolo_cache_{os.getpid()}")
            worker_cache_dir.mkdir(parents=True, exist_ok=True)
            # Include original path hash to avoid collisions between datasets
            path_hash = abs(hash(str(path.parent))) % 10000
            new_name = f"{path.stem}_{path_hash}_{os.getpid()}{path.suffix}"
            return worker_cache_dir / new_name
        
        def safe_save_cache(prefix, path, x, version):
            """Save cache to per-worker unique file with file locking."""
            worker_path = per_worker_cache_path(path)
            x["version"] = version
            
            # Write to temporary file first, then atomic rename
            temp_path = worker_path.with_suffix('.tmp')
            try:
                with open(str(temp_path), "wb") as file:
                    # Get exclusive lock
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                    np.save(file, x)
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                # Atomic rename (POSIX guarantees this is atomic)
                temp_path.replace(worker_path)
                # Reduce log spam - only log occasionally
            except Exception as e:
                print(f"{prefix}Cache save failed: {e}")
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
        
        def safe_load_cache(path):
            """Load cache from per-worker unique file ONLY - never use shared cache."""
            worker_path = per_worker_cache_path(path)
            
            # Note: Do not copy from the shared cache.
            # If per-worker cache doesn't exist, force recreation
            if not worker_path.exists():
                # Raise FileNotFoundError to trigger cache_labels() recreation
                raise FileNotFoundError(f"Per-worker cache not found: {worker_path}")
            
            # Load per-worker cache with file locking
            try:
                with open(str(worker_path), "rb") as file:
                    # Get shared lock for reading
                    fcntl.flock(file.fileno(), fcntl.LOCK_SH)
                    cache = np.load(file, allow_pickle=True).item()
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                return cache
            except Exception as e:
                print(f"[Cache] Load failed for {worker_path.name}: {e}")
                # If load fails, delete corrupted cache
                if worker_path.exists():
                    try:
                        worker_path.unlink()
                    except:
                        pass
                raise FileNotFoundError(f"Cache corrupted: {worker_path}")
        
        # Apply patches to BOTH modules - THIS IS THE KEY FIX!
        # 1. Patch the utils module (for new imports)
        data_utils.save_dataset_cache_file = safe_save_cache
        data_utils.load_dataset_cache_file = safe_load_cache
        
        # 2. Patch the dataset module's already-imported references
        # This is the critical fix - dataset.py imports at module load time
        dataset_module.save_dataset_cache_file = safe_save_cache
        dataset_module.load_dataset_cache_file = safe_load_cache
        
        print(f"[Client PID={os.getpid()}] Ultralytics cache patched v3 (utils + dataset modules)")
        
    except Exception as e:
        print(f"[Client] Warning: Could not patch Ultralytics cache: {e}")
        import traceback
        traceback.print_exc()

# Apply patch at module load time
_patch_ultralytics_cache()


class YOLOv10Client(fl.client.NumPyClient):
    """
    Federated Learning client for YOLOv10-S.
    
    Supports:
    - FedAvg: Aggregate all parameters
    - FedBN: Aggregate non-BN parameters only
    - PhotoScreen: Extract BN statistics for quality scoring
    """
    
    def __init__(
        self,
        client_id: int,
        data_yaml: str,
        model_path: str = 'yolov10s.pt',
        local_epochs: int = 5,
        batch_size: int = 12,
        imgsz: int = 1280,
        device: str = '0',
        fedbn: bool = False,
        extract_bn_stats: bool = True,
        seed: int = 0
    ):
        """
        Initialize YOLOv10 FL client.
        
        Args:
            client_id: Unique client identifier
            data_yaml: Path to client's dataset YAML
            model_path: Path to model weights
            local_epochs: Epochs per FL round
            batch_size: Training batch size
            imgsz: Image size
            device: GPU device
            fedbn: If True, don't aggregate BN params
            extract_bn_stats: If True, compute BN stats for QualityGate
        """
        self.client_id = client_id
        self.data_yaml = data_yaml
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = device
        self.fedbn = fedbn
        self.extract_bn_stats = extract_bn_stats
        self.seed = int(seed)
        
        # Fix: Create a per-worker copy of the model to avoid 
        # race conditions when multiple Ray workers load simultaneously
        print(f"\n[Client {client_id}] Initializing YOLOv10-S...")
        
        # Create client-specific model file to prevent corruption
        model_dir = Path(tempfile.gettempdir()) / "fl_yolo_models"
        model_dir.mkdir(exist_ok=True)
        
        worker_model_path = model_dir / f"yolov10s_client_{client_id}_{os.getpid()}.pt"
        
        # Use threading lock to safely copy the model (within same process)
        # Different workers (different PIDs) can copy in parallel safely
        with _model_copy_lock:
            if not worker_model_path.exists():
                shutil.copy(model_path, worker_model_path)
        
        self.model = YOLO(str(worker_model_path))
        self._worker_model_path = worker_model_path  # Keep reference for cleanup
        
        # BN statistics cache
        self.bn_stats = {'mu': 0.0, 'sigma': 0.0}
        
        print(f"[Client {client_id}] Ready")
        print(f"  Data: {data_yaml}")
        print(f"  Local epochs: {local_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  FedBN: {fedbn}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Extract model parameters for aggregation.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of parameter arrays
        """
        params = []
        
        for name, param in self.model.model.named_parameters():
            # Skip BN params if FedBN enabled
            if self.fedbn and ('bn' in name.lower() or 'norm' in name.lower()):
                continue
            
            params.append(param.detach().cpu().numpy())
        
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load aggregated parameters from server.
        
        Args:
            parameters: List of parameter arrays
        """
        if not parameters:
            return
        
        param_dict = {}
        param_idx = 0
        skipped = 0
        
        for name, param in self.model.model.named_parameters():
            # Skip BN params if FedBN enabled (keep local BN)
            if self.fedbn and ('bn' in name.lower() or 'norm' in name.lower()):
                continue
            
            # Safety check
            if param_idx >= len(parameters):
                print(f"WARNING: Not enough parameters. Expected {param_idx+1}, got {len(parameters)}")
                break
            
            # Defensive: if the incoming tensor doesn't match this layer's
            # expected size, keep the local tensor (graceful per-layer
            # skip instead of crashing the whole evaluate phase). This
            # matters for the per-city YOLOv10 detection heads whose
            # shapes vary across clients.
            incoming = parameters[param_idx]
            param_shape = param.shape
            expected_size = int(np.prod(param_shape))
            if int(np.asarray(incoming).size) != expected_size:
                skipped += 1
                param_idx += 1
                continue
            
            param_dict[name] = torch.from_numpy(incoming).reshape(param_shape)
            param_idx += 1
        
        if skipped:
            print(f"  set_parameters: kept local copies of {skipped} layer(s) with shape mismatch")
        
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
        
        # Per-client deterministic seeding 
        # Different across clients but reproducible across runs.
        eff_seed = int(self.seed) * 1000 + int(self.client_id)
        try:
            import torch as _torch
            _torch.manual_seed(eff_seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(eff_seed)
        except Exception:
            pass
        np.random.seed(eff_seed)
        
        print(f"\n[Client {self.client_id}] Starting local training (seed={eff_seed})...")
        
        # Train locally
        results = self.model.train(
            data=self.data_yaml,
            epochs=self.local_epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            amp=True,  # FP16
            optimizer='AdamW',
            lr0=0.001,
            verbose=False,  # Quiet
            plots=False,
            val=True,
            save=False,  # Note: Don't save weights - prevents corruption with parallel workers
            project=f"/tmp/fl_yolo_runs_{os.getpid()}",  # Per-worker run directory
            cache=False,  # Note: Disable caching to prevent corruption with Ray workers
            seed=eff_seed  # Pass to ultralytics for inner-loop reproducibility
        )
        
        # Extract metrics
        metrics_dict = results.results_dict
        num_examples = config.get('num_train_samples', 100)  # Placeholder
        
        # Extract BN statistics for QualityGate
        if self.extract_bn_stats:
            self.bn_stats = self._extract_bn_stats()
        
        metrics = {
            'map50': float(metrics_dict.get('metrics/mAP50(B)', 0)),
            'map50_95': float(metrics_dict.get('metrics/mAP50-95(B)', 0)),
            'precision': float(metrics_dict.get('metrics/precision(B)', 0)),
            'recall': float(metrics_dict.get('metrics/recall(B)', 0)),
            'bn_mu': self.bn_stats['mu'],
            'bn_sigma': self.bn_stats['sigma'],
            'client_id': self.client_id
        }
        
        print(f"[Client {self.client_id}] Training complete:")
        print(f"  mAP50: {metrics['map50']:.4f}, mAP50-95: {metrics['map50_95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"  BN Stats: μ={metrics['bn_mu']:.4f}, σ={metrics['bn_sigma']:.4f}")
        
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
        # Load parameters (only if provided)
        if parameters:
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
        
        loss = 1.0 - metrics['map50']  # Inverse metric as loss
        num_examples = config.get('num_val_samples', 50)
        
        return loss, num_examples, metrics
    
    def _extract_bn_stats(self) -> Dict[str, float]:
        """
        Extract auditable photometric statistics as the illumination proxy.

        Earlier runs used a random placeholder here. For the new
        PBT project we need deterministic, inspectable statistics. We keep the
        legacy metric names (`bn_mu`, `bn_sigma`) for strategy compatibility,
        but the values are image-derived photometric mean/std from the client's
        configured train/val images.
        
        Returns:
            {'mu': mean_intensity, 'sigma': intensity_std}
        """
        try:
            import yaml
            from PIL import Image

            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)

            root = Path(data_config.get('path', Path(self.data_yaml).parent))
            image_dirs = []
            for key in ('val', 'train'):
                rel = data_config.get(key)
                if not rel:
                    continue
                path = Path(rel)
                image_dirs.append(path if path.is_absolute() else root / path)

            image_paths = []
            for image_dir in image_dirs:
                if image_dir.is_file():
                    image_paths.append(image_dir)
                    continue
                if not image_dir.exists():
                    continue
                for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                    image_paths.extend(sorted(image_dir.rglob(pattern)))

            if not image_paths:
                raise FileNotFoundError(f"No images found from YAML: {self.data_yaml}")

            # Deterministic subsample to cap overhead and keep stats reproducible.
            max_images = 32
            image_paths = image_paths[:max_images]
            means = []
            stds = []
            for image_path in image_paths:
                with Image.open(image_path) as img:
                    arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
                means.append(float(arr.mean()))
                stds.append(float(arr.std()))

            return {
                'mu': float(np.mean(means)),
                'sigma': float(np.mean(stds)),
            }

        except Exception as e:
            print(f" Could not extract photometric stats: {e}")
            return {'mu': 0.29, 'sigma': 0.19}  # Compatibility fallback


def create_yolo_client_fn(
    client_configs: Dict[int, Dict],
    fedbn: bool = False,
    extract_bn_stats: bool = True,
    attack = None,
    malicious_clients: List[int] = None,
    seed: int = 0
):
    """
    Factory function to create FL client instances.
    
    Args:
        client_configs: {client_id: {'data_yaml': ..., 'model_path': ...}}
        fedbn: Enable FedBN mode
        extract_bn_stats: Enable QualityGate stats extraction
        attack: Byzantine attack instance (optional)
        malicious_clients: List of client IDs that are malicious (optional)
        
    Returns:
        client_fn(cid) -> YOLOv10Client
    """
    malicious_clients = malicious_clients or []
    
    def client_fn(cid: str) -> YOLOv10Client:
        """Create client instance."""
        client_id = int(cid)
        config = client_configs[client_id]
        
        # Determine if this client is malicious
        is_malicious = client_id in malicious_clients
        client_attack = attack if is_malicious else None
        
        return YOLOv10ClientWithAttack(
            client_id=client_id,
            data_yaml=config['data_yaml'],
            model_path=config.get('model_path', 'yolov10s.pt'),
            local_epochs=config.get('local_epochs', 5),
            batch_size=config.get('batch_size', 12),
            imgsz=config.get('imgsz', 1280),
            device=config.get('device', '0'),
            fedbn=fedbn,
            extract_bn_stats=extract_bn_stats,
            attack=client_attack,
            seed=seed
        )
    
    return client_fn


class YOLOv10ClientWithAttack(YOLOv10Client):
    """
    YOLOv10 Client with Byzantine attack support.
    Extends base client to apply attacks during training.
    """
    
    def __init__(
        self,
        attack = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attack = attack
        self.is_malicious = attack is not None
        
        if self.is_malicious:
            print(f"[Client {self.client_id}]  MALICIOUS - Attack: {self.attack.name}")
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train locally with optional Byzantine attack.
        """
        # Standard training first
        updated_params, num_examples, metrics = super().fit(parameters, config)
        
        # Apply model poisoning attack if malicious
        if self.is_malicious and self.attack:
            print(f"[Client {self.client_id}] Applying {self.attack.name} attack...")
            
            # Poison the parameters before sending to server
            updated_params = self.attack.poison_parameters(
                parameters=updated_params,
                global_params=parameters
            )

            # Inject spoofed BN metrics if the attack provides them (e.g. IlluminationSpoofAttack).
            # Without this, QualityGate reads real bn_mu/bn_sigma and the spoof never reaches the server.
            if hasattr(self.attack, 'spoofed_metrics') and self.attack.spoofed_metrics:
                metrics.update(self.attack.spoofed_metrics)
                print(f"[Client {self.client_id}] Metrics spoofed: {self.attack.spoofed_metrics}")

            print(f"[Client {self.client_id}] Attack applied")
        
        return updated_params, num_examples, metrics
