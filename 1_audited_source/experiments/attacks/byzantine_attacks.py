"""
Byzantine Attack Implementations for FL
Testing PhotoScreen robustness against illumination-based attacks.

Attack categories:
1. Data poisoning: Manipulate local data
2. Model poisoning: Manipulate gradients/parameters
3. Hybrid: Both data + model manipulation
"""

import numpy as np
import torch
from typing import List, Dict
from abc import ABC, abstractmethod


class ByzantineAttack(ABC):
    """Base class for Byzantine attacks."""
    
    def __init__(self, attack_ratio: float = 0.3, intensity: float = 1.0):
        """
        Args:
            attack_ratio: Fraction of clients to compromise (α)
            intensity: Attack strength multiplier
        """
        self.attack_ratio = attack_ratio
        self.intensity = intensity
        self.name = self.__class__.__name__
    
    @abstractmethod
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Poison training data (data poisoning attacks)."""
        pass
    
    @abstractmethod
    def poison_parameters(
        self,
        parameters: List[np.ndarray],
        global_params: List[np.ndarray] = None
    ) -> List[np.ndarray]:
        """Poison model parameters (model poisoning attacks)."""
        pass
    
    def is_malicious_client(self, client_id: int, total_clients: int) -> bool:
        """Determine if client should be malicious."""
        num_malicious = int(total_clients * self.attack_ratio)
        return client_id < num_malicious


class BrightnessFloodAttack(ByzantineAttack):
    """
    Attack 1: Brightness Flooding
    
    Increase image brightness to maximum, simulating overexposed images.
    Targets: Night-time detectors trained on dark scenes.
    Goal: Degrade model performance on normal night images.
    """
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Flood images with brightness."""
        # Increase brightness dramatically
        poisoned_images = np.clip(
            images + self.intensity * 0.5,  # Add 0.5 brightness
            0.0, 1.0
        )
        return poisoned_images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        """No parameter poisoning for this attack."""
        return parameters


class DarknessInjectionAttack(ByzantineAttack):
    """
    Attack 2: Darkness Injection
    
    Decrease image brightness to near-zero, simulating pitch-black scenes.
    Targets: Models expected to handle low-light conditions.
    Goal: Force model to overfit to extremely dark scenes.
    """
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Inject extreme darkness."""
        poisoned_images = np.clip(
            images - self.intensity * 0.3,  # Subtract brightness
            0.0, 1.0
        )
        return poisoned_images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        return parameters


class NoiseStormAttack(ByzantineAttack):
    """
    Attack 3: Noise Storm
    
    Add Gaussian noise to images, degrading visual quality.
    Targets: All detectors (general disruption).
    Goal: Reduce model's ability to extract clean features.
    """
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.intensity * 0.1, images.shape)
        poisoned_images = np.clip(images + noise, 0.0, 1.0)
        return poisoned_images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        return parameters


class SignFlippingAttack(ByzantineAttack):
    """
    Attack 4: Sign Flipping (Gradient Reversal)
    
    Flip signs of gradients to push model in opposite direction.
    Targets: Gradient-based aggregation (FedAvg, FedBN).
    Goal: Diverge global model from convergence.
    """
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        return images, labels  # No data poisoning
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        """Flip gradient direction - scaled to avoid overflow."""
        if global_params is None or len(global_params) == 0:
            # Round-1 / no-global case: scale-flip in place (preserves shapes).
            return [p * (-self.intensity * 0.5 + 0.5) for p in parameters]
        else:
            # Flip gradient: instead of (p - g), send (g - (p - g)) = 2g - p
            # Scale to prevent explosion: g + scale * (g - p)
            scale = min(self.intensity, 1.0)  # Cap intensity
            poisoned = []
            for p, g in zip(parameters, global_params):
                delta = p - g  # Local update
                # Reverse the update direction
                poisoned_param = g - scale * delta
                # Clip to prevent extreme values
                poisoned_param = np.clip(poisoned_param, -10.0, 10.0)
                poisoned.append(poisoned_param)
            return poisoned


class ModelReplacementAttack(ByzantineAttack):
    """
    Attack 5: Model Replacement
    
    Replace local model with scaled random noise.
    Targets: All FL methods.
    Goal: Introduce random noise into aggregation.
    """
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        return images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        """Replace with noise scaled to parameter magnitude."""
        poisoned = []
        for p in parameters:
            # Scale noise to match parameter magnitude
            param_std = np.std(p) + 1e-8
            noise = np.random.randn(*p.shape).astype(p.dtype) * param_std * self.intensity
            # Clip to prevent extreme values
            poisoned_param = np.clip(noise, -10.0, 10.0)
            poisoned.append(poisoned_param)
        return poisoned


class TargetedFalseNegativeAttack(ByzantineAttack):
    """
    Attack 6: Targeted False Negative
    
    Remove labels for specific class (e.g., pedestrians).
    Targets: Safety-critical applications.
    Goal: Make model miss pedestrians (catastrophic for AV).
    """
    
    def __init__(self, target_class: int = 0, **kwargs):
        """
        Args:
            target_class: Class to suppress (0 = pedestrian)
        """
        super().__init__(**kwargs)
        self.target_class = target_class
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Remove target class labels."""
        # Filter out target class
        mask = labels[:, 0] != self.target_class  # Assuming labels are [class, x, y, w, h]
        poisoned_labels = labels[mask]
        return images, poisoned_labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        return parameters


class FeaturePoisoningAttack(ByzantineAttack):
    """
    Attack 7: Feature Poisoning
    
    Add adversarial perturbations to features.
    Targets: Feature extractors.
    Goal: Degrade feature quality.
    """
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Add adversarial noise (FGSM-like)."""
        # Simple approximation: Add high-frequency noise
        noise_pattern = np.sin(images * 30) * self.intensity * 0.05
        poisoned_images = np.clip(images + noise_pattern, 0.0, 1.0)
        return poisoned_images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        return parameters


class BackdoorAttack(ByzantineAttack):
    """
    Attack 8: Backdoor Trigger
    
    Insert trigger pattern in corner, associate with wrong label.
    Targets: All models.
    Goal: Trigger-activated misclassification.
    """
    
    def __init__(self, trigger_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.trigger_size = trigger_size
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Add trigger pattern (white square in corner)."""
        poisoned_images = images.copy()
        
        # Add trigger (top-right corner)
        h, w = images.shape[-2:]
        poisoned_images[..., :self.trigger_size, -self.trigger_size:] = 1.0
        
        # Optional: Flip labels (for classification tasks)
        # poisoned_labels = (labels + 1) % num_classes  # Shift labels
        
        return poisoned_images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        return parameters


class QualityAwareEvasionAttack(ByzantineAttack):
    """
    Attack 9: Quality-Aware Evasion
    
    Manipulate BN statistics to pass QualityGate quality checks while poisoning.
    Targets: QualityGate defense specifically.
    Goal: Evade quality scoring while maintaining attack effectiveness.
    """
    
    def __init__(self, target_mu: float = 0.29, target_sigma: float = 0.19, **kwargs):
        """
        Args:
            target_mu: Target BN mean (to mimic honest client)
            target_sigma: Target BN std
        """
        super().__init__(**kwargs)
        self.target_mu = target_mu
        self.target_sigma = target_sigma
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Normalize images to target BN statistics."""
        # Adjust brightness to match target μ
        current_mu = np.mean(images)
        images_adjusted = images + (self.target_mu - current_mu)
        
        # Adjust variance to match target σ
        current_sigma = np.std(images_adjusted)
        if current_sigma > 0:
            images_adjusted = (images_adjusted - np.mean(images_adjusted)) * (self.target_sigma / current_sigma) + self.target_mu
        
        # Add subtle poison
        poisoned_images = np.clip(images_adjusted + np.random.normal(0, 0.01, images.shape), 0, 1)
        
        return poisoned_images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        """Subtle parameter perturbation with clipping."""
        poisoned = []
        for p in parameters:
            noise = np.random.normal(0, self.intensity * 0.001, p.shape).astype(p.dtype)
            poisoned_param = np.clip(p + noise, -10.0, 10.0)
            poisoned.append(poisoned_param)
        return poisoned


class MultiRoundConsistencyAttack(ByzantineAttack):
    """
    Attack 10: Multi-Round Consistency
    
    Gradually poison model over multiple rounds (stealth attack).
    Targets: Temporal consistency checks.
    Goal: Avoid detection by slowly shifting statistics.
    """
    
    def __init__(self, max_rounds: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.max_rounds = max_rounds
        self.current_round = 0
    
    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        """Gradually increase poison intensity."""
        # Linear ramp-up
        effective_intensity = self.intensity * (self.current_round / self.max_rounds)
        
        poisoned_images = np.clip(
            images + effective_intensity * 0.3,
            0.0, 1.0
        )
        
        self.current_round += 1
        return poisoned_images, labels
    
    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        return parameters


# Attack factory
ATTACK_REGISTRY = {
    'brightness_flood': BrightnessFloodAttack,
    'darkness_injection': DarknessInjectionAttack,
    'noise_storm': NoiseStormAttack,
    'sign_flipping': SignFlippingAttack,
    'model_replacement': ModelReplacementAttack,
    'targeted_fn': TargetedFalseNegativeAttack,
    'feature_poisoning': FeaturePoisoningAttack,
    'backdoor': BackdoorAttack,
    'quality_aware_evasion': QualityAwareEvasionAttack,
    'multi_round_consistency': MultiRoundConsistencyAttack,
}


# ---------------------------------------------------------------------------
# Adaptive attack.
# Goal: bypass QualityGate Layer 1 by reporting BN statistics that match the
# illumination reference profile exactly, while still flipping the sign of the
# gradient update. PhotoScreen's Layers 2 (MAD) and 3 (cosine) must catch this.
# ---------------------------------------------------------------------------
class IlluminationSpoofAttack(ByzantineAttack):
    """Brightness-matched, sign-flipped gradient attack.

 Reports `bn_mu = mu_ref` and `bn_sigma = sigma_ref` (so QualityGate gives this
 client weight ~ 1.0) but the parameter update is sign-flipped and scaled
 by `intensity`. The reported BN values are picked up via the metric channel
 in `client_yolo10s.py` -- see the `attack` config block in
 `supplementary_experiments/configs/adaptive_illumination_spoof.yaml`.
 """

    def __init__(
        self,
        attack_ratio: float = 0.30,
        intensity: float = 1.0,
        mu_ref: float = 0.2942,
        sigma_ref: float = 0.1888,
    ):
        super().__init__(attack_ratio=attack_ratio, intensity=intensity)
        self.mu_ref = mu_ref
        self.sigma_ref = sigma_ref
        self.spoofed_metrics = {"bn_mu": mu_ref, "bn_sigma": sigma_ref}

    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        # No data poisoning -- the attack is purely model-side.
        return images, labels

    def poison_parameters(
        self,
        parameters: List[np.ndarray],
        global_params: List[np.ndarray] = None,
    ) -> List[np.ndarray]:
        # Sign flip relative to current update; if global is given, flip the delta.
        if global_params is None or len(global_params) == 0:
            return [-self.intensity * p for p in parameters]
        out = []
        for p, g in zip(parameters, global_params):
            delta = p - g
            out.append(g - self.intensity * delta)
        return out


ATTACK_REGISTRY['illumination_spoof'] = IlluminationSpoofAttack


# ---------------------------------------------------------------------------
# Photometric Byzantine Threats (PBT) for follow-up work.
#
# These attacks are designed to stress the new research claim: an attacker can
# appear benign in illumination statistics, update norm, and update direction
# while still pushing the detector toward photometric failure modes.
# ---------------------------------------------------------------------------
def _safe_delta(parameters: List[np.ndarray], global_params: List[np.ndarray] = None) -> List[np.ndarray]:
    """Return local update deltas, using zeros if no global parameters exist."""
    if global_params is None or len(global_params) == 0:
        return [np.zeros_like(p) for p in parameters]
    return [p - g for p, g in zip(parameters, global_params)]


def _l2_norm(arrays: List[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in arrays)))


def _scale_delta(delta: List[np.ndarray], target_norm: float) -> List[np.ndarray]:
    norm = _l2_norm(delta)
    if norm <= 1e-12:
        return delta
    scale = target_norm / norm
    return [d * scale for d in delta]


def _apply_delta(global_params: List[np.ndarray], delta: List[np.ndarray]) -> List[np.ndarray]:
    return [np.clip(g + d, -10.0, 10.0).astype(g.dtype, copy=False) for g, d in zip(global_params, delta)]


class DirectionAlignedPhotometricAttack(ByzantineAttack):
    """Keep update direction benign-looking while injecting a small harmful residual.

    This approximates the PBT setting where cosine/direction filters are not
    enough: the malicious update remains positively aligned with the client's
    real update, then adds a bounded residual concentrated on high-dimensional
    parameters.
    """

    def __init__(
        self,
        attack_ratio: float = 0.30,
        intensity: float = 1.0,
        alignment: float = 0.90,
        residual_scale: float = 0.15,
        mu_ref: float = 0.2942,
        sigma_ref: float = 0.1888,
    ):
        super().__init__(attack_ratio=attack_ratio, intensity=intensity)
        self.alignment = alignment
        self.residual_scale = residual_scale
        self.spoofed_metrics = {"bn_mu": mu_ref, "bn_sigma": sigma_ref}

    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        return images, labels

    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        if global_params is None or len(global_params) == 0:
            return parameters
        delta = _safe_delta(parameters, global_params)
        delta_norm = _l2_norm(delta)
        poisoned_delta = []
        rng = np.random.default_rng(20260618)
        for d in delta:
            noise = rng.normal(0.0, 1.0, d.shape).astype(d.dtype)
            noise = noise * (np.std(d) + 1e-8)
            # Positive aligned component plus subtle residual; cosine should
            # remain high for small residual_scale.
            pbt_delta = self.alignment * d + self.residual_scale * self.intensity * noise
            poisoned_delta.append(pbt_delta)
        poisoned_delta = _scale_delta(poisoned_delta, delta_norm)
        return _apply_delta(global_params, poisoned_delta)


class MagnitudeBoundedPhotometricAttack(ByzantineAttack):
    """Project a malicious residual into the benign update-norm envelope."""

    def __init__(
        self,
        attack_ratio: float = 0.30,
        intensity: float = 1.0,
        norm_ratio: float = 1.0,
        mu_ref: float = 0.2942,
        sigma_ref: float = 0.1888,
    ):
        super().__init__(attack_ratio=attack_ratio, intensity=intensity)
        self.norm_ratio = norm_ratio
        self.spoofed_metrics = {"bn_mu": mu_ref, "bn_sigma": sigma_ref}

    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        return images, labels

    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        if global_params is None or len(global_params) == 0:
            return parameters
        delta = _safe_delta(parameters, global_params)
        target_norm = max(_l2_norm(delta) * self.norm_ratio, 1e-12)
        malicious_delta = [-self.intensity * d for d in delta]
        malicious_delta = _scale_delta(malicious_delta, target_norm)
        return _apply_delta(global_params, malicious_delta)


class LowLightSuppressionAttack(ByzantineAttack):
    """Photometric data poison plus bounded head perturbation for dark misses."""

    def __init__(
        self,
        attack_ratio: float = 0.30,
        intensity: float = 1.0,
        dark_factor: float = 0.35,
        head_fraction: float = 0.20,
        mu_ref: float = 0.2942,
        sigma_ref: float = 0.1888,
    ):
        super().__init__(attack_ratio=attack_ratio, intensity=intensity)
        self.dark_factor = dark_factor
        self.head_fraction = head_fraction
        self.spoofed_metrics = {"bn_mu": mu_ref, "bn_sigma": sigma_ref}

    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        poisoned_images = np.clip(images * self.dark_factor, 0.0, 1.0)
        return poisoned_images, labels

    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        if global_params is None or len(global_params) == 0:
            return parameters
        poisoned = [p.copy() for p in parameters]
        start = int(max(0, len(poisoned) * (1.0 - self.head_fraction)))
        for i in range(start, len(poisoned)):
            delta = poisoned[i] - global_params[i]
            poisoned[i] = np.clip(global_params[i] - self.intensity * 0.5 * delta, -10.0, 10.0)
        return poisoned


class GlareTriggeredMissAttack(ByzantineAttack):
    """Add a synthetic glare trigger and apply a bounded model residual."""

    def __init__(
        self,
        attack_ratio: float = 0.30,
        intensity: float = 1.0,
        trigger_alpha: float = 0.65,
        trigger_size: int = 32,
        mu_ref: float = 0.2942,
        sigma_ref: float = 0.1888,
    ):
        super().__init__(attack_ratio=attack_ratio, intensity=intensity)
        self.trigger_alpha = trigger_alpha
        self.trigger_size = trigger_size
        self.spoofed_metrics = {"bn_mu": mu_ref, "bn_sigma": sigma_ref}

    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        poisoned = images.copy()
        h, w = poisoned.shape[-2:]
        size = min(self.trigger_size, h, w)
        yy, xx = np.ogrid[:size, :size]
        center = (size - 1) / 2.0
        radius = max(size / 2.0, 1.0)
        mask = np.clip(1.0 - ((yy - center) ** 2 + (xx - center) ** 2) / (radius ** 2), 0.0, 1.0)
        patch = self.trigger_alpha * mask.astype(poisoned.dtype)
        poisoned[..., :size, -size:] = np.clip(poisoned[..., :size, -size:] + patch, 0.0, 1.0)
        return poisoned, labels

    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        if global_params is None or len(global_params) == 0:
            return parameters
        delta = _safe_delta(parameters, global_params)
        target_norm = _l2_norm(delta)
        rng = np.random.default_rng(20260619)
        residual = [
            rng.normal(0.0, np.std(d) + 1e-8, d.shape).astype(d.dtype)
            for d in delta
        ]
        residual = _scale_delta(residual, target_norm * min(0.5, self.intensity * 0.2))
        poisoned_delta = [d + r for d, r in zip(delta, residual)]
        poisoned_delta = _scale_delta(poisoned_delta, target_norm)
        return _apply_delta(global_params, poisoned_delta)


class CombinedPhotometricByzantineAttack(ByzantineAttack):
    """Killshot PBT attack: spoof illumination, preserve norm, align direction."""

    def __init__(
        self,
        attack_ratio: float = 0.30,
        intensity: float = 1.0,
        alignment: float = 0.90,
        norm_ratio: float = 1.0,
        residual_scale: float = 0.20,
        mu_ref: float = 0.2942,
        sigma_ref: float = 0.1888,
    ):
        super().__init__(attack_ratio=attack_ratio, intensity=intensity)
        self.alignment = alignment
        self.norm_ratio = norm_ratio
        self.residual_scale = residual_scale
        self.spoofed_metrics = {"bn_mu": mu_ref, "bn_sigma": sigma_ref}

    def poison_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        dark = np.clip(images * 0.55, 0.0, 1.0)
        return dark, labels

    def poison_parameters(self, parameters: List[np.ndarray], global_params=None) -> List[np.ndarray]:
        if global_params is None or len(global_params) == 0:
            return parameters
        delta = _safe_delta(parameters, global_params)
        target_norm = max(_l2_norm(delta) * self.norm_ratio, 1e-12)
        rng = np.random.default_rng(20260620)
        poisoned_delta = []
        for d in delta:
            residual = rng.normal(0.0, np.std(d) + 1e-8, d.shape).astype(d.dtype)
            # Positive direction alignment plus bounded destructive residual.
            poisoned_delta.append(self.alignment * d + self.residual_scale * self.intensity * residual)
        poisoned_delta = _scale_delta(poisoned_delta, target_norm)
        return _apply_delta(global_params, poisoned_delta)


ATTACK_REGISTRY.update({
    'direction_aligned_photometric': DirectionAlignedPhotometricAttack,
    'magnitude_bounded_photometric': MagnitudeBoundedPhotometricAttack,
    'low_light_suppression': LowLightSuppressionAttack,
    'glare_triggered_miss': GlareTriggeredMissAttack,
    'combined_photometric_byzantine': CombinedPhotometricByzantineAttack,
})


def create_attack(attack_name: str, **kwargs) -> ByzantineAttack:
    """
    Factory function to create attack instances.
    
    Args:
        attack_name: Name of attack (see ATTACK_REGISTRY)
        **kwargs: Attack-specific parameters
    
    Returns:
        Attack instance
    """
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}. Available: {list(ATTACK_REGISTRY.keys())}")
    
    return ATTACK_REGISTRY[attack_name](**kwargs)
