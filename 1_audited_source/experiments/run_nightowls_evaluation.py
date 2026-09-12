"""
NightOwls Cross-Dataset Evaluation
Tests trained models on NightOwls validation set for generalization

Federated learning pipeline
"""

import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
from ultralytics import YOLO


class NightOwlsEvaluator:
    """
    Cross-dataset evaluation on NightOwls.
    Tests models trained on ECP night-time data.
    """
    
    def __init__(
        self,
        nightowls_yaml: str = 'data/yolo/client_configs/nightowls_global.yaml',
        output_dir: str = 'results/nightowls_evaluation'
    ):
        """
        Initialize NightOwls evaluator.
        
        Args:
            nightowls_yaml: Path to NightOwls dataset config
            output_dir: Directory for results
        """
        self.nightowls_yaml = nightowls_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
        print("\n" + "="*60)
        print("NIGHTOWLS CROSS-DATASET EVALUATION")
        print("="*60)
        print(f"Dataset config: {nightowls_yaml}")
        print(f"Output: {output_dir}")
        print("="*60 + "\n")
    
    def evaluate_model(
        self,
        model_path: str,
        model_name: str,
        batch_size: int = 8,
        imgsz: int = 640,
        device: str = '0'
    ) -> Dict:
        """
        Evaluate a single model on NightOwls.
        
        Args:
            model_path: Path to trained model weights
            model_name: Name for logging
            batch_size: Evaluation batch size
            imgsz: Image size
            device: GPU device
            
        Returns:
            Evaluation metrics dict
        """
        print(f"\n--- Evaluating: {model_name} ---")
        print(f"Model: {model_path}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run validation on NightOwls
            results = model.val(
                data=self.nightowls_yaml,
                batch=batch_size,
                imgsz=imgsz,
                device=device,
                verbose=True,
                split='val'
            )
            
            metrics = {
                'model_name': model_name,
                'model_path': model_path,
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.p.mean()) if hasattr(results.box.p, 'mean') else float(results.box.p),
                'recall': float(results.box.r.mean()) if hasattr(results.box.r, 'mean') else float(results.box.r),
                'status': 'success'
            }
            
            print(f"{model_name}: mAP50={metrics['map50']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
            
        except Exception as e:
            print(f"Failed: {e}")
            metrics = {
                'model_name': model_name,
                'model_path': model_path,
                'error': str(e),
                'status': 'failed'
            }
        
        return metrics
    
    def evaluate_all_baselines(
        self,
        baseline_dir: str = 'runs/detect',
        model_names: List[str] = None
    ) -> List[Dict]:
        """
        Evaluate all baseline models.
        
        Args:
            baseline_dir: Directory containing trained models
            model_names: List of model names to evaluate
            
        Returns:
            List of evaluation results
        """
        if model_names is None:
            model_names = ['fedavg', 'fedbn', 'photoscreen']
        
        all_results = []
        
        for name in model_names:
            # Find best model for this baseline
            model_path = self._find_best_model(baseline_dir, name)
            
            if model_path:
                result = self.evaluate_model(
                    model_path=str(model_path),
                    model_name=name
                )
                all_results.append(result)
            else:
                print(f"No model found for {name}")
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _find_best_model(self, baseline_dir: str, name: str) -> Path:
        """Find best model weights for a baseline."""
        baseline_path = Path(baseline_dir)
        
        # Look for pattern: train*_{name}/weights/best.pt
        candidates = list(baseline_path.glob(f"*{name}*/weights/best.pt"))
        
        if not candidates:
            # Try without prefix
            candidates = list(baseline_path.glob(f"*/weights/best.pt"))
        
        if candidates:
            # Return most recent
            return sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
        
        return None
    
    def _save_results(self, results: List[Dict]):
        """Save evaluation results."""
        output_file = self.output_dir / 'nightowls_results.json'
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'num_models': len(results),
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        # Print summary table
        print("\n" + "="*60)
        print("NIGHTOWLS EVALUATION SUMMARY")
        print("="*60)
        print(f"{'Model':<15} {'mAP50':<10} {'Precision':<10} {'Recall':<10}")
        print("-"*45)
        
        for r in results:
            if r.get('status') == 'success':
                print(f"{r['model_name']:<15} {r['map50']:.4f}     {r['precision']:.4f}      {r['recall']:.4f}")
        
        print("="*60)


def evaluate_saved_models(
    model_paths: Dict[str, str],
    nightowls_yaml: str = 'data/nightowls_global.yaml',
    output_dir: str = 'results/nightowls_evaluation'
):
    """
    Evaluate specific saved model paths.
    
    Args:
        model_paths: Dict of {model_name: model_path}
        nightowls_yaml: Path to NightOwls config
        output_dir: Output directory
    """
    evaluator = NightOwlsEvaluator(
        nightowls_yaml=nightowls_yaml,
        output_dir=output_dir
    )
    
    all_results = []
    
    for name, path in model_paths.items():
        result = evaluator.evaluate_model(
            model_path=path,
            model_name=name
        )
        all_results.append(result)
    
    evaluator._save_results(all_results)
    
    return all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NightOwls Cross-Dataset Evaluation')
    parser.add_argument(
        '--nightowls-yaml',
        type=str,
        default='data/nightowls_global.yaml',
        help='Path to NightOwls dataset config'
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default='runs/detect',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/nightowls_evaluation',
        help='Output directory'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['fedavg', 'fedbn', 'photoscreen'],
        help='Model names to evaluate'
    )
    parser.add_argument(
        '--model-paths',
        nargs='+',
        default=None,
        help='Specific model paths (format: name:path)'
    )
    
    args = parser.parse_args()
    
    if args.model_paths:
        # Parse model paths
        model_paths = {}
        for item in args.model_paths:
            name, path = item.split(':')
            model_paths[name] = path
        
        evaluate_saved_models(
            model_paths=model_paths,
            nightowls_yaml=args.nightowls_yaml,
            output_dir=args.output
        )
    else:
        evaluator = NightOwlsEvaluator(
            nightowls_yaml=args.nightowls_yaml,
            output_dir=args.output
        )
        
        evaluator.evaluate_all_baselines(
            baseline_dir=args.baseline_dir,
            model_names=args.models
        )


if __name__ == '__main__':
    main()
