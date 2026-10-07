#!/usr/bin/env python3
"""
Run remaining 47 Byzantine experiments with disk cleanup.
Batch runner for the remaining experiments.

This script:
1. Runs the remaining 47 experiments one by one
2. Cleans disk space between experiments
3. Handles errors gracefully and continues
4. Saves results after each experiment
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 47 remaining experiments as of Jan 13, 2026
REMAINING_EXPERIMENTS = [
    ('sign_flipping', 0.3, 'fedbn'),
    ('sign_flipping', 0.3, 'photoscreen'),
    ('targeted_fn', 0.1, 'fedavg'),
    ('targeted_fn', 0.1, 'fedbn'),
    ('targeted_fn', 0.1, 'photoscreen'),
    ('targeted_fn', 0.2, 'fedavg'),
    ('targeted_fn', 0.2, 'fedbn'),
    ('targeted_fn', 0.2, 'photoscreen'),
    ('targeted_fn', 0.3, 'fedavg'),
    ('targeted_fn', 0.3, 'fedbn'),
    ('targeted_fn', 0.3, 'photoscreen'),
    ('feature_poisoning', 0.1, 'fedavg'),
    ('feature_poisoning', 0.1, 'fedbn'),
    ('feature_poisoning', 0.1, 'photoscreen'),
    ('feature_poisoning', 0.2, 'fedavg'),
    ('feature_poisoning', 0.2, 'fedbn'),
    ('feature_poisoning', 0.2, 'photoscreen'),
    ('feature_poisoning', 0.3, 'fedavg'),
    ('feature_poisoning', 0.3, 'fedbn'),
    ('feature_poisoning', 0.3, 'photoscreen'),
    ('backdoor', 0.1, 'fedavg'),
    ('backdoor', 0.1, 'fedbn'),
    ('backdoor', 0.1, 'photoscreen'),
    ('backdoor', 0.2, 'fedavg'),
    ('backdoor', 0.2, 'fedbn'),
    ('backdoor', 0.2, 'photoscreen'),
    ('backdoor', 0.3, 'fedavg'),
    ('backdoor', 0.3, 'fedbn'),
    ('backdoor', 0.3, 'photoscreen'),
    ('quality_aware_evasion', 0.1, 'fedavg'),
    ('quality_aware_evasion', 0.1, 'fedbn'),
    ('quality_aware_evasion', 0.1, 'photoscreen'),
    ('quality_aware_evasion', 0.2, 'fedavg'),
    ('quality_aware_evasion', 0.2, 'fedbn'),
    ('quality_aware_evasion', 0.2, 'photoscreen'),
    ('quality_aware_evasion', 0.3, 'fedavg'),
    ('quality_aware_evasion', 0.3, 'fedbn'),
    ('quality_aware_evasion', 0.3, 'photoscreen'),
    ('multi_round_consistency', 0.1, 'fedavg'),
    ('multi_round_consistency', 0.1, 'fedbn'),
    ('multi_round_consistency', 0.1, 'photoscreen'),
    ('multi_round_consistency', 0.2, 'fedavg'),
    ('multi_round_consistency', 0.2, 'fedbn'),
    ('multi_round_consistency', 0.2, 'photoscreen'),
    ('multi_round_consistency', 0.3, 'fedavg'),
    ('multi_round_consistency', 0.3, 'fedbn'),
    ('multi_round_consistency', 0.3, 'photoscreen'),
]


def get_disk_free_gb():
    """Get free disk space in GB."""
    stat = os.statvfs(str(PROJECT_ROOT))
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


def cleanup_disk():
    """Clean up disk space between experiments."""
    print("\nCleaning up disk space...")
    
    cleaned = 0
    
    # 1. Stop Ray and clean Ray temp
    try:
        subprocess.run(['ray', 'stop', '--force'], 
                      capture_output=True, timeout=30)
    except:
        pass
    
    # Clean /tmp/ray
    for path in Path('/tmp').glob('ray*'):
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                cleaned += 1
        except:
            pass
    
    # 2. Clean YOLO cache directories
    for pattern in ['yolo_cache_*', 'fl_yolo_runs_*']:
        for path in Path('/tmp').glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    cleaned += 1
            except:
                pass
    
    # 3. Clean runs/detect (YOLO validation outputs)
    runs_detect = PROJECT_ROOT / 'runs' / 'detect'
    if runs_detect.exists():
        try:
            shutil.rmtree(runs_detect, ignore_errors=True)
            cleaned += 1
        except:
            pass
    
    # 4. Clean runs/train
    runs_train = PROJECT_ROOT / 'runs' / 'train'
    if runs_train.exists():
        try:
            shutil.rmtree(runs_train, ignore_errors=True)
            cleaned += 1
        except:
            pass
    
    # 5. Clean __pycache__
    for pycache in PROJECT_ROOT.rglob('__pycache__'):
        try:
            shutil.rmtree(pycache, ignore_errors=True)
        except:
            pass
    
    free_gb = get_disk_free_gb()
    print(f"   Cleaned {cleaned} directories. Free disk: {free_gb:.1f} GB")
    
    return free_gb


def run_experiment(attack, ratio, method, output_dir, num_rounds=50):
    """Run a single experiment using ByzantineExperimentRunner."""
    from experiments.run_byzantine_evaluation import ByzantineExperimentRunner
    
    runner = ByzantineExperimentRunner(
        base_dir=PROJECT_ROOT,
        output_dir=output_dir,
        num_rounds=num_rounds
    )
    
    result = runner.run_single_experiment(attack, ratio, method)
    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("ROBUST BYZANTINE EVALUATION - REMAINING 47 EXPERIMENTS")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Total experiments: {len(REMAINING_EXPERIMENTS)}")
    print(f"Estimated time: ~{len(REMAINING_EXPERIMENTS) * 1.5:.1f} hours")
    print("=" * 70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = PROJECT_ROOT / 'results' / f'byzantine_remaining_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'byzantine_results.json'
    
    # Track progress
    all_results = []
    completed_count = 0
    failed_count = 0
    
    for idx, (attack, ratio, method) in enumerate(REMAINING_EXPERIMENTS, 1):
        exp_id = f"{attack}_{int(ratio*100)}pct_{method}"
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(REMAINING_EXPERIMENTS)}] {exp_id}")
        print(f"{'='*70}")
        
        # Clean disk before each experiment
        free_gb = cleanup_disk()
        
        # Check disk space (minimum 20GB required)
        if free_gb < 20:
            print(f"CRITICAL: Low disk space ({free_gb:.1f} GB). Stopping.")
            break
        
        # Run experiment
        try:
            start_time = time.time()
            result = run_experiment(attack, ratio, method, output_dir)
            duration = time.time() - start_time
            
            all_results.append(result)
            
            if result.get('status') == 'success':
                completed_count += 1
                print(f"Completed in {duration/60:.1f} min")
                fm = result.get('final_metrics', {})
                print(f"   mAP50: {result.get('final_map50', 0):.4f}")
                print(f"   mAP50-95: {fm.get('mAP50_95', 0):.4f}")
                print(f"   Precision: {fm.get('precision', 0):.4f}")
                print(f"   Recall: {fm.get('recall', 0):.4f}")
            else:
                failed_count += 1
                print(f"Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            failed_count += 1
            print(f"Exception: {e}")
            all_results.append({
                'experiment_id': exp_id,
                'attack': attack,
                'malicious_ratio': ratio,
                'method': method,
                'error': str(e),
                'status': 'failed'
            })
        
        # Save results after each experiment
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(REMAINING_EXPERIMENTS),
                'completed': completed_count,
                'failed': failed_count,
                'results': all_results
            }, f, indent=2)
        
        print(f"Results saved ({completed_count} completed, {failed_count} failed)")
        
        # Estimate remaining time
        remaining = len(REMAINING_EXPERIMENTS) - idx
        if idx > 0:
            avg_time = (time.time() - start_time) if idx == 1 else 90 * 60  # ~90 min
            eta_hours = (remaining * 90) / 60
            print(f"⏱Remaining: {remaining} experiments (~{eta_hours:.1f} hours)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total completed: {completed_count}/{len(REMAINING_EXPERIMENTS)}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to: {results_file}")
    print(f"End time: {datetime.now()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
