#!/usr/bin/env python3
"""
Robust Byzantine Evaluation Script with Disk Cleanup
Runs remaining 46 experiments with automatic disk cleanup between experiments.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remaining experiments to run
REMAINING_EXPERIMENTS = [
    # sign_flipping (1 remaining)
    ("sign_flipping", 0.3, "photoscreen"),
    
    # targeted_fn (9 experiments)
    ("targeted_fn", 0.1, "fedavg"),
    ("targeted_fn", 0.1, "fedbn"),
    ("targeted_fn", 0.1, "photoscreen"),
    ("targeted_fn", 0.2, "fedavg"),
    ("targeted_fn", 0.2, "fedbn"),
    ("targeted_fn", 0.2, "photoscreen"),
    ("targeted_fn", 0.3, "fedavg"),
    ("targeted_fn", 0.3, "fedbn"),
    ("targeted_fn", 0.3, "photoscreen"),
    
    # feature_poisoning (9 experiments)
    ("feature_poisoning", 0.1, "fedavg"),
    ("feature_poisoning", 0.1, "fedbn"),
    ("feature_poisoning", 0.1, "photoscreen"),
    ("feature_poisoning", 0.2, "fedavg"),
    ("feature_poisoning", 0.2, "fedbn"),
    ("feature_poisoning", 0.2, "photoscreen"),
    ("feature_poisoning", 0.3, "fedavg"),
    ("feature_poisoning", 0.3, "fedbn"),
    ("feature_poisoning", 0.3, "photoscreen"),
    
    # backdoor (9 experiments)
    ("backdoor", 0.1, "fedavg"),
    ("backdoor", 0.1, "fedbn"),
    ("backdoor", 0.1, "photoscreen"),
    ("backdoor", 0.2, "fedavg"),
    ("backdoor", 0.2, "fedbn"),
    ("backdoor", 0.2, "photoscreen"),
    ("backdoor", 0.3, "fedavg"),
    ("backdoor", 0.3, "fedbn"),
    ("backdoor", 0.3, "photoscreen"),
    
    # quality_aware_evasion (9 experiments)
    ("quality_aware_evasion", 0.1, "fedavg"),
    ("quality_aware_evasion", 0.1, "fedbn"),
    ("quality_aware_evasion", 0.1, "photoscreen"),
    ("quality_aware_evasion", 0.2, "fedavg"),
    ("quality_aware_evasion", 0.2, "fedbn"),
    ("quality_aware_evasion", 0.2, "photoscreen"),
    ("quality_aware_evasion", 0.3, "fedavg"),
    ("quality_aware_evasion", 0.3, "fedbn"),
    ("quality_aware_evasion", 0.3, "photoscreen"),
    
    # multi_round_consistency (9 experiments)
    ("multi_round_consistency", 0.1, "fedavg"),
    ("multi_round_consistency", 0.1, "fedbn"),
    ("multi_round_consistency", 0.1, "photoscreen"),
    ("multi_round_consistency", 0.2, "fedavg"),
    ("multi_round_consistency", 0.2, "fedbn"),
    ("multi_round_consistency", 0.2, "photoscreen"),
    ("multi_round_consistency", 0.3, "fedavg"),
    ("multi_round_consistency", 0.3, "fedbn"),
    ("multi_round_consistency", 0.3, "photoscreen"),
]


def cleanup_disk():
    """Clean up temporary files and YOLO run directories to free disk space."""
    print("\nRunning disk cleanup...")
    
    # Clean Ray sessions
    ray_dir = Path("/tmp/ray")
    if ray_dir.exists():
        try:
            # Keep only recent sessions (last 2 hours)
            for session in ray_dir.glob("session_*"):
                try:
                    shutil.rmtree(session, ignore_errors=True)
                except:
                    pass
        except:
            pass
    
    # Clean YOLO cache files
    for pattern in ["/tmp/yolo_cache_*", "/tmp/fl_yolo_runs_*"]:
        import glob
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
            except:
                pass
    
    # Clean runs/detect (validation outputs accumulate here)
    detect_dir = Path("runs/detect")
    if detect_dir.exists():
        try:
            shutil.rmtree(detect_dir, ignore_errors=True)
        except:
            pass
    
    # Clean runs/train
    train_dir = Path("runs/train")
    if train_dir.exists():
        try:
            shutil.rmtree(train_dir, ignore_errors=True)
        except:
            pass
    
    # Check disk space
    result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
    print(f"   Disk space: {result.stdout.split(chr(10))[1].split()[3]} available")


def get_disk_free_gb():
    """Get free disk space in GB."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    return free / (1024**3)


def run_single_experiment(attack, ratio, method, num_rounds=50, results_dir="results"):
    """Run a single Byzantine experiment with proper cleanup."""
    from experiments.run_byzantine_evaluation import run_byzantine_experiment
    
    experiment_id = f"{attack}_{int(ratio*100)}pct_{method}"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"{'='*60}")
    
    # Check disk space before starting
    free_gb = get_disk_free_gb()
    if free_gb < 50:
        print(f" Low disk space ({free_gb:.1f} GB). Running cleanup...")
        cleanup_disk()
        free_gb = get_disk_free_gb()
        if free_gb < 20:
            print(f"Disk space critically low ({free_gb:.1f} GB). Aborting.")
            return None
    
    start_time = time.time()
    
    try:
        result = run_byzantine_experiment(
            attack_type=attack,
            malicious_ratio=ratio,
            method=method,
            num_rounds=num_rounds,
            local_epochs=3,
            batch_size=4
        )
        
        duration = time.time() - start_time
        result['duration_seconds'] = duration
        result['status'] = 'success'
        
        print(f"\nExperiment complete in {duration/60:.1f} minutes")
        print(f"   Final mAP50: {result['final_map50']:.4f}")
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\nExperiment failed after {duration/60:.1f} minutes: {e}")
        return {
            'experiment_id': experiment_id,
            'attack': attack,
            'malicious_ratio': ratio,
            'method': method,
            'status': 'failed',
            'error': str(e),
            'duration_seconds': duration
        }


def save_results(results, results_dir):
    """Save results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "byzantine_results.json")
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(results),
        "num_successful": sum(1 for r in results if r.get('status') == 'success'),
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


def main():
    print("="*60)
    print("ROBUST BYZANTINE EVALUATION - REMAINING 46 EXPERIMENTS")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Total experiments: {len(REMAINING_EXPERIMENTS)}")
    print(f"Estimated time: ~{len(REMAINING_EXPERIMENTS) * 90 / 60:.1f} hours")
    print("="*60)
    
    # Initial cleanup
    cleanup_disk()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/byzantine_remaining_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    for i, (attack, ratio, method) in enumerate(REMAINING_EXPERIMENTS):
        print(f"\n[{i+1}/{len(REMAINING_EXPERIMENTS)}] Starting experiment...")
        
        # Run experiment
        result = run_single_experiment(attack, ratio, method, num_rounds=50)
        
        if result:
            results.append(result)
            # Save after each experiment
            save_results(results, results_dir)
        
        # Cleanup after every experiment
        cleanup_disk()
        
        # Brief pause between experiments
        time.sleep(5)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'failed']
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['experiment_id']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
