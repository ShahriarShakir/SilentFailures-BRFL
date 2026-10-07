#!/usr/bin/env python3
"""Parse mAP50 values from Byzantine experiment logs."""
import re
import json
import sys
from pathlib import Path

def parse_log_for_map50(log_file: str) -> dict:
    """Parse log file to extract mAP50 values per experiment."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find all experiments and their mAP50 values
    experiments = {}
    current_exp = None
    map50_values = []
    
    for line in content.split('\n'):
        # Match experiment start
        exp_match = re.match(r'EXPERIMENT: (\S+)', line)
        if exp_match:
            # Save previous experiment
            if current_exp and map50_values:
                experiments[current_exp] = {
                    'map50_history': map50_values,
                    'final_map50': map50_values[-1] if map50_values else 0.0
                }
            current_exp = exp_match.group(1)
            map50_values = []
        
        # Match Avg mAP50 values (these are the aggregated per-round values)
        map50_match = re.search(r'Avg mAP50: ([\d.]+)', line)
        if map50_match and current_exp:
            map50_values.append(float(map50_match.group(1)))
    
    # Save last experiment
    if current_exp and map50_values:
        experiments[current_exp] = {
            'map50_history': map50_values,
            'final_map50': map50_values[-1] if map50_values else 0.0
        }
    
    return experiments

def update_results_json(results_file: str, parsed_metrics: dict):
    """Update results JSON with parsed mAP50 values."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    for result in data['results']:
        exp_id = result['experiment_id']
        if exp_id in parsed_metrics:
            result['final_map50'] = parsed_metrics[exp_id]['final_map50']
            result['map50_history'] = parsed_metrics[exp_id]['map50_history']
    
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'sanity_test2.log'
    results_file = sys.argv[2] if len(sys.argv) > 2 else 'results/sanity_test/byzantine_results.json'
    
    print(f"Parsing log file: {log_file}")
    parsed = parse_log_for_map50(log_file)
    
    print(f"\nFound {len(parsed)} experiments:")
    for exp_id, metrics in parsed.items():
        print(f"  {exp_id}: final_map50={metrics['final_map50']:.4f}, rounds={len(metrics['map50_history'])}")
    
    if Path(results_file).exists():
        print(f"\nUpdating results file: {results_file}")
        updated = update_results_json(results_file, parsed)
        print("Results updated successfully!")
    else:
        print(f"Results file not found: {results_file}")
