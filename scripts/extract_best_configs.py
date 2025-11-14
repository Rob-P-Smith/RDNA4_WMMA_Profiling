#!/usr/bin/env python3
"""
Extract the best performing configs from empirical test results.
Creates vLLM-compatible config files with optimal settings per batch size.
"""

import json
import os
from typing import Dict, List, Tuple

def load_results(filepath: str) -> Dict:
    """Load test results JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_config_name(config_name: str) -> Dict[str, int]:
    """Parse config name like 'M128_N32_K128_G8_W4_KP2' into parameters"""
    parts = config_name.split('_')
    return {
        'BLOCK_SIZE_M': int(parts[0][1:]),
        'BLOCK_SIZE_N': int(parts[1][1:]),
        'BLOCK_SIZE_K': int(parts[2][1:]),
        'GROUP_SIZE_M': int(parts[3][1:]),
        'num_warps': int(parts[4][1:]),
        'kpack': int(parts[5][2:]),
        'matrix_instr_nonkdim': 16  # Always 16 for RDNA4
    }

def find_best_config_for_batch(matrix_results: Dict, batch_size: str) -> Tuple[str, Dict, float]:
    """Find the best performing config for a specific batch size"""
    best_config_name = None
    best_tops = -1
    best_params = None

    for config_name, batch_data in matrix_results.items():
        if batch_size in batch_data:
            tops = batch_data[batch_size]['TOPS']
            if tops > best_tops:
                best_tops = tops
                best_config_name = config_name
                best_params = parse_config_name(config_name)

    return best_config_name, best_params, best_tops

def generate_optimal_config_file(results: Dict, matrix_key: str, output_dir: str):
    """Generate optimal config file for a matrix size based on empirical results"""

    # Parse matrix dimensions from key like "matrix_1024_1024"
    parts = matrix_key.split('_')
    N = int(parts[1])
    K = int(parts[2])

    matrix_results = results['results'][matrix_key]

    # vLLM expects this filename format
    filename = f"N={N},K={K},device_name=0x7551,dtype=fp8_w8a8,block_shape=[128,128].json"
    filepath = os.path.join(output_dir, filename)

    configs = {}
    batch_sizes = ['16', '32', '64', '128', '256', '512', '1024', '2048', '4096']

    print(f"\n{'='*80}")
    print(f"Extracting optimal configs for N={N}, K={K}")
    print(f"{'='*80}")

    for batch_size in batch_sizes:
        config_name, config_params, tops = find_best_config_for_batch(matrix_results, batch_size)

        if config_params:
            configs[batch_size] = config_params
            print(f"  Batch {batch_size:4s}: {config_name:35s} -> {tops:6.2f} TFLOPS")
        else:
            print(f"  Batch {batch_size:4s}: NO DATA")

    # Write config file
    with open(filepath, 'w') as f:
        json.dump(configs, f, indent=4)

    print(f"\n  Saved to: {filename}")

    return filepath

def main():
    results_file = "/mnt/share-drive/dev-vllm-experiment/resultscopy.json"
    output_dir = "/mnt/share-drive/dev-vllm-experiment/EmpiricalOptimalConfigs"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("EMPIRICAL OPTIMAL CONFIG EXTRACTOR")
    print("="*80)
    print(f"\nLoading results from: {results_file}")

    results = load_results(results_file)

    print(f"\nFound {len(results['results'])} matrix size(s) with test results")

    for matrix_key in results['results'].keys():
        generate_optimal_config_file(results, matrix_key, output_dir)

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nOptimal config files saved to:")
    print(f"  {output_dir}")
    print(f"\nThese configs are based on empirical test results:")
    print(f"  - Best TFLOPS performance for each batch size")
    print(f"  - Validated on RDNA4 hardware")
    print(f"  - Ready to use with vLLM")

if __name__ == "__main__":
    main()
