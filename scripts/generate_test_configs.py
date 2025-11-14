#!/usr/bin/env python3
"""
Generate all valid FP8 WMMA config files for testing
Creates config files for each matrix size with all valid parameter combinations
"""

import json
import os
import itertools
from typing import Dict, List, Tuple

# All matrix sizes to test (RDNA4 + MI350)
MATRIX_SIZES = [
    # RDNA4 specific sizes
    (1024, 1024),   # K, V projections
    (2048, 1024),   # Q projection
    (3072, 1024),   # Gate, Up projections
    (1024, 3072),   # Down projection
    (1024, 2048),   # O projection (transposed)
    (512, 512),
    (1024, 512),
    (512, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    # MI350 sizes
    (1024, 8192),
    (2112, 7168),
    (3072, 1536),
    (32768, 8192),
    (4096, 7168),
    (4608, 7168),
    (512, 7168),
    (7168, 2048),
    (7168, 256),
    (8192, 1024),
    (8192, 32768),
]

# Batch sizes to optimize for
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Parameter ranges
BLOCK_SIZES = [16, 32, 64, 128]  # Must be multiple of 16 for RDNA4
GROUP_SIZES_M = [1, 2, 4, 8, 16, 32]
NUM_WARPS = [2, 4, 8]
MATRIX_INSTR_NONKDIM = 16  # Fixed for RDNA4
KPACK_VALUES = [1, 2, 4]  # K-dimension packing factor

def is_valid_config(config: Dict, batch_size: int) -> bool:
    """Check if a config is valid for the given batch size"""

    # Block sizes must be at least 16 (RDNA4 WMMA minimum)
    if config["BLOCK_SIZE_M"] < 16 or config["BLOCK_SIZE_N"] < 16 or config["BLOCK_SIZE_K"] < 16:
        return False

    # GROUP_SIZE_M must divide batch size evenly
    if batch_size % config["GROUP_SIZE_M"] != 0:
        return False

    # GROUP_SIZE_M cannot exceed BLOCK_SIZE_M
    if config["GROUP_SIZE_M"] > config["BLOCK_SIZE_M"]:
        return False

    # Block sizes should be powers of 2
    for size in [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]]:
        if size & (size - 1) != 0:  # Check if power of 2
            return False

    return True

def generate_configs_for_batch_size(batch_size: int) -> List[Dict]:
    """Generate all valid configs for a specific batch size"""
    configs = []

    for bs_m, bs_n, bs_k, gs_m, nw, kp in itertools.product(
        BLOCK_SIZES, BLOCK_SIZES, BLOCK_SIZES,
        GROUP_SIZES_M, NUM_WARPS, KPACK_VALUES
    ):
        config = {
            "BLOCK_SIZE_M": bs_m,
            "BLOCK_SIZE_N": bs_n,
            "BLOCK_SIZE_K": bs_k,
            "GROUP_SIZE_M": gs_m,
            "num_warps": nw,
            "matrix_instr_nonkdim": MATRIX_INSTR_NONKDIM,
            "kpack": kp
        }

        if is_valid_config(config, batch_size):
            configs.append(config)

    return configs

def generate_config_id(config: Dict) -> str:
    """Generate a unique identifier for a config"""
    return f"M{config['BLOCK_SIZE_M']}_N{config['BLOCK_SIZE_N']}_K{config['BLOCK_SIZE_K']}_G{config['GROUP_SIZE_M']}_W{config['num_warps']}_KP{config['kpack']}"

def write_config_file(matrix_size: Tuple[int, int], config_id: str, batch_configs: Dict[str, Dict]):
    """Write a config file for a specific matrix size and config combination"""
    N, K = matrix_size
    folder_name = f"{N}_{K}"
    base_dir = "/mnt/share-drive/dev-vllm-experiment/testconfigs"
    folder_path = os.path.join(base_dir, folder_name)

    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Filename format: config_id.json
    filename = f"{config_id}.json"
    filepath = os.path.join(folder_path, filename)

    # Write the config file
    with open(filepath, 'w') as f:
        json.dump(batch_configs, f, indent=4)

    return filepath

def main():
    print("=" * 80)
    print("FP8 WMMA Config File Generator")
    print("=" * 80)

    total_configs = 0

    for N, K in MATRIX_SIZES:
        print(f"\nGenerating configs for matrix size N={N}, K={K}")

        # For each unique config combination, generate configs for all batch sizes
        # First, get all possible configs from batch size 16 (smallest)
        all_configs = {}

        for batch_size in BATCH_SIZES:
            batch_configs = generate_configs_for_batch_size(batch_size)

            # Group by config parameters (excluding batch-specific optimizations)
            for config in batch_configs:
                config_id = generate_config_id(config)

                if config_id not in all_configs:
                    all_configs[config_id] = {}

                # Store config for this batch size
                all_configs[config_id][str(batch_size)] = config

        # Write one file per unique config combination
        for config_id, batch_configs in all_configs.items():
            write_config_file((N, K), config_id, batch_configs)
            total_configs += 1

        print(f"  Generated {len(all_configs)} config files")

    print("\n" + "=" * 80)
    print(f"Total config files generated: {total_configs}")
    print(f"Across {len(MATRIX_SIZES)} matrix sizes")
    print(f"Testing {len(BATCH_SIZES)} batch sizes each")
    print("=" * 80)

    # Print summary stats
    sample_batch_configs = generate_configs_for_batch_size(64)
    print(f"\nExample: {len(sample_batch_configs)} valid configs for batch size 64")
    print(f"Total parameter combinations before filtering: {len(BLOCK_SIZES)**3 * len(GROUP_SIZES_M) * len(NUM_WARPS) * len(KPACK_VALUES)}")

if __name__ == "__main__":
    main()
