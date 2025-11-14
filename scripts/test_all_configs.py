#!/usr/bin/env python3
"""
FP8 WMMA Config Testing Script
Tests all generated configs and records performance metrics
"""

import json
import os
import time
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale

# Paths
TESTCONFIGS_BASE = "/workspace/testconfigs"
VLLM_CONFIGS_DIR = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs"
DEVICE_NAME = "0x7551"  # AMD Radeon RX 9070 XT
BLOCK_SHAPE = "[128,128]"
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

class ConfigTester:
    def __init__(self, smoke_test: bool = False, gpu_id: int = 0):
        self.smoke_test = smoke_test
        self.iterations = 1 if smoke_test else 100
        self.warmup_iterations = 1 if smoke_test else 10
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.current_matrix_key = None
        self.current_results_file = None
        self.results = {}

        # Set the default CUDA device for this tester
        torch.cuda.set_device(self.gpu_id)
        print(f"[GPU {self.gpu_id}] Initialized on {torch.cuda.get_device_name(self.gpu_id)}")

    def get_results_file(self, N: int, K: int) -> str:
        """Get results filename for specific matrix size"""
        return f"{TESTCONFIGS_BASE}/{N}x{K}_results.json"

    def load_results(self, N: int, K: int) -> Dict:
        """Load existing results for a matrix size or create new structure"""
        results_file = self.get_results_file(N, K)
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        return {"results": {}}

    def save_results(self):
        """Save results to matrix-specific file"""
        if self.current_results_file:
            with open(self.current_results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"  [SAVED] Results written to {self.current_results_file}")

    def get_vllm_config_filename(self, N: int, K: int) -> str:
        """Generate vLLM expected config filename"""
        return f"N={N},K={K},device_name={DEVICE_NAME},dtype=fp8_w8a8,block_shape={BLOCK_SHAPE}.json"

    def is_config_tested(self, matrix_key: str, config_id: str) -> bool:
        """Check if a config has been completely tested (all batch sizes)"""
        if matrix_key not in self.results["results"]:
            return False
        if config_id not in self.results["results"][matrix_key]:
            return False

        # Check if all batch sizes are present
        config_data = self.results["results"][matrix_key][config_id]
        return all(str(bs) in config_data for bs in BATCH_SIZES)

    def test_config(self, N: int, K: int, batch_size: int, config: Dict) -> List[float]:
        """Run FP8 operations and measure TFLOPS"""
        M = batch_size
        tflops_results = []

        try:
            # Create test tensors
            input_tensor = torch.randn(M, K, dtype=torch.float16, device=self.device)
            weight = torch.randn(N, K, dtype=torch.float16, device=self.device)

            # Quantize input
            input_fp8, input_scale = per_token_group_quant_fp8(
                input_tensor,
                group_size=128,
                dtype=torch.float8_e4m3fn
            )

            # Quantize weight
            weight_fp8, weight_scale = per_token_group_quant_fp8(
                weight.T.contiguous(),
                group_size=128,
                dtype=torch.float8_e4m3fn
            )
            weight_fp8 = weight_fp8.T.contiguous()
            weight_scale = weight_scale.T.contiguous()

            # Warmup
            for _ in range(self.warmup_iterations):
                output = gemm_a8w8_blockscale(
                    input_fp8,
                    weight_fp8,
                    input_scale,
                    weight_scale,
                    dtype=torch.float16,
                    config=config
                )
            torch.cuda.synchronize()

            # Measure
            for _ in range(self.iterations):
                start = time.perf_counter()
                output = gemm_a8w8_blockscale(
                    input_fp8,
                    weight_fp8,
                    input_scale,
                    weight_scale,
                    dtype=torch.float16,
                    config=config
                )
                torch.cuda.synchronize()
                end = time.perf_counter()

                # Calculate TFLOPS
                flops = 2 * M * N * K
                time_s = end - start
                tflops = flops / (time_s * 1e12)
                tflops_results.append(tflops)

            # Clean up tensors
            del input_tensor, weight, input_fp8, input_scale, weight_fp8, weight_scale, output

            return tflops_results

        except Exception as e:
            print(f"    [ERROR] Test failed: {e}")
            return []

    def calculate_statistics(self, tflops_list: List[float]) -> Dict:
        """Calculate statistics from TFLOPS measurements"""
        if not tflops_list:
            return {"TOPS": 0, "Min": 0, "Max": 0, "Q1": 0, "Q3": 0}

        arr = np.array(tflops_list)
        return {
            "TOPS": float(np.mean(arr)),
            "Min": float(np.min(arr)),
            "Max": float(np.max(arr)),
            "Q1": float(np.percentile(arr, 25)),
            "Q3": float(np.percentile(arr, 75))
        }

    def test_matrix_size(self, matrix_folder: str):
        """Test all configs for a specific matrix size"""
        # Parse matrix size from folder name
        parts = matrix_folder.split('_')
        N, K = int(parts[0]), int(parts[1])
        matrix_key = f"matrix_{N}_{K}"

        print(f"\n{'='*80}")
        print(f"[GPU {self.gpu_id}] Testing Matrix Size: N={N}, K={K}")
        print(f"{'='*80}")

        # Load existing results for this matrix
        self.current_results_file = self.get_results_file(N, K)
        self.results = self.load_results(N, K)
        self.current_matrix_key = matrix_key

        # Initialize results structure for this matrix
        if matrix_key not in self.results["results"]:
            self.results["results"][matrix_key] = {}

        # Get all config files
        folder_path = os.path.join(TESTCONFIGS_BASE, matrix_folder)
        config_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

        print(f"Found {len(config_files)} config files to test")
        print(f"Results will be saved to: {self.current_results_file}")

        for idx, config_file in enumerate(config_files, 1):
            config_id = config_file.replace('.json', '')
            config_path = os.path.join(folder_path, config_file)

            # Skip if entire config already tested
            if self.is_config_tested(matrix_key, config_id):
                print(f"\n[{idx}/{len(config_files)}] SKIPPING config: {config_id} (already complete)")
                continue

            print(f"\n[{idx}/{len(config_files)}] Testing config: {config_id}")

            # Load config file
            with open(config_path, 'r') as f:
                batch_configs = json.load(f)

            # Initialize results structure for this config
            if config_id not in self.results["results"][matrix_key]:
                self.results["results"][matrix_key][config_id] = {}

            # Test each batch size
            for batch_size in BATCH_SIZES:
                batch_key = str(batch_size)

                # Get config for this batch size
                if batch_key not in batch_configs:
                    print(f"  [SKIP] No config for batch size {batch_size}")
                    continue

                config = batch_configs[batch_key]

                # Add MI350X default parameters that AITER expects
                config.setdefault("NUM_KSPLIT", 1)
                config.setdefault("num_stages", 2)
                config.setdefault("waves_per_eu", 2)
                config.setdefault("cache_modifier", ".cg")

                print(f"  [TEST] Batch size: {batch_size} ({self.iterations} iterations)")

                # Run tests
                tflops_results = self.test_config(N, K, batch_size, config)

                if tflops_results:
                    # Calculate statistics
                    stats = self.calculate_statistics(tflops_results)

                    # Store results
                    self.results["results"][matrix_key][config_id][batch_key] = stats

                    print(f"    ✓ TOPS: {stats['TOPS']:.2f} (Min: {stats['Min']:.2f}, Max: {stats['Max']:.2f})")
                else:
                    print(f"    ✗ Test failed for batch size {batch_size}")

            # Save results after completing all batch sizes for this config
            self.save_results()

            # Clear GPU memory after each config
            torch.cuda.empty_cache()
            print(f"  [CLEARED] GPU memory")

    def run_all_tests(self):
        """Test all matrix sizes and configs"""
        print("="*80)
        print("FP8 WMMA Config Testing")
        print(f"Mode: {'SMOKE TEST (1 iteration)' if self.smoke_test else 'FULL TEST (100 iterations)'}")
        print("="*80)

        # Get all matrix size folders
        matrix_folders = sorted([
            d for d in os.listdir(TESTCONFIGS_BASE)
            if os.path.isdir(os.path.join(TESTCONFIGS_BASE, d)) and '_' in d
        ])

        print(f"\nFound {len(matrix_folders)} matrix sizes to test")
        print(f"Batch sizes: {BATCH_SIZES}")

        for matrix_folder in matrix_folders:
            self.test_matrix_size(matrix_folder)

        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)
        print(f"Results saved to individual matrix files: {TESTCONFIGS_BASE}/<N>x<K>_results.json")

def main():
    parser = argparse.ArgumentParser(description='Test FP8 WMMA configurations')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run smoke test with only 1 iteration per test')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (0 or 1). If not specified, runs all matrices on default GPU')
    args = parser.parse_args()

    # Balanced split of 22 matrices across 2 GPUs
    # Small, medium, and large matrices distributed evenly
    GPU_0_MATRICES = [
        '1024_512',     # Small
        '1024_1024',    # Small
        '1024_2048',    # Medium
        '2048_1024',    # Medium
        '2048_2048',    # Medium
        '3072_1536',    # Medium
        '4096_4096',    # Large
        '4608_7168',    # Large
        '8192_1024',    # Large
        '32768_8192',   # Huge
    ]

    GPU_1_MATRICES = [
        '512_1024',     # Small
        '512_7168',     # Small
        '1024_3072',    # Medium
        '1024_8192',    # Large
        '3072_1024',    # Medium
        '2112_7168',    # Large
        '4096_7168',    # Large
        '7168_256',     # Medium
        '7168_2048',    # Large
        '8192_8192',    # Huge
        '8192_32768',   # Huge
    ]

    if args.gpu is not None:
        # Run only matrices assigned to specified GPU
        if args.gpu == 0:
            matrix_list = GPU_0_MATRICES
        elif args.gpu == 1:
            matrix_list = GPU_1_MATRICES
        else:
            print(f"Error: GPU must be 0 or 1, got {args.gpu}")
            return

        tester = ConfigTester(smoke_test=args.smoke_test, gpu_id=args.gpu)

        print(f"\n[GPU {args.gpu}] Testing {len(matrix_list)} assigned matrices:")
        for m in matrix_list:
            print(f"  - {m}")
        print()

        for matrix_folder in matrix_list:
            tester.test_matrix_size(matrix_folder)

        print("\n" + "="*80)
        print(f"[GPU {args.gpu}] TESTING COMPLETE")
        print("="*80)
        print(f"Results saved to individual matrix files")
    else:
        # Run all tests on default GPU (original behavior)
        tester = ConfigTester(smoke_test=args.smoke_test)
        tester.run_all_tests()

if __name__ == "__main__":
    main()
