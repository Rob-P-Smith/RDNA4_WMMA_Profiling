#!/usr/bin/env python3
"""
Extract the best performing configs from *_top.md files and create optimal config JSON files.
Reads markdown files and extracts rank 1 configs for each batch size.
"""

import json
import os
import re
from pathlib import Path

TESTCONFIGS_DIR = "/mnt/share-drive/dev-vllm-experiment/testconfigs"
OUTPUT_DIR = "/mnt/share-drive/dev-vllm-experiment/EmpiricalOptimalConfigs"

def parse_config_name(config_name):
    """Parse config name like 'M128_N32_K128_G8_W4_KP2' into parameters"""
    parts = config_name.split('_')
    return {
        'BLOCK_SIZE_M': int(parts[0][1:]),
        'BLOCK_SIZE_N': int(parts[1][1:]),
        'BLOCK_SIZE_K': int(parts[2][1:]),
        'GROUP_SIZE_M': int(parts[3][1:]),
        'num_warps': int(parts[4][1:]),
        'kpack': int(parts[5][2:]),
        'matrix_instr_nonkdim': 16
    }

def extract_top_configs_from_md(md_file):
    """Extract top config for each batch size from markdown file"""
    with open(md_file, 'r') as f:
        content = f.read()

    # Extract matrix dimensions from header
    match = re.search(r'# Top 5 Configs per Batch Size - (\d+)x(\d+) Matrix', content)
    if not match:
        return None, None, None

    N, K = int(match.group(1)), int(match.group(2))

    # Extract top config from each batch size table
    configs = {}

    # Find all batch size sections
    batch_sections = re.finditer(r'## Batch Size: (\d+)\n\n(.*?)(?=\n##|\n---|\Z)', content, re.DOTALL)

    for section in batch_sections:
        batch_size = section.group(1)
        table_content = section.group(2)

        # Extract first row (rank 1) from table
        match = re.search(r'\| 1 \| `([^`]+)` \|', table_content)
        if match:
            config_name = match.group(1)
            configs[batch_size] = parse_config_name(config_name)

    return N, K, configs

def main():
    # Find all *_top.md files
    top_md_files = list(Path(TESTCONFIGS_DIR).glob("*_top.md"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print("EXTRACTING OPTIMAL CONFIGS FROM TOP.MD FILES")
    print("="*80)
    print(f"\nFound {len(top_md_files)} top.md files to process\n")

    success_count = 0
    for md_file in sorted(top_md_files):
        print(f"Processing: {md_file.name}")

        try:
            N, K, configs = extract_top_configs_from_md(md_file)

            if configs:
                # Create vLLM-compatible filename
                filename = f"N={N},K={K},device_name=0x7551,dtype=fp8_w8a8,block_shape=[128,128].json"
                filepath = os.path.join(OUTPUT_DIR, filename)

                # Write config file
                with open(filepath, 'w') as f:
                    json.dump(configs, f, indent=4)

                print(f"  ✓ Created {filename}")
                print(f"    Configured {len(configs)} batch sizes")
                success_count += 1
            else:
                print(f"  ✗ Failed to extract configs")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nSuccessfully created {success_count} config files")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
