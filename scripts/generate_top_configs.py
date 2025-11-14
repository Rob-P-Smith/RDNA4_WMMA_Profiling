#!/usr/bin/env python3
"""
Generate top.md files for all *results.json files showing top 5 configs per batch size.
"""

import json
import os
from pathlib import Path

TESTCONFIGS_DIR = "/mnt/share-drive/dev-vllm-experiment/testconfigs"
BATCH_SIZES = ['16', '32', '64', '128', '256', '512', '1024', '2048', '4096']

def analyze_results_file(results_file):
    """Analyze a results.json file and generate markdown output."""

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Get matrix key (should be only one)
    matrix_key = list(results['results'].keys())[0]
    matrix_results = results['results'][matrix_key]

    # Parse matrix dimensions from key (e.g., "matrix_512_512")
    parts = matrix_key.split('_')
    N, K = int(parts[1]), int(parts[2])

    # Generate markdown output
    md_lines = []
    md_lines.append(f"# Top 5 Configs per Batch Size - {N}x{K} Matrix")
    md_lines.append("")
    md_lines.append(f"**Matrix Dimensions**: N={N}, K={K}")
    md_lines.append("")
    md_lines.append("⚠️ = High variance (CV > 0.5) - potentially unreliable/aberrant")
    md_lines.append("")
    md_lines.append("**Stable** = Config with lowest CV (most stable) for this batch size (shown if not in top 5)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    for batch_size in BATCH_SIZES:
        md_lines.append(f"## Batch Size: {batch_size}")
        md_lines.append("")

        # Collect all configs for this batch size
        batch_perf = []
        for config_name, batch_data in matrix_results.items():
            if batch_size in batch_data:
                metrics = batch_data[batch_size]
                tops = metrics['TOPS']
                min_val = metrics['Min']
                max_val = metrics['Max']
                q1 = metrics.get('Q1', min_val)  # Fallback to Min if Q1 not available
                q3 = metrics.get('Q3', max_val)  # Fallback to Max if Q3 not available
                variance = max_val - min_val
                cv = variance / tops if tops > 0 else 0
                batch_perf.append((config_name, tops, min_val, max_val, q1, q3, cv))

        if not batch_perf:
            md_lines.append("*No data for this batch size*")
            md_lines.append("")
            continue

        # Sort by TOPS and get top 5
        batch_perf.sort(key=lambda x: x[1], reverse=True)
        top5 = batch_perf[:5]

        # Find config with lowest CV (most stable)
        lowest_cv_config = min(batch_perf, key=lambda x: x[6])

        # Check if lowest CV config is already in top 5
        top5_config_names = [config[0] for config in top5]
        add_lowest_cv = lowest_cv_config[0] not in top5_config_names

        # Create table
        md_lines.append("| Rank | Config | Mean TOPS | Min | Q1 | Q3 | Max | CV |")
        md_lines.append("|------|--------|-----------|-----|-----|-----|-----|-----|")

        for rank, (config_name, tops, min_val, max_val, q1, q3, cv) in enumerate(top5, 1):
            cv_flag = " ⚠️" if cv > 0.5 else ""
            md_lines.append(f"| {rank} | `{config_name}` | {tops:.2f} | {min_val:.2f} | {q1:.2f} | {q3:.2f} | {max_val:.2f} | {cv:.3f}{cv_flag} |")

        # Add lowest CV config if not in top 5
        if add_lowest_cv:
            config_name, tops, min_val, max_val, q1, q3, cv = lowest_cv_config
            cv_flag = " ⚠️" if cv > 0.5 else ""
            md_lines.append(f"| **Stable** | `{config_name}` | {tops:.2f} | {min_val:.2f} | {q1:.2f} | {q3:.2f} | {max_val:.2f} | {cv:.3f}{cv_flag} |")

        md_lines.append("")

    # Summary statistics
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- **Matrix Size**: {N}x{K}")
    md_lines.append(f"- **Total Configs Tested**: {len(matrix_results)}")
    md_lines.append(f"- **Batch Sizes**: {', '.join(BATCH_SIZES)}")

    # Analyze K-dimension patterns
    k_values = set()
    for batch_size in BATCH_SIZES:
        batch_perf = []
        for config_name, batch_data in matrix_results.items():
            if batch_size in batch_data:
                tops = batch_data[batch_size]['TOPS']
                batch_perf.append((config_name, tops))

        if batch_perf:
            batch_perf.sort(key=lambda x: x[1], reverse=True)
            top_config = batch_perf[0][0]
            # Parse K value from config name
            parts = top_config.split('_')
            k_val = int(parts[2][1:])
            k_values.add(k_val)

    md_lines.append(f"- **K-dimension values in top configs**: {sorted(k_values)}")
    md_lines.append("")

    return '\n'.join(md_lines)

def main():
    # Find all *results.json files
    results_files = list(Path(TESTCONFIGS_DIR).glob("*results.json"))

    print(f"Found {len(results_files)} results files to process")
    print("="*80)

    for results_file in sorted(results_files):
        # Generate output filename (e.g., 512x512_results.json -> 512x512_top.md)
        output_name = results_file.stem.replace('_results', '_top') + '.md'
        output_path = results_file.parent / output_name

        print(f"\nProcessing: {results_file.name}")
        print(f"  Output: {output_name}")

        try:
            # Generate markdown
            markdown_content = analyze_results_file(results_file)

            # Write output
            with open(output_path, 'w') as f:
                f.write(markdown_content)

            print(f"  ✓ Generated successfully")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*80)
    print("Processing complete!")
    print(f"\nGenerated files in: {TESTCONFIGS_DIR}")

if __name__ == "__main__":
    main()
