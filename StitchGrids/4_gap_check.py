"""4_gap_check.py

Unified gap-check script for the StitchGrids workflow.

Usage:
  python 4_gap_check.py            # run all checks
  python 4_gap_check.py --mode all
  python 4_gap_check.py --mode summary
  python 4_gap_check.py --mode quality
  python 4_gap_check.py --mode quick
  - basic mask/L_before/L_filled statistics
  - quick L_filled validity check

"""

import argparse
import os
import numpy as np
from scipy.ndimage import binary_dilation

# ===== Set File Path =====
OUTPUT_DIR = 'C:/.../BouguerAnomaly/StitchGrids'
MASK_FILE = os.path.join(OUTPUT_DIR, 'PriorityMask.asc')
L_BEFORE_FILE = os.path.join(OUTPUT_DIR, 'L0_MergedLaplacian_BeforeInterp.asc')
L_FILLED_FILE = os.path.join(OUTPUT_DIR, 'L_FilledLaplacian.asc')
# ====================


def read_asc_grid(filename):
    with open(filename, 'r') as f:
        for _ in range(6):
            f.readline()
        return np.loadtxt(f)


def print_summary(mask, l_before, l_filled):
    print('=== Summary Check ===')
    print(f'Mask Sahpe: {mask.shape}')
    print(f'  mask Valid Points ( Non NaN): {np.sum(~np.isnan(mask)):,}')
    print(f'  mask=1 (Sea): {np.sum(mask == 1):,}')
    print(f'  mask=0 (Land): {np.sum(mask == 0):,}')
    print(f'  mask=NaN: {np.sum(np.isnan(mask)):,}')

    print(f'\nL_before Shape: {l_before.shape}')
    print(f'  L_before Valid Points: {np.sum(~np.isnan(l_before)):,}')
    print(f'  L_before NaN: {np.sum(np.isnan(l_before)):,}')

    print(f'\nL_filled Shape: {l_filled.shape}')
    print(f'  L_filled Valid Points: {np.sum(~np.isnan(l_filled)):,}')
    print(f'  L_filled NaN: {np.sum(np.isnan(l_filled)):,}')

    gap_mask = (~np.isnan(mask)) & np.isnan(l_before)
    print(f'\ngap_mask (mask valid and L_before is NaN): {np.sum(gap_mask):,}')
    if np.sum(gap_mask) > 0:
        gap_filled = l_filled[gap_mask]
        print(f'  Those locations are valid in L_filled: {np.sum(~np.isnan(gap_filled)):,}')


def print_quality(mask, l_before, l_filled):
    print('=== Quality Check ===')
    gap_mask = (~np.isnan(mask)) & np.isnan(l_before)
    gap_count = np.sum(gap_mask)
    print(f'Gap Region numbers: {gap_count:,}')

    if gap_count == 0:
        print('  No Gap Region for analysis.')
        return

    gap_after = l_filled[gap_mask]
    print(f'  The number of NaN before interpolation: {np.sum(np.isnan(gap_after)):,}')
    print(f'  The number of NaN after interpolation: {np.sum(~np.isnan(gap_after)):,}')
    if np.sum(~np.isnan(gap_after)) > 0:
        print(f'  The data range after interpolation: {np.nanmin(gap_after):.6f} ~ {np.nanmax(gap_after):.6f}')
        print(f'  The mean value after interpolation: {np.nanmean(gap_after):.6f}')

    neighbor_mask = binary_dilation(gap_mask, iterations=1)
    surround_mask = neighbor_mask & ~gap_mask & ~np.isnan(l_filled)
    surround_vals = l_filled[surround_mask]

    if surround_vals.size == 0:
        print('  No valid neighbouring region for comparison.')
        return

    print(f'\nGap Neighbouring region statics:')
    print(f'  Points: {surround_vals.size:,}')
    print(f'  Reange: {np.min(surround_vals):.6f} ~ {np.max(surround_vals):.6f}')
    print(f'  Mean Value: {np.mean(surround_vals):.6f}')

    gap_mean = np.nanmean(gap_after)
    surround_mean = np.mean(surround_vals)
    diff = abs(gap_mean - surround_mean)
    print(f'\nSmoothness check:')
    print(f'  Gap reigion global mean value: {gap_mean:.6f}')
    print(f'  Neighbouring region mean value: {surround_mean:.6f}')
    print(f'  Difference: {diff:.6f}')

    if diff < np.std(surround_vals):
        print('  ✓ Gap region is identical with neighbouring region, interpolation is acceptable')
    else:
        print('  ⚠️ Gap region has apparent difference with neighbouring region')


def print_quick(l_filled):
    print('=== Quick File Check ===')
    valid_count = np.sum(~np.isnan(l_filled))
    print(f'    L_filled Shape: {l_filled.shape}')
    print(f'    Valid Points (Non NaN): {valid_count:,}')
    if valid_count > 0:
        print(f'    Minimum: {np.nanmin(l_filled):.6f}')
        print(f'    Maximum: {np.nanmax(l_filled):.6f}')


def main():
    parser = argparse.ArgumentParser(description='Unified gap checking for StitchGrids outputs.')
    parser.add_argument('--mode', choices=['all', 'summary', 'quality', 'quick'], default='all',
                        help='Check Mode: all/summary/quality/quick')
    args = parser.parse_args()

    mask = read_asc_grid(MASK_FILE)
    l_before = read_asc_grid(L_BEFORE_FILE)
    l_filled = read_asc_grid(L_FILLED_FILE)

    if args.mode in ['all', 'summary']:
        print_summary(mask, l_before, l_filled)
        if args.mode != 'all':
            return
    if args.mode in ['all', 'quality']:
        print('\n')
        print_quality(mask, l_before, l_filled)
        if args.mode != 'all':
            return
    if args.mode in ['all', 'quick']:
        print('\n')
        print_quick(l_filled)


if __name__ == '__main__':
    main()
