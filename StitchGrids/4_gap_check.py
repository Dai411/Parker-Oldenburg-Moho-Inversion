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

# ===== 设置路径 =====
OUTPUT_DIR = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
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
    print(f'Mask 形状: {mask.shape}')
    print(f'  mask 有效点 (非NaN): {np.sum(~np.isnan(mask)):,}')
    print(f'  mask=1 (Sea): {np.sum(mask == 1):,}')
    print(f'  mask=0 (Land): {np.sum(mask == 0):,}')
    print(f'  mask=NaN: {np.sum(np.isnan(mask)):,}')

    print(f'\nL_before 形状: {l_before.shape}')
    print(f'  L_before 有效点: {np.sum(~np.isnan(l_before)):,}')
    print(f'  L_before NaN: {np.sum(np.isnan(l_before)):,}')

    print(f'\nL_filled 形状: {l_filled.shape}')
    print(f'  L_filled 有效点: {np.sum(~np.isnan(l_filled)):,}')
    print(f'  L_filled NaN: {np.sum(np.isnan(l_filled)):,}')

    gap_mask = (~np.isnan(mask)) & np.isnan(l_before)
    print(f'\ngap_mask (mask 有效且 L_before 为 NaN): {np.sum(gap_mask):,}')
    if np.sum(gap_mask) > 0:
        gap_filled = l_filled[gap_mask]
        print(f'  这些位置在 L_filled 中有效: {np.sum(~np.isnan(gap_filled)):,}')


def print_quality(mask, l_before, l_filled):
    print('=== Quality Check ===')
    gap_mask = (~np.isnan(mask)) & np.isnan(l_before)
    gap_count = np.sum(gap_mask)
    print(f'Gap 区域点数: {gap_count:,}')

    if gap_count == 0:
        print('  无 Gap 区域可分析。')
        return

    gap_after = l_filled[gap_mask]
    print(f'  插值前 NaN 数量: {np.sum(np.isnan(gap_after)):,}')
    print(f'  插值后有效数量: {np.sum(~np.isnan(gap_after)):,}')
    if np.sum(~np.isnan(gap_after)) > 0:
        print(f'  插值后数值范围: {np.nanmin(gap_after):.6f} ~ {np.nanmax(gap_after):.6f}')
        print(f'  插值后均值: {np.nanmean(gap_after):.6f}')

    neighbor_mask = binary_dilation(gap_mask, iterations=1)
    surround_mask = neighbor_mask & ~gap_mask & ~np.isnan(l_filled)
    surround_vals = l_filled[surround_mask]

    if surround_vals.size == 0:
        print('  没有可用的周围区域值进行比较。')
        return

    print(f'\nGap 周围区域统计:')
    print(f'  点数: {surround_vals.size:,}')
    print(f'  范围: {np.min(surround_vals):.6f} ~ {np.max(surround_vals):.6f}')
    print(f'  均值: {np.mean(surround_vals):.6f}')

    gap_mean = np.nanmean(gap_after)
    surround_mean = np.mean(surround_vals)
    diff = abs(gap_mean - surround_mean)
    print(f'\n平滑性检查:')
    print(f'  Gap 区域均值: {gap_mean:.6f}')
    print(f'  周围区域均值: {surround_mean:.6f}')
    print(f'  差异: {diff:.6f}')

    if diff < np.std(surround_vals):
        print('  ✓ Gap 区域与周围区域一致，插值合理')
    else:
        print('  ⚠️ Gap 区域与周围区域存在差异')


def print_quick(l_filled):
    print('=== Quick File Check ===')
    valid_count = np.sum(~np.isnan(l_filled))
    print(f'    L_filled 形状: {l_filled.shape}')
    print(f'    有效点 (非 NaN): {valid_count:,}')
    if valid_count > 0:
        print(f'    最小值: {np.nanmin(l_filled):.6f}')
        print(f'    最大值: {np.nanmax(l_filled):.6f}')


def main():
    parser = argparse.ArgumentParser(description='Unified gap checking for StitchGrids outputs.')
    parser.add_argument('--mode', choices=['all', 'summary', 'quality', 'quick'], default='all',
                        help='选择检查模式: all/summary/quality/quick')
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
