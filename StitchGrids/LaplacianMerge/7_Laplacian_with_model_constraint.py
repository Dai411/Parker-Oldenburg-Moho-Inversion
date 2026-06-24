"""
7_Laplacian_with_model_constraint.py

Usage and CLI docstring moved to file header. This script applies a large-scale model constraint
(e.g., ICGEM BouguerModelled.asc) to a merged/interpolated Laplacian grid to preserve large-scale
trends and avoid inversion-induced "flattening" of extrema.

Typical place in workflow: after 3_merge_and_interpolate_dual.py (which produces L_FilledLaplacian.asc)
and before 5_frequency_inversion.py. The script aligns grids, identifies core/transition/external
regions, estimates boundary offsets, and blends model and observed Laplacian in the transition band.

Example usage:
  python 7_Laplacian_with_model_constraint.py \
    --laplacian L_FilledLaplacian.asc \
    --our L_FilledLaplacian.asc \
    --model BouguerModelled.asc \
    --output BouguerLaplacianConstrained.asc \
    --transition-sigma 20 --safety-padding 10 --boundary-width 5

Defaults:
  --laplacian: L_FilledLaplacian.asc
  --our: L_FilledLaplacian.asc
  --model: BouguerModelled.asc
  --output: BouguerLaplacianConstrained.asc
  --transition-sigma: 20
  --safety-padding: 10
  --boundary-width: 5

Notes:
- The script uses the model to constrain low-frequency components in the source domain (Laplacian).
- Keep transition-sigma small if model and data have similar resolution; increase it if the model is
  much smoother than the observations.

"""

import argparse
import numpy as np
import os
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation


def read_asc_header(filename):
    with open(filename, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            key = line[0].lower()
            if '.' in line[1] or 'e' in line[1].lower():
                value = float(line[1])
            else:
                value = int(line[1])
            header[key] = value
    return header


def read_asc_grid(filename, header=None):
    if header is None:
        header = read_asc_header(filename)
    data = np.loadtxt(filename, skiprows=6)
    nodata = header.get('nodata_value', -9999)
    data[data == nodata] = np.nan
    return data


def write_asc_grid(filename, data, header):
    output_data = data.copy()
    nodata = header.get('nodata_value', -9999)
    output_data[np.isnan(output_data)] = nodata
    with open(filename, 'w') as f:
        f.write(f"ncols        {header['ncols']}\n")
        f.write(f"nrows        {header['nrows']}\n")
        f.write(f"xllcorner    {header['xllcorner']:.6f}\n")
        f.write(f"yllcorner    {header['yllcorner']:.6f}\n")
        f.write(f"cellsize     {header['cellsize']}\n")
        f.write(f"nodata_value {nodata}\n")
        for row in range(output_data.shape[0]):
            f.write(' '.join(f"{val:.6f}" for val in output_data[row]) + '\n')


def align_grid_to_model(src_data, src_header, model_header):
    model_nrows = model_header['nrows']
    model_ncols = model_header['ncols']
    model_xmin = model_header['xllcorner']
    model_ymax = model_header['yllcorner'] + model_header['nrows'] * model_header['cellsize']
    cellsize = model_header['cellsize']

    src_xmin = src_header['xllcorner']
    src_ymax = src_header['yllcorner'] + src_header['nrows'] * src_header['cellsize']

    aligned = np.full((model_nrows, model_ncols), np.nan)

    for i in range(src_header['nrows']):
        src_y = src_ymax - i * cellsize
        model_row = int(round((model_ymax - src_y) / cellsize))
        if model_row < 0 or model_row >= model_nrows:
            continue
        for j in range(src_header['ncols']):
            src_val = src_data[i, j]
            if np.isnan(src_val):
                continue
            src_x = src_xmin + j * cellsize
            model_col = int(round((src_x - model_xmin) / cellsize))
            if 0 <= model_col < model_ncols:
                aligned[model_row, model_col] = src_val
    return aligned


def parse_args():
    parser = argparse.ArgumentParser(description='Apply model constraint to a merged Laplacian grid.')
    parser.add_argument('--laplacian', default='L_FilledLaplacian.asc',
                        help='Input Laplacian file (merged & filled)')
    parser.add_argument('--our', default='L_FilledLaplacian.asc',
                        help='Our merged/interpolated data (used to define valid/core regions)')
    parser.add_argument('--model', default='BouguerModelled.asc',
                        help='Model/ICGEM file used as external constraint')
    parser.add_argument('--output', default='BouguerLaplacianConstrained.asc',
                        help='Output constrained Laplacian file')
    parser.add_argument('--transition-sigma', type=float, default=20.0,
                        help='Gaussian sigma for transition weight (pixels)')
    parser.add_argument('--safety-padding', type=int, default=10,
                        help='Erosion iterations to define core region (pixels)')
    parser.add_argument('--boundary-width', type=int, default=5,
                        help='Boundary layer width for offset estimation (pixels)')
    return parser.parse_args()


def main():
    args = parse_args()

    laplacian_file = args.laplacian
    our_file = args.our
    model_file = args.model
    output_file = args.output

    TRANSITION_SIGMA = args.transition_sigma
    SAFETY_PADDING = args.safety_padding
    BOUNDARY_WIDTH = args.boundary_width

    print("=" * 60)
    print("Laplacian 域结果 + 模型数据约束校正 (CLI 版)")
    print("=" * 60)

    # 1. 读取数据和 header
    print("\n1. 读取数据...")
    model_header = read_asc_header(model_file)
    our_header = read_asc_header(our_file)
    lap_header = read_asc_header(laplacian_file)

    model_data = read_asc_grid(model_file, model_header)
    our_data = read_asc_grid(our_file, our_header)
    lap_data = read_asc_grid(laplacian_file, lap_header)

    print(f"   模型数据: {model_data.shape}, 有效点: {np.sum(~np.isnan(model_data))}")
    print(f"   我们的数据: {our_data.shape}, 有效点: {np.sum(~np.isnan(our_data))}")
    print(f"   Laplacian结果: {lap_data.shape}, 有效点: {np.sum(~np.isnan(lap_data))}")

    # 2.  对齐到模型网格
    print("\n2. 对齐到模型网格...")
    our_aligned = align_grid_to_model(our_data, our_header, model_header)
    lap_aligned = align_grid_to_model(lap_data, lap_header, model_header)

    print(f"   对齐后我们的数据有效点: {np.sum(~np.isnan(our_aligned))}")
    print(f"   对齐后Laplacian有效点: {np.sum(~np.isnan(lap_aligned))}")

    # 3. 识别各区域
    print("\n3. 识别区域...")
    our_valid_mask = ~np.isnan(our_aligned)
    core_mask = binary_erosion(our_valid_mask, iterations=SAFETY_PADDING)
    boundary_mask = binary_dilation(our_valid_mask, iterations=BOUNDARY_WIDTH) & ~our_valid_mask
    boundary_mask = boundary_mask & ~np.isnan(lap_aligned)
    external_mask = ~np.isnan(model_data) & ~our_valid_mask
    transition_mask = our_valid_mask & ~core_mask

    print(f"   核心区点数: {np.sum(core_mask)}")
    print(f"   过渡带点数: {np.sum(transition_mask)}")
    print(f"   边界层点数 (用于估算): {np.sum(boundary_mask)}")
    print(f"   外部区域点数: {np.sum(external_mask)}")

    # 4. 估计外部偏移
    print("\n4. 计算外部区域校正量...")
    if np.sum(boundary_mask) > 0:
        boundary_diff = model_data[boundary_mask] - lap_aligned[boundary_mask]
        offset_global = np.nanmean(boundary_diff)
        offset_std = np.nanstd(boundary_diff)
        print(f"   边界层差异 (模型 - Laplacian): 均值 = {offset_global:.4f}, 标准差 = {offset_std:.4f}")
    else:
        print("   警告: 边界层为空，使用默认偏移 0")
        offset_global = 0.0

    # 5. 创建过渡权重
    print("\n5. 创建过渡权重...")
    weight_initial = np.zeros_like(model_data, dtype=float)
    weight_initial[core_mask] = 1.0
    weight_smooth = gaussian_filter(weight_initial, sigma=TRANSITION_SIGMA)
    weight_smooth = np.clip(weight_smooth, 0.0, 1.0)
    weight_smooth[core_mask] = 1.0
    print(f"   权重范围: [{np.nanmin(weight_smooth):.4f}, {np.nanmax(weight_smooth):.4f}]")

    # 6. 应用校正并生成被约束场
    print("\n6. 应用模型约束校正...")
    offset_field = offset_global * (1.0 - weight_smooth)
    g_corrected = lap_aligned.copy()

    # 在外部区域直接用模型值
    g_corrected[external_mask] = model_data[external_mask]

    # 过渡带混合
    transition_rows, transition_cols = np.where(transition_mask)
    for r, c in zip(transition_rows, transition_cols):
        w = weight_smooth[r, c]
        lap_val = lap_aligned[r, c]
        model_val = model_data[r, c]
        if not np.isnan(lap_val) and not np.isnan(model_val):
            g_corrected[r, c] = (1.0 - w) * model_val + w * lap_val
        elif not np.isnan(lap_val):
            g_corrected[r, c] = lap_val
        elif not np.isnan(model_val):
            g_corrected[r, c] = model_val

    g_corrected[core_mask] = lap_aligned[core_mask]

    # 7. 统计与验证
    print("\n7. 结果统计:")
    print(f"\n   校正前 Laplacian 全场均值: {np.nanmean(lap_aligned):.4f}")
    if np.sum(core_mask) > 0:
        print(f"   校正前核区均值: {np.nanmean(lap_aligned[core_mask]):.4f}")
    print(f"\n   校正后 全场均值: {np.nanmean(g_corrected):.4f}")
    if np.sum(core_mask) > 0:
        print(f"   校正后核区均值: {np.nanmean(g_corrected[core_mask]):.4f}")
    if np.sum(external_mask) > 0:
        print(f"   外部区域均值: {np.nanmean(g_corrected[external_mask]):.4f}")

    if np.sum(core_mask) > 0:
        core_unchanged = np.allclose(g_corrected[core_mask], lap_aligned[core_mask], equal_nan=True)
        print(f"   核心区保持不变: {'✓ 通过' if core_unchanged else '✗ 失败'}")

    if np.sum(external_mask) > 0:
        external_match = np.allclose(g_corrected[external_mask], model_data[external_mask], equal_nan=True, rtol=1e-5)
        print(f"   外部区域等于模型: {'✓ 通过' if external_match else '✗ 失败'}")

    # 8. 保存
    print("\n8. 保存结果...")
    write_asc_grid(output_file, g_corrected, model_header)
    print(f"   ✓ 已保存: {output_file}")
    print("\n" + "=" * 60)
    print("✅ Laplacian + 模型约束完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
