# merge_grids_weight_gaussian.py
"""
加权平均拼接海陆重力数据
- 纯陆区：保留 Land 原始值
- 纯海区：保留 Sea 原始值
- 重叠区：加权平均平滑过渡
"""

import numpy as np
import os
from scipy.ndimage import gaussian_filter

# ============================================================
# 参数配置
# ============================================================

SIGMA = 15  # 高斯平滑 sigma (像素)，约 4.5 km

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
output_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
output_file = os.path.join(output_dir, 'BouguerWeightedMerge_Gaussian.asc')

# ============================================================
# 读取函数
# ============================================================

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
    nodata = header['nodata_value']
    data[data == nodata] = np.nan
    return data

def write_asc_grid(filename, data, header):
    output_data = data.copy()
    output_data[np.isnan(output_data)] = header['nodata_value']
    with open(filename, 'w') as f:
        f.write(f"ncols        {header['ncols']}\n")
        f.write(f"nrows        {header['nrows']}\n")
        f.write(f"xllcorner    {header['xllcorner']:.6f}\n")
        f.write(f"yllcorner    {header['yllcorner']:.6f}\n")
        f.write(f"cellsize     {header['cellsize']}\n")
        f.write(f"nodata_value {header['nodata_value']}\n")
        for row in range(output_data.shape[0]):
            f.write(' '.join(f"{val:.6f}" for val in output_data[row]) + '\n')

def align_sea_to_land(sea_data, sea_header, land_header):
    """将 Sea 对齐到 Land 网格"""
    land_nrows = land_header['nrows']
    land_ncols = land_header['ncols']
    land_xmin = land_header['xllcorner']
    land_ymax = land_header['yllcorner'] + land_header['nrows'] * land_header['cellsize']
    cellsize = land_header['cellsize']
    
    sea_xmin = sea_header['xllcorner']
    sea_ymax = sea_header['yllcorner'] + sea_header['nrows'] * sea_header['cellsize']
    
    aligned = np.full((land_nrows, land_ncols), np.nan)
    
    for i in range(sea_header['nrows']):
        sea_y = sea_ymax - i * cellsize
        land_row = int(round((land_ymax - sea_y) / cellsize))
        if land_row < 0 or land_row >= land_nrows:
            continue
        for j in range(sea_header['ncols']):
            sea_val = sea_data[i, j]
            if np.isnan(sea_val):
                continue
            sea_x = sea_xmin + j * cellsize
            land_col = int(round((sea_x - land_xmin) / cellsize))
            if 0 <= land_col < land_ncols:
                aligned[land_row, land_col] = sea_val
    return aligned

# ============================================================
# 主程序
# ============================================================

print("=" * 60)
print("加权平均拼接海陆重力数据 (最终版)")
print("=" * 60)

# 1. 读取数据
print("\n1. 读取数据...")
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)

print(f"   Land: {land_data.shape}, 有效点: {np.sum(~np.isnan(land_data))}")
print(f"   Sea:  {sea_data.shape}, 有效点: {np.sum(~np.isnan(sea_data))}")

# 2. 对齐 Sea
print("\n2. 对齐 Sea 到 Land 网格...")
sea_aligned = align_sea_to_land(sea_data, sea_header, land_header)
print(f"   对齐后 Sea 有效点: {np.sum(~np.isnan(sea_aligned))}")

# 3. 创建全场平滑权重
print(f"\n3. 创建全场平滑权重 (sigma={SIGMA} 像素, ~{SIGMA*300/1000:.1f} km)...")

# 初始权重：海区为 1，其他为 0
weight_initial = np.zeros_like(land_data)
sea_mask = ~np.isnan(sea_aligned)
weight_initial[sea_mask] = 1.0

# 全场高斯平滑（包括纯海区、纯陆区、无数据区）
weight_clean = np.nan_to_num(weight_initial, nan=0.0)
weight_smooth = gaussian_filter(weight_clean, sigma=SIGMA)
weight_smooth = np.clip(weight_smooth, 0, 1)

print(f"   权重范围: [{np.nanmin(weight_smooth):.4f}, {np.nanmax(weight_smooth):.4f}]")

# 4. 合并
print("\n4. 加权平均合并...")

# 初始化结果：先复制 Land 数据
result = land_data.copy()

# 将 Sea 数据覆盖到海区（包括纯海区）
result[sea_mask] = sea_aligned[sea_mask]

# 对重叠区进行加权平均
overlap_mask = ~np.isnan(land_data) & ~np.isnan(sea_aligned)
print(f"   重叠区点数: {np.sum(overlap_mask)}")

overlap_rows, overlap_cols = np.where(overlap_mask)
for r, c in zip(overlap_rows, overlap_cols):
    w = weight_smooth[r, c]
    result[r, c] = (1 - w) * land_data[r, c] + w * sea_aligned[r, c]

# 5. 统计结果
print("\n5. 结果统计:")
print(f"   最终有效点: {np.sum(~np.isnan(result))}")
print(f"   最小值: {np.nanmin(result):.2f} mGal")
print(f"   最大值: {np.nanmax(result):.2f} mGal")
print(f"   平均值: {np.nanmean(result):.2f} mGal")
print(f"   标准差: {np.nanstd(result):.6f} mGal")

# 分区域统计
land_only = ~np.isnan(land_data) & np.isnan(sea_aligned)
sea_only = ~np.isnan(sea_aligned) & np.isnan(land_data)
print(f"\n   纯陆区均值: {np.nanmean(result[land_only]):.2f} mGal")
print(f"   纯海区均值: {np.nanmean(result[sea_only]):.2f} mGal")
print(f"   重叠区均值: {np.nanmean(result[overlap_mask]):.2f} mGal")

# 6. 保存
print("\n6. 保存结果...")
write_asc_grid(output_file, result, land_header)
print(f"   ✓ 已保存: {output_file}")

print("\n" + "=" * 60)
print("✅ 加权平均拼接完成！")
print("=" * 60)
