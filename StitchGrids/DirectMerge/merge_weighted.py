# merge_weighted.py
"""
加权平均拼接海陆重力数据
支持三种过渡模式：
- 'linear': 线性距离加权
- 'gaussian': 高斯模糊 (推荐)
- 'sine': 正弦曲线过渡 (最平滑)
"""

import numpy as np
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ============================================================
# 参数配置 (可调)
# ============================================================

# 过渡模式: 'linear', 'gaussian', 'sine'
TRANSITION_MODE = 'gaussian'  # 推荐 'gaussian'

# 过渡带宽度（像素数）
# 建议：10 ~ 30 像素
# 你的网格间距 300m，10像素 = 3km，20像素 = 6km
TRANSITION_WIDTH = 15  # 像素

# 高斯平滑的 sigma (仅当 MODE='gaussian' 时使用)
# sigma ≈ TRANSITION_WIDTH / 2
SIGMA = 7.5  # 像素

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
output_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
output_file = os.path.join(output_dir, 'BouguerWeightedMerge.asc')

# ============================================================
# 读取函数
# ============================================================

def read_asc_header(filename):
    """读取 ESRI ASCII 栅格文件头"""
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
    """读取 ESRI ASCII 栅格文件的数据部分"""
    if header is None:
        header = read_asc_header(filename)
    
    data = np.loadtxt(filename, skiprows=6)
    nodata = header['nodata_value']
    data[data == nodata] = np.nan
    
    return data

def write_asc_grid(filename, data, header):
    """写入 ESRI ASCII 栅格文件"""
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
    """将 Sea 网格对齐到 Land 网格"""
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
# 过渡权重计算函数
# ============================================================

def compute_weight_linear(overlap_mask, transition_width):
    """线性距离加权"""
    # 计算到重叠区边界的距离
    # 距离变换：内部点到边界的距离
    dist_to_boundary = distance_transform_edt(~overlap_mask)
    
    # 截断到过渡宽度
    dist_to_boundary = np.clip(dist_to_boundary, 0, transition_width)
    
    # 归一化权重：0（边界）→ 1（内部深处）
    weight = dist_to_boundary / transition_width
    weight[~overlap_mask] = 0
    
    return weight

def compute_weight_gaussian(overlap_mask, sigma):
    """高斯模糊加权"""
    weight = overlap_mask.astype(float)
    weight_smooth = gaussian_filter(weight, sigma=sigma)
    
    # 归一化到 [0, 1]
    weight_max = np.max(weight_smooth[overlap_mask])
    if weight_max > 0:
        weight_smooth = weight_smooth / weight_max
    
    # 限制范围
    weight_smooth = np.clip(weight_smooth, 0, 1)
    
    return weight_smooth

def compute_weight_sine(overlap_mask, transition_width):
    """正弦曲线加权（最平滑的过渡）"""
    # 先计算线性权重
    weight_linear = compute_weight_linear(overlap_mask, transition_width)
    
    # 应用正弦变换：w = 0.5 - 0.5*cos(π*w_linear)
    weight = 0.5 - 0.5 * np.cos(np.pi * weight_linear)
    
    return weight

# ============================================================
# 主程序
# ============================================================

print("=" * 60)
print("加权平均拼接海陆重力数据")
print("=" * 60)

# 1. 读取数据
print("\n1. 读取数据...")
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)

print(f"   Land: {land_data.shape}, 有效点: {np.sum(~np.isnan(land_data))}")
print(f"   Sea:  {sea_data.shape}, 有效点: {np.sum(~np.isnan(sea_data))}")

# 2. 对齐 Sea 到 Land 网格
print("\n2. 对齐 Sea 到 Land 网格...")
sea_aligned = align_sea_to_land(sea_data, sea_header, land_header)
print(f"   对齐后 Sea 有效点: {np.sum(~np.isnan(sea_aligned))}")

# 3. 创建重叠区掩码
print("\n3. 识别重叠区...")
overlap_mask = ~np.isnan(sea_aligned) & ~np.isnan(land_data)
print(f"   重叠区点数: {np.sum(overlap_mask)}")

# 非重叠区
land_only_mask = ~np.isnan(land_data) & np.isnan(sea_aligned)
sea_only_mask = np.isnan(land_data) & ~np.isnan(sea_aligned)
print(f"   陆区独有点数: {np.sum(land_only_mask)}")
print(f"   海区独有点数: {np.sum(sea_only_mask)}")

# 4. 计算过渡权重
print(f"\n4. 计算过渡权重 (模式: {TRANSITION_MODE})...")

if TRANSITION_MODE == 'linear':
    weight = compute_weight_linear(overlap_mask, TRANSITION_WIDTH)
    print(f"   过渡带宽度: {TRANSITION_WIDTH} 像素 (~{TRANSITION_WIDTH * 300 / 1000:.1f} km)")
    
elif TRANSITION_MODE == 'gaussian':
    weight = compute_weight_gaussian(overlap_mask, SIGMA)
    print(f"   Sigma: {SIGMA} 像素 (~{SIGMA * 300 / 1000:.1f} km)")
    
elif TRANSITION_MODE == 'sine':
    weight = compute_weight_sine(overlap_mask, TRANSITION_WIDTH)
    print(f"   过渡带宽度: {TRANSITION_WIDTH} 像素 (~{TRANSITION_WIDTH * 300 / 1000:.1f} km)")
    
else:
    raise ValueError(f"未知的过渡模式: {TRANSITION_MODE}")

print(f"   权重范围: [{np.min(weight):.4f}, {np.max(weight):.4f}]")

# 5. 加权平均合并
print("\n5. 加权平均合并...")

# 初始化结果数组
result = np.full_like(land_data, np.nan)

# 非重叠区：直接使用原始值
result[land_only_mask] = land_data[land_only_mask]
result[sea_only_mask] = sea_aligned[sea_only_mask]

# 重叠区：加权平均
overlap_rows, overlap_cols = np.where(overlap_mask)
for r, c in zip(overlap_rows, overlap_cols):
    w = weight[r, c]
    result[r, c] = (1 - w) * land_data[r, c] + w * sea_aligned[r, c]

print(f"   重叠区加权平均完成")

# 6. 统计结果
print("\n6. 结果统计:")
print(f"   最终有效点: {np.sum(~np.isnan(result))}")
print(f"   最小值: {np.nanmin(result):.2f} mGal")
print(f"   最大值: {np.nanmax(result):.2f} mGal")
print(f"   平均值: {np.nanmean(result):.2f} mGal")
print(f"   标准差: {np.nanstd(result):.6f} mGal")

# 7. 保存结果
print("\n7. 保存结果...")
write_asc_grid(output_file, result, land_header)
print(f"   ✓ 已保存: {output_file}")

# 8. 对比验证
print("\n8. 与原始数据对比:")
print(f"   原始 Land 均值: {np.nanmean(land_data):.2f} mGal")
print(f"   原始 Sea 均值:  {np.nanmean(sea_aligned):.2f} mGal")
print(f"   最终结果均值:   {np.nanmean(result):.2f} mGal")

print("\n" + "=" * 60)
print("✅ 加权平均拼接完成！")
print("=" * 60)
