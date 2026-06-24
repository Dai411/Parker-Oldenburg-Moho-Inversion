# 7_Laplacian_with_model_constraint.py
"""
Laplacian 域结果 + 模型数据约束校正 (修正版)
- 外部区域：模型数据有效，我们的数据无效
- 用边界附近的 Laplacian 值估算外部校正量
"""

import numpy as np
import os
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

print("DEBUG: 脚本开始运行")

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
stitch_dir = os.path.join(data_dir, 'StitchGrids')

laplacian_file = os.path.join(stitch_dir, 'BouguerFinal.asc')
model_file = os.path.join(data_dir, 'BouguerModelled.asc')
our_file = os.path.join(stitch_dir, 'BouguerInterpGap.asc')
output_file = os.path.join(stitch_dir, 'BouguerLaplacianConstrained.asc')

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

def align_grid_to_model(src_data, src_header, model_header):
    """将数据对齐到模型网格"""
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

# ============================================================
# 主程序
# ============================================================

print("=" * 60)
print("Laplacian 域结果 + 模型数据约束校正 (修正版)")
print("=" * 60)

# 参数配置
TRANSITION_SIGMA = 20  # 过渡带宽度 (像素)，约 6 km
SAFETY_PADDING = 10    # 内部保护区 (像素)，约 3 km
BOUNDARY_WIDTH = 5     # 边界层宽度 (像素)，用于估算外部偏移

print(f"  过渡带 sigma: {TRANSITION_SIGMA} 像素 (~{TRANSITION_SIGMA * 300 / 1000:.1f} km)")
print(f"  内部保护区: {SAFETY_PADDING} 像素 (~{SAFETY_PADDING * 300 / 1000:.1f} km)")

# 1. 读取数据
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

# 2. 对齐到模型网格
print("\n2. 对齐到模型网格...")
our_aligned = align_grid_to_model(our_data, our_header, model_header)
lap_aligned = align_grid_to_model(lap_data, lap_header, model_header)

print(f"   对齐后我们的数据有效点: {np.sum(~np.isnan(our_aligned))}")
print(f"   对齐后Laplacian有效点: {np.sum(~np.isnan(lap_aligned))}")

# 3. 识别各区域
print("\n3. 识别区域...")

# 我们的数据有效区
our_valid_mask = ~np.isnan(our_aligned)

# 核心区 (内部保护区)
core_mask = binary_erosion(our_valid_mask, iterations=SAFETY_PADDING)

# 边界层 (我们的数据边缘，用于估算外部偏移)
boundary_mask = binary_dilation(our_valid_mask, iterations=BOUNDARY_WIDTH) & ~our_valid_mask
boundary_mask = boundary_mask & ~np.isnan(lap_aligned)

# 外部区域：模型数据有效，我们的数据无效
external_mask = ~np.isnan(model_data) & ~our_valid_mask

# 过渡带：我们的数据有效但不在核心区
transition_mask = our_valid_mask & ~core_mask

print(f"   核心区点数: {np.sum(core_mask)}")
print(f"   过渡带点数: {np.sum(transition_mask)}")
print(f"   边界层点数 (用于估算): {np.sum(boundary_mask)}")
print(f"   外部区域点数: {np.sum(external_mask)}")

# 4. 计算外部区域的校正量
print("\n4. 计算外部区域校正量...")

# 方法：用边界层的 Laplacian 值，与模型数据比较
if np.sum(boundary_mask) > 0:
    # 在边界层计算差异
    boundary_diff = model_data[boundary_mask] - lap_aligned[boundary_mask]
    offset_global = np.nanmean(boundary_diff)
    offset_std = np.nanstd(boundary_diff)
    print(f"   边界层差异 (Laplacian vs 模型): 均值 = {offset_global:.4f} mGal, 标准差 = {offset_std:.4f} mGal")
else:
    print("   警告: 边界层为空，使用默认偏移 0")
    offset_global = 0.0

# 5. 创建过渡权重
print("\n5. 创建过渡权重...")

# 权重：核心区=1，外部=0
weight_initial = np.zeros_like(model_data)
weight_initial[core_mask] = 1.0

# 高斯平滑
weight_smooth = gaussian_filter(weight_initial, sigma=TRANSITION_SIGMA)
weight_smooth = np.clip(weight_smooth, 0, 1)
weight_smooth[core_mask] = 1.0

print(f"   权重范围: [{np.nanmin(weight_smooth):.4f}, {np.nanmax(weight_smooth):.4f}]")

# 6. 应用校正
print("\n6. 应用模型约束校正...")

# 生成偏移场 (外部完全偏移，内部无偏移)
# 注意：lap_aligned 在外部可能是 NaN，需要用模型数据填充
offset_field = offset_global * (1 - weight_smooth)

# 初始化结果
g_corrected = lap_aligned.copy()

# 在外部区域，先用模型数据
g_corrected[external_mask] = model_data[external_mask]

# 在过渡带，应用偏移
transition_rows, transition_cols = np.where(transition_mask)
for r, c in zip(transition_rows, transition_cols):
    w = weight_smooth[r, c]
    lap_val = lap_aligned[r, c]
    model_val = model_data[r, c]
    if not np.isnan(lap_val) and not np.isnan(model_val):
        g_corrected[r, c] = (1 - w) * model_val + w * lap_val
    elif not np.isnan(lap_val):
        g_corrected[r, c] = lap_val
    elif not np.isnan(model_val):
        g_corrected[r, c] = model_val

# 核心区强制保留 Laplacian 值
g_corrected[core_mask] = lap_aligned[core_mask]

# 7. 统计结果
print("\n7. 结果统计:")

print(f"\n   校正前 Laplacian 结果:")
print(f"      全场均值: {np.nanmean(lap_aligned):.2f} mGal")
if np.sum(core_mask) > 0:
    print(f"      核心区均值: {np.nanmean(lap_aligned[core_mask]):.2f} mGal")

print(f"\n   校正后结果:")
print(f"      全场均值: {np.nanmean(g_corrected):.2f} mGal")
if np.sum(core_mask) > 0:
    print(f"      核心区均值: {np.nanmean(g_corrected[core_mask]):.2f} mGal")
if np.sum(external_mask) > 0:
    print(f"      外部区域均值: {np.nanmean(g_corrected[external_mask]):.2f} mGal")

# 8. 验证
print("\n8. 验证...")

if np.sum(core_mask) > 0:
    core_unchanged = np.allclose(g_corrected[core_mask], lap_aligned[core_mask], equal_nan=True)
    print(f"   核心区保持不变: {'✓ 通过' if core_unchanged else '✗ 失败'}")

if np.sum(external_mask) > 0:
    # 外部区域应该等于模型数据
    external_match = np.allclose(g_corrected[external_mask], model_data[external_mask], 
                                  equal_nan=True, rtol=1e-5)
    print(f"   外部区域等于模型: {'✓ 通过' if external_match else '✗ 失败'}")

# 9. 保存
print("\n9. 保存结果...")
write_asc_grid(output_file, g_corrected, model_header)
print(f"   ✓ 已保存: {output_file}")

print("\n" + "=" * 60)
print("✅ Laplacian + 模型约束完成！")
print("=" * 60)
