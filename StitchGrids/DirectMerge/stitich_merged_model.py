# stitich_merged_model.py
"""
高精度数据 + 模型数据 合并 (核心区保护版)
- 核心区：完全保留我们的数据 (不被修改)
- 过渡带：模型数据向我们的数据平滑过渡
- 外部：完全保留模型数据

核心区保护策略：通过掩码确保我们的数据不被任何操作修改
"""

import numpy as np
import os
from scipy.ndimage import gaussian_filter, binary_erosion

# ============================================================
# 参数配置 (可调)
# ============================================================

# 过渡带宽度 (像素)
#   建议范围: 10 ~ 50 像素
#   模型网格间距 300m:
#       10像素 = 3.0 km
#       20像素 = 6.0 km (推荐起始值)
#       30像素 = 9.0 km
TRANSITION_WIDTH = 20  # 像素

# 高斯平滑 sigma (自动计算)
SIGMA = TRANSITION_WIDTH / 2  # 像素

# 核心区保护内边距 (像素)
#   我们的数据边界向内收缩多少像素，形成"绝对保护区"
#   建议: 5 ~ 15 像素
#   值越大，保护越强，但过渡带越窄
SAFETY_PADDING = 10  # 像素 (从 2 增加到 10)

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
model_file = os.path.join(data_dir, 'BouguerModelled.asc')
stitch_dir = os.path.join(data_dir, 'StitchGrids')
our_file = os.path.join(stitch_dir, 'BouguerInterpGap.asc')
output_file = os.path.join(stitch_dir, 'BouguerFinalWithModel.asc')

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
    """将我们的数据对齐到模型网格"""
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
print("高精度数据 + 模型数据 合并 (核心区保护版)")
print("=" * 60)
print(f"  过渡带宽度: {TRANSITION_WIDTH} 像素 (~{TRANSITION_WIDTH * 300 / 1000:.1f} km)")
print(f"  高斯 sigma: {SIGMA:.1f} 像素")
print(f"  核心区保护内边距: {SAFETY_PADDING} 像素 (~{SAFETY_PADDING * 300 / 1000:.1f} km)")
print("=" * 60)

# 1. 读取数据
print("\n1. 读取数据...")
model_header = read_asc_header(model_file)
our_header = read_asc_header(our_file)

model_data = read_asc_grid(model_file, model_header)
our_data = read_asc_grid(our_file, our_header)

print(f"   模型数据: {model_data.shape}, 有效点: {np.sum(~np.isnan(model_data))}")
print(f"   我们的数据: {our_data.shape}, 有效点: {np.sum(~np.isnan(our_data))}")

# 2. 对齐我们的数据到模型网格
print("\n2. 对齐我们的数据到模型网格...")
our_aligned = align_grid_to_model(our_data, our_header, model_header)
print(f"   对齐后有效点: {np.sum(~np.isnan(our_aligned))}")

# 3. 创建权重掩码 (模型向我们的数据过渡)
print("\n3. 创建权重掩码...")

# 我们的数据有效区掩码
our_valid_mask = ~np.isnan(our_aligned)

# ========== 核心区保护 ==========
# 核心区：我们的数据内部，不受任何平滑影响
core_mask = binary_erosion(our_valid_mask, iterations=SAFETY_PADDING)

# ========== 过渡带权重 ==========
# 初始权重：核心区为 1，其他地方为 0
weight_initial = np.zeros_like(model_data)
weight_initial[core_mask] = 1.0

# 高斯平滑生成过渡权重
weight_clean = np.nan_to_num(weight_initial, nan=0.0)
weight_smooth = gaussian_filter(weight_clean, sigma=SIGMA)
weight_smooth = np.clip(weight_smooth, 0, 1)

# ========== 约束：我们的数据有效区内，权重不低于 0.5 ==========
# 但核心区已经被保护，这里确保过渡带向我们的数据侧权重足够高
weight_smooth[our_valid_mask] = np.maximum(weight_smooth[our_valid_mask], 0.5)

# 核心区强制权重 = 1 (双重保险)
weight_smooth[core_mask] = 1.0

print(f"   我们的数据总有效点: {np.sum(our_valid_mask)}")
print(f"   核心区点数 (完全保护): {np.sum(core_mask)}")
print(f"   过渡带点数: {np.sum((weight_smooth > 0) & (weight_smooth < 1))}")
print(f"   权重范围: [{np.nanmin(weight_smooth):.4f}, {np.nanmax(weight_smooth):.4f}]")

# 4. 合并数据
print("\n4. 合并数据...")

# 初始化结果：先用模型数据
result = model_data.copy()

# 先在我们的数据有效区放入我们的数据
result[our_valid_mask] = our_aligned[our_valid_mask]

# 过渡带加权平均 (核心区被排除，因为权重=1)
transition_mask = (weight_smooth > 0) & (weight_smooth < 1) & our_valid_mask & ~core_mask
transition_rows, transition_cols = np.where(transition_mask)

print(f"   过渡带内加权点数: {len(transition_rows)}")

for r, c in zip(transition_rows, transition_cols):
    w = weight_smooth[r, c]
    model_val = model_data[r, c]
    our_val = our_aligned[r, c]
    if not np.isnan(model_val) and not np.isnan(our_val):
        # 加权平均：w 越大，我们的数据权重越大
        result[r, c] = (1 - w) * model_val + w * our_val

# 双重保险：强制核心区完全等于我们的数据
result[core_mask] = our_aligned[core_mask]

# 5. 验证
print("\n5. 验证...")

# 检查核心区是否完全保留我们的值
if np.sum(core_mask) > 0:
    core_unchanged = np.allclose(result[core_mask], our_aligned[core_mask], equal_nan=True)
    print(f"   核心区保留我们的值: {'✓ 通过' if core_unchanged else '✗ 失败'}")
else:
    print("   核心区为空，跳过验证")

# 检查我们的数据有效区内是否都被我们的值覆盖 (允许过渡带混合)
our_area_correct = np.allclose(result[our_valid_mask], our_aligned[our_valid_mask], 
                                equal_nan=True, rtol=1e-5, atol=1e-5)
print(f"   我们的数据区一致性: {'✓ 通过' if our_area_correct else '⚠ 过渡带被混合'}")

# 检查外部是否完全保留模型值
external_mask = ~our_valid_mask & ~np.isnan(model_data)
if np.sum(external_mask) > 0:
    external_unchanged = np.allclose(result[external_mask], model_data[external_mask], equal_nan=True)
    print(f"   外部保留模型值: {'✓ 通过' if external_unchanged else '✗ 失败'}")
else:
    print("   外部为空，跳过验证")

# 6. 统计结果
print("\n6. 结果统计:")
print(f"   最终有效点: {np.sum(~np.isnan(result))}")
print(f"   最小值: {np.nanmin(result):.2f} mGal")
print(f"   最大值: {np.nanmax(result):.2f} mGal")
print(f"   平均值: {np.nanmean(result):.2f} mGal")
print(f"   标准差: {np.nanstd(result):.6f} mGal")

# 分区域统计
if np.sum(core_mask) > 0:
    print(f"\n   核心区均值 (我们的数据): {np.nanmean(result[core_mask]):.2f} mGal")
if np.sum(transition_mask) > 0:
    print(f"   过渡带均值: {np.nanmean(result[transition_mask]):.2f} mGal")
if np.sum(external_mask) > 0:
    print(f"   外部均值 (模型数据): {np.nanmean(result[external_mask]):.2f} mGal")

# 检查两个数据在过渡带边界的差异
boundary_mask = (weight_smooth > 0.45) & (weight_smooth < 0.55) & our_valid_mask & ~core_mask
if np.sum(boundary_mask) > 0:
    diff = np.abs(our_aligned[boundary_mask] - model_data[boundary_mask])
    print(f"\n   边界处 (权重≈0.5) 我们的数据与模型差异:")
    print(f"      均值 = {np.nanmean(diff):.2f} mGal")
    print(f"      最大 = {np.nanmax(diff):.2f} mGal")
    print(f"      标准差 = {np.nanstd(diff):.2f} mGal")

# 7. 保存
print("\n7. 保存结果...")
os.makedirs(stitch_dir, exist_ok=True)
write_asc_grid(output_file, result, model_header)
print(f"   ✓ 已保存: {output_file}")

print("\n" + "=" * 60)
print("✅ 合并完成！")
print("   核心区: 完全保留我们的高精度数据")
print("   过渡带: 模型数据向我们的数据平滑过渡")
print("   外部: 完全保留模型数据")
print("=" * 60)
