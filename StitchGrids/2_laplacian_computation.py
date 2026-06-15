import numpy as np
import os
from scipy.signal import convolve2d

# ===== Set File Path =====
data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
output_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
# =========================

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

def compute_laplacian_fast(data, cellsize):
    """
    离散 Laplacian（5点差分）- 公式 (13)
    如果 stencil 中任意一点为 NaN，输出 NaN - 公式 (14)
    """
    # 5点差分核
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    # 标记 NaN 位置
    nan_mask = np.isnan(data)
    
    # NaN 替换为 0 以便卷积
    data_clean = np.nan_to_num(data, nan=0.0)
    
    # 卷积计算
    laplacian = convolve2d(data_clean, kernel, mode='same', boundary='symm')
    laplacian = laplacian / (cellsize * cellsize)
    
    # 扩展 NaN 掩码：5点中任意一点为 NaN 则输出 NaN
    extended_nan = nan_mask.copy()
    if nan_mask.shape[0] > 1:
        extended_nan[1:, :] |= nan_mask[:-1, :]      # 上
        extended_nan[:-1, :] |= nan_mask[1:, :]      # 下
    if nan_mask.shape[1] > 1:
        extended_nan[:, 1:] |= nan_mask[:, :-1]      # 左
        extended_nan[:, :-1] |= nan_mask[:, 1:]      # 右
    
    laplacian[extended_nan] = np.nan
    
    return laplacian

def align_grid_to_land(data, data_header, land_header):
    """
    将任意网格数据对齐到 Land 网格
    """
    land_nrows = land_header['nrows']
    land_ncols = land_header['ncols']
    land_xmin = land_header['xllcorner']
    land_ymax = land_header['yllcorner'] + land_header['nrows'] * land_header['cellsize']
    cellsize = land_header['cellsize']
    
    data_xmin = data_header['xllcorner']
    data_ymax = data_header['yllcorner'] + data_header['nrows'] * data_header['cellsize']
    
    aligned = np.full((land_nrows, land_ncols), np.nan)
    
    for i in range(data_header['nrows']):
        data_y = data_ymax - i * cellsize
        land_row = int(round((land_ymax - data_y) / cellsize))
        
        if land_row < 0 or land_row >= land_nrows:
            continue
            
        for j in range(data_header['ncols']):
            data_val = data[i, j]
            if np.isnan(data_val):
                continue
            
            data_x = data_xmin + j * cellsize
            land_col = int(round((data_x - land_xmin) / cellsize))
            
            if 0 <= land_col < land_ncols:
                aligned[land_row, land_col] = data_val
    
    return aligned

# ===== 主程序 =====
print("=" * 60)
print("Step 2: Independent Laplacian Computation (文档2 Step 2)")
print("=" * 60)

# 1. 读取头文件
print("\n1. 读取头文件...")
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

# 2. 读取原始数据
print("\n2. 读取原始重力数据...")
land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)
print(f"   Land 数据形状: {land_data.shape}")
print(f"   Sea 数据形状:  {sea_data.shape}")

# 3. 计算 Laplacian（在原始网格上，独立计算）
print("\n3. 计算离散 Laplacian - 公式 (13)...")
cellsize = land_header['cellsize']

print("   计算 Land Laplacian...")
land_laplacian = compute_laplacian_fast(land_data, cellsize)

print("   计算 Sea Laplacian...")
sea_laplacian = compute_laplacian_fast(sea_data, cellsize)

print(f"   Land Laplacian 有效点: {np.sum(~np.isnan(land_laplacian)):,}")
print(f"   Sea Laplacian 有效点:  {np.sum(~np.isnan(sea_laplacian)):,}")

# 4. 将 Sea Laplacian 对齐到 Land 网格（为 Step 3 做准备）
print("\n4. 对齐 Sea Laplacian 到 Land 网格...")
sea_laplacian_aligned = align_grid_to_land(sea_laplacian, sea_header, land_header)
print(f"   对齐后 Sea Laplacian 有效点: {np.sum(~np.isnan(sea_laplacian_aligned)):,}")

# 5. 保存结果
print("\n5. 保存 Laplacian 结果...")

land_laplacian_file = os.path.join(output_dir, 'Land_Laplacian.asc')
write_asc_grid(land_laplacian_file, land_laplacian, land_header)
print(f"   ✓ Land Laplacian: {land_laplacian_file}")

sea_laplacian_original_file = os.path.join(output_dir, 'Sea_Laplacian_Original.asc')
write_asc_grid(sea_laplacian_original_file, sea_laplacian, sea_header)
print(f"   ✓ Sea Laplacian (原始网格): {sea_laplacian_original_file}")

sea_laplacian_aligned_file = os.path.join(output_dir, 'Sea_Laplacian_Aligned.asc')
write_asc_grid(sea_laplacian_aligned_file, sea_laplacian_aligned, land_header)
print(f"   ✓ Sea Laplacian (对齐到 Land): {sea_laplacian_aligned_file}")

# 6. 统计信息
print("\n6. Laplacian 统计:")
land_l_valid = land_laplacian[~np.isnan(land_laplacian)]
sea_l_valid = sea_laplacian[~np.isnan(sea_laplacian)]

print(f"   Land Laplacian:")
print(f"       min: {np.nanmin(land_laplacian):.6f}")
print(f"       max: {np.nanmax(land_laplacian):.6f}")
print(f"       mean: {np.nanmean(land_laplacian):.6f}")
print(f"       std: {np.nanstd(land_laplacian):.6f}")

print(f"   Sea Laplacian:")
print(f"       min: {np.nanmin(sea_laplacian):.6f}")
print(f"       max: {np.nanmax(sea_laplacian):.6f}")
print(f"       mean: {np.nanmean(sea_laplacian):.6f}")
print(f"       std: {np.nanstd(sea_laplacian):.6f}")

print("\n✅ Step 2 完成！")
print("\n📌 下一步: Step 3 - 合并 Laplacian (使用优先级掩码) 并插值间隙")
