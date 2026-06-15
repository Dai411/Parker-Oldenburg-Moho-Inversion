import numpy as np
import os

# ===== Set File Path =====
data_dir = 'C:/.../BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
output_dir = 'C:/.../BouguerAnomaly/StitchGrids'
# =========================

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
    """将数据写入 ESRI ASCII 栅格文件"""
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

def align_sea_to_land_grid(sea_data, sea_header, land_header):
    """
    将 Sea 网格对齐到 Land 的网格尺寸和坐标
    返回：aligned_sea (Land 尺寸)
    """
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

# ===== 主程序 =====
print("=" * 60)
print("Step 1: Geometry Setup and Priority Masking (文档2 Step 1)")
print("=" * 60)

# 1. 读取头文件
print("\n1. 读取头文件...")
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

print(f"   Land 网格: {land_header['ncols']} x {land_header['nrows']}")
print(f"   Sea 网格:  {sea_header['ncols']} x {sea_header['nrows']}")
print(f"   cellsize: {land_header['cellsize']} m (一致)")

# 2. 读取原始数据
print("\n2. 读取原始重力数据...")
land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)
print(f"   Land 有效点: {np.sum(~np.isnan(land_data))}")
print(f"   Sea 有效点:  {np.sum(~np.isnan(sea_data))}")

# 3. 将 Sea 对齐到 Land 网格（定义全局域 Ω）
print("\n3. 对齐 Sea 到 Land 网格 (定义全局域 Ω)...")
sea_aligned = align_sea_to_land_grid(sea_data, sea_header, land_header)

# 4. 创建优先级掩码 m(x,y) - 公式 (11)
print("\n4. 创建优先级掩码 m(x,y) - 公式 (11)...")
print("   规则: Sea 优先 (mask=1), Land 填充 (mask=0), 无数据 (mask=NaN)")

mask = np.full_like(land_data, np.nan, dtype=np.float32)

# 条件 1: (x,y) ∈ Sea 网格 → mask = 1
mask[~np.isnan(sea_aligned)] = 1.0

# 条件 2: (x,y) ∈ Land 网格 且 Sea 为 NaN → mask = 0
mask[(~np.isnan(land_data)) & np.isnan(sea_aligned)] = 0.0

# 条件 3: 其他 → mask = NaN (已经初始化为 NaN)

print(f"   mask=1 (Sea 优先区域): {np.sum(mask == 1):,}")
print(f"   mask=0 (Land 填充区域): {np.sum(mask == 0):,}")
print(f"   mask=NaN (无数据区域): {np.sum(np.isnan(mask)):,}")

# 5. 创建初始复合场 g0(x,y) - 公式 (12)
print("\n5. 创建初始复合场 g0(x,y) - 公式 (12)...")
g0 = np.full_like(land_data, np.nan)

# g0 = g_sea if mask = 1
g0[mask == 1] = sea_aligned[mask == 1]

# g0 = g_land if mask = 0
g0[mask == 0] = land_data[mask == 0]

# g0 = NaN elsewhere (already)

print(f"   g0 有效点: {np.sum(~np.isnan(g0)):,}")

# 6. 保存结果
print("\n6. 保存中间结果...")

# 保存优先级掩码
mask_file = os.path.join(output_dir, 'PriorityMask.asc')
write_asc_grid(mask_file, mask, land_header)
print(f"   ✓ 优先级掩码: {mask_file}")

# 保存初始复合场（仅用于可视化，不用于后续合并）
g0_file = os.path.join(output_dir, 'InitialComposite_g0.asc')
write_asc_grid(g0_file, g0, land_header)
print(f"   ✓ 初始复合场 g0: {g0_file}")

# 保存对齐后的 Sea 数据（供后续步骤使用）
sea_aligned_file = os.path.join(output_dir, 'Sea_Aligned_To_LandGrid.asc')
write_asc_grid(sea_aligned_file, sea_aligned, land_header)
print(f"   ✓ Sea 对齐数据: {sea_aligned_file}")

# 7. 统计信息（识别间隙）
print("\n7. 间隙分析 (Ω_gap)...")
# 间隙区域：mask 有效但 g0 无效？实际上，根据文档，间隙是在 Laplacian 合并时
# 由于 stencil 边界产生的 NaN 区域。这里先记录有效边界。

# 找出海陆边界（mask 从 1 到 0 的过渡区域）
from scipy.ndimage import binary_dilation, binary_erosion

mask_sea_region = (mask == 1)
mask_land_region = (mask == 0)

# 海陆边界：海区膨胀后与陆区相交的区域
boundary = binary_dilation(mask_sea_region, iterations=1) & mask_land_region
print(f"   海陆边界点数（潜在缝合线）: {np.sum(boundary):,}")

print("\n✅ Step 1 完成！")
print("\n📌 下一步: Step 2 - 独立 Laplacian 计算")
