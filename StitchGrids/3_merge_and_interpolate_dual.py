import numpy as np
import os
import time
from scipy.interpolate import griddata
from scipy.ndimage import label, distance_transform_edt

# ===== Set File Path=====
output_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
mask_file = os.path.join(output_dir, 'PriorityMask.asc')
land_laplacian_file = os.path.join(output_dir, 'Land_Laplacian.asc')
sea_laplacian_aligned_file = os.path.join(output_dir, 'Sea_Laplacian_Aligned.asc')
# ========================

# ===== Interpolation Method Selection =====
INTERP_MODE = 'local'  # Optional: 'global' or 'local'
PADDING = 30           # Number of neighbor expansion grids in local mode
# ==========================================

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


def filter_outliers(data, valid_mask, sigma=3, min_points=10):
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    filtered = valid_mask & (np.abs(data - mean_val) <= sigma * std_val)
    if np.sum(filtered) < min_points:
        return valid_mask
    return filtered


def fill_remaining_nans(L_filled):
    nan_mask = np.isnan(L_filled)
    if np.sum(nan_mask) == 0:
        return L_filled
    if np.sum(~nan_mask) == 0:
        return L_filled
    indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
    L_filled[nan_mask] = L_filled[tuple(indices)][nan_mask]
    return L_filled


def interpolate_global(L0, gap_mask):
    """
    Global interpolation: Interpolation upon the who grids
    Suitable for small grids and widely-distributed gaps
    """
    print("   Global Interpolation...")
    
    ny, nx = L0.shape
    
    # Valid points（No NaN and not in Gap）
    valid_mask = ~np.isnan(L0)
    
    if np.sum(valid_mask) == 0:
        print("   Error: No valid points")
        return L0
    
    # Remove/Clean errors （3-times STD）
    mean_val = np.nanmean(L0)
    std_val = np.nanstd(L0)
    valid_mask_clean = valid_mask & (np.abs(L0 - mean_val) <= 3 * std_val)
    
    if np.sum(valid_mask_clean) < 10:
        valid_mask_clean = valid_mask
    
    print(f"   Valid Interpolation Points: {np.sum(valid_mask_clean):,}")
    
    # Build Grid Coordinate
    x = np.arange(nx)
    y = np.arange(ny)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Extract Valid Points
    valid_y, valid_x = np.where(valid_mask_clean)
    valid_vals = L0[valid_mask_clean]
    
    # Global Interpolation
    points = np.column_stack([valid_x, valid_y])
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    print("   执行全局插值（可能需要几分钟,‘cubic’换成‘linear’可快速验证）...")
    interpolated = griddata(points, valid_vals, grid_points, 
                            method='cubic', fill_value=np.nan)
    
    L_filled = interpolated.reshape(ny, nx)
    
    # 保留原始有效点
    L_filled[valid_mask] = L0[valid_mask]
    
    return L_filled

def interpolate_local(L0, gap_mask, padding=30):
    """
    局部插值：只对每个间隙区域的扩展邻域进行插值
    适用于大网格、间隙集中的情况
    """
    print(f"   使用局部插值模式，padding={padding}...")
    
    # 标记连通间隙区域
    labeled_gaps, num_gaps = label(gap_mask)
    print(f"   找到 {num_gaps} 个连通间隙区域")
    
    L_filled = L0.copy()
    
    for gap_id in range(1, num_gaps + 1):
        if gap_id % 1000 == 0:
            print(f"     处理间隙 {gap_id}/{num_gaps}...")
        
        region_mask = (labeled_gaps == gap_id)
        
        rows, cols = np.where(region_mask)
        if len(rows) == 0:
            continue
        
        # 区域边界（扩展 padding）
        r_min = max(0, np.min(rows) - padding)
        r_max = min(L0.shape[0], np.max(rows) + padding + 1)
        c_min = max(0, np.min(cols) - padding)
        c_max = min(L0.shape[1], np.max(cols) + padding + 1)
        
        # 提取局部区域
        local_region = L0[r_min:r_max, c_min:c_max].copy()
        local_gap = region_mask[r_min:r_max, c_min:c_max]
        
        # 局部区域内的有效点
        local_valid_mask = ~np.isnan(local_region) & ~local_gap
        
        if np.sum(local_valid_mask) < 4:
            continue
        
        # 局部有效点去除异常值
        local_valid_mask_clean = filter_outliers(local_region, local_valid_mask)
        if np.sum(local_valid_mask_clean) < 4:
            local_valid_mask_clean = local_valid_mask
        
        # 需要插值的位置
        interp_rows, interp_cols = np.where(local_gap)
        
        if len(interp_rows) == 0:
            continue
        
        try:
            local_valid_rows, local_valid_cols = np.where(local_valid_mask_clean)
            local_vals = local_region[local_valid_mask_clean]
            
            points = np.column_stack([local_valid_cols, local_valid_rows])
            interp_points = np.column_stack([interp_cols, interp_rows])
            
            interpolated = griddata(points, local_vals, interp_points, 
                                    method='cubic', fill_value=np.nan)
            
            for idx, (r, c) in enumerate(zip(interp_rows, interp_cols)):
                if not np.isnan(interpolated[idx]):
                    L_filled[r_min + r, c_min + c] = interpolated[idx]
                    
        except Exception:
            continue
    
    return L_filled

# ===== 主程序 =====
print("=" * 60)
print("Step 3+4: Merge Laplacian + Interpolate Gaps (文档2 Step 3-4)")
print("=" * 60)
print(f"\n插值模式: {INTERP_MODE.upper()}")
if INTERP_MODE == 'local':
    print(f"  Padding: {PADDING} 格 (~{PADDING * 300 / 1000:.1f} km)")

# 1. 读取数据
print("\n1. 读取数据...")
mask_header = read_asc_header(mask_file)
mask = read_asc_grid(mask_file, mask_header)

land_header = read_asc_header(land_laplacian_file)
land_laplacian = read_asc_grid(land_laplacian_file, land_header)

sea_laplacian = read_asc_grid(sea_laplacian_aligned_file, land_header)

print(f"   Mask 形状: {mask.shape}")
print(f"   Land Laplacian 形状: {land_laplacian.shape}")

# 2. Step 3: 合并 Laplacian - 公式 (15)
print("\n2. Step 3: 合并 Laplacian (公式 15)...")

L0 = np.full_like(land_laplacian, np.nan)

sea_mask = (mask == 1)
L0[sea_mask] = sea_laplacian[sea_mask]

land_mask = (mask == 0)
L0[land_mask] = land_laplacian[land_mask]

print(f"   L0 有效点: {np.sum(~np.isnan(L0)):,}")
print(f"   L0 中 NaN: {np.sum(np.isnan(L0)):,}")

# 3. 识别间隙 Ω_gap
print("\n3. 识别间隙 Ω_gap...")
gap_mask = (~np.isnan(mask)) & np.isnan(L0)
print(f"   间隙点数 (Ω_gap): {np.sum(gap_mask):,}")

if np.sum(gap_mask) == 0:
    print("   没有间隙，无需插值")
    L_filled = L0
else:
    # 4. Step 4: 插值填充间隙
    print("\n4. Step 4: 插值填充间隙...")
    
    # 保存插值前的 L0
    L0_before_file = os.path.join(output_dir, 'L0_MergedLaplacian_BeforeInterp.asc')
    write_asc_grid(L0_before_file, L0, land_header)
    print(f"   ✓ 插值前 L0 已保存: {L0_before_file}")
    
    # 根据模式选择插值方法
    start_time = time.time()
    
    if INTERP_MODE == 'global':
        L_filled = interpolate_global(L0, gap_mask)
    else:  # local
        L_filled = interpolate_local(L0, gap_mask, padding=PADDING)
    
    elapsed = time.time() - start_time
    print(f"   插值耗时: {elapsed:.2f} 秒")

    remaining_nans_before = np.sum(np.isnan(L_filled))
    if remaining_nans_before > 0:
        print(f"   ⚠️ 插值后仍有 {remaining_nans_before:,} 个 NaN，使用最近邻填充")
        L_filled = fill_remaining_nans(L_filled)
        remaining_nans_after = np.sum(np.isnan(L_filled))
        print(f"   最近邻填充后 NaN: {remaining_nans_after:,}")

# 5. 保存结果
print("\n5. 保存插值后的 Laplacian...")
L_filled_file = os.path.join(output_dir, 'L_FilledLaplacian.asc')
write_asc_grid(L_filled_file, L_filled, land_header)
print(f"   ✓ 插值后 Laplacian: {L_filled_file}")

# 6. 统计插值效果
print("\n6. 插值效果统计:")
print(f"   插值前 NaN: {np.sum(np.isnan(L0)):,}")
print(f"   插值后 NaN: {np.sum(np.isnan(L_filled)):,}")
print(f"   成功填充: {np.sum(np.isnan(L0)) - np.sum(np.isnan(L_filled)):,}")

if np.sum(np.isnan(L_filled)) > 0:
    print(f"   ⚠️ 警告：仍有 {np.sum(np.isnan(L_filled)):,} 个 NaN 未被填充")

print("\n✅ Step 3+4 完成！")
print("\n📌 下一步: Step 5 - 频域反解得到最终布格异常")
