import numpy as np
import os
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation

# ===== 设置路径 =====
output_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
mask_file = os.path.join(output_dir, 'PriorityMask.asc')
land_laplacian_file = os.path.join(output_dir, 'Land_Laplacian.asc')
sea_laplacian_aligned_file = os.path.join(output_dir, 'Sea_Laplacian_Aligned.asc')
# ====================

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

def bicubic_interpolate_laplacian(L_merged, mask):
    """
    双三次样条插值填充 Laplacian 中的 NaN 间隙 (Ω_gap)
    使用 scipy 的 RectBivariateSpline
    """
    # 找到有效数据点
    valid_mask = ~np.isnan(L_merged)
    
    if np.sum(valid_mask) == 0:
        print("   错误：没有有效 Laplacian 数据")
        return L_merged
    
    # 创建坐标网格
    ny, nx = L_merged.shape
    x = np.arange(nx)
    y = np.arange(ny)
    
    # 提取有效点的坐标和值
    valid_x, valid_y = np.where(valid_mask)
    valid_values = L_merged[valid_mask]
    
    # 检查有效点是否足够进行插值
    if len(valid_values) < 20:
        print(f"   警告：有效点太少 ({len(valid_values)})，跳过插值")
        return L_merged
    
    # 创建样条插值器
    # 注意：RectBivariateSpline 要求输入是规则的网格数据
    # 我们创建一个临时网格，只包含有效数据点
    
    # 为了稳定插值，对数据进行简单的预处理
    # 去除异常值（大于 3 倍标准差的点）
    mean_val = np.nanmean(L_merged)
    std_val = np.nanstd(L_merged)
    valid_mask_clean = valid_mask & (np.abs(L_merged - mean_val) <= 3 * std_val)
    
    if np.sum(valid_mask_clean) < 20:
        valid_mask_clean = valid_mask
    
    # 创建插值函数
    try:
        # 构建插值器需要完整网格，我们使用 griddata 更合适
        from scipy.interpolate import griddata
        
        # 准备网格点
        grid_x, grid_y = np.meshgrid(x, y)
        
        # 提取清理后的有效点
        clean_x, clean_y = np.where(valid_mask_clean)
        clean_vals = L_merged[valid_mask_clean]
        
        # 使用 griddata 进行插值（更稳定）
        points = np.column_stack([clean_y, clean_x])  # (x, y) 顺序
        grid_points = np.column_stack([grid_y.ravel(), grid_x.ravel()])
        
        interpolated = griddata(points, clean_vals, grid_points, 
                                method='cubic', fill_value=np.nan)
        
        L_filled = interpolated.reshape(ny, nx)
        
        # 保留原始有效点不变
        L_filled[valid_mask] = L_merged[valid_mask]
        
        # 检查插值效果
        nan_after = np.sum(np.isnan(L_filled))
        print(f"   插值前 NaN: {np.sum(np.isnan(L_merged)):,}")
        print(f"   插值后 NaN: {nan_after:,}")
        print(f"   填充点数: {np.sum(np.isnan(L_merged)) - nan_after:,}")
        
        return L_filled
        
    except Exception as e:
        print(f"   插值失败: {e}")
        return L_merged

# ===== 主程序 =====
print("=" * 60)
print("Step 3+4: Merge Laplacian + Interpolate Gaps (文档2 Step 3-4)")
print("=" * 60)

# 1. 读取数据
print("\n1. 读取数据...")
mask_header = read_asc_header(mask_file)
mask = read_asc_grid(mask_file, mask_header)

land_header = read_asc_header(land_laplacian_file)
land_laplacian = read_asc_grid(land_laplacian_file, land_header)

sea_laplacian = read_asc_grid(sea_laplacian_aligned_file, land_header)

print(f"   Mask 形状: {mask.shape}")
print(f"   Land Laplacian 形状: {land_laplacian.shape}")
print(f"   Sea Laplacian 形状: {sea_laplacian.shape}")

# 2. Step 3: 合并 Laplacian - 公式 (15)
print("\n2. Step 3: 合并 Laplacian (公式 15)...")
print("   规则: L0 = L_sea if mask=1, L_land if mask=0, NaN elsewhere")

L0 = np.full_like(land_laplacian, np.nan)

# mask=1 (Sea 优先) -> 使用 Sea Laplacian
sea_mask = (mask == 1)
L0[sea_mask] = sea_laplacian[sea_mask]

# mask=0 (Land 填充) -> 使用 Land Laplacian
land_mask = (mask == 0)
L0[land_mask] = land_laplacian[land_mask]

# mask=NaN -> 保持 NaN

print(f"   L0 有效点 (非NaN): {np.sum(~np.isnan(L0)):,}")
print(f"   L0 来自 Sea: {np.sum(~np.isnan(sea_laplacian[sea_mask])):,}")
print(f"   L0 来自 Land: {np.sum(~np.isnan(land_laplacian[land_mask])):,}")
print(f"   L0 中 NaN: {np.sum(np.isnan(L0)):,}")

# 3. 识别间隙 Ω_gap
print("\n3. 识别间隙 Ω_gap (由于 Laplacian stencil 边界产生)...")

# 间隙：mask 有效但 L0 为 NaN 的区域
# 这些是由于 Laplacian 计算时 stencil 包含边界点导致的
gap_mask = (~np.isnan(mask)) & np.isnan(L0)
print(f"   间隙点数 (Ω_gap): {np.sum(gap_mask):,}")

# 检查间隙的连通性（是否是连续的缝合带）
from scipy.ndimage import label
labeled_gaps, num_gaps = label(gap_mask)
print(f"   间隙连通区域数: {num_gaps}")

if num_gaps > 0:
    gap_sizes = [np.sum(labeled_gaps == i+1) for i in range(num_gaps)]
    print(f"   最大间隙区域大小: {max(gap_sizes):,} 点")
    print(f"   最小间隙区域大小: {min(gap_sizes):,} 点")

# 4. Step 4: 双三次样条插值填充间隙
print("\n4. Step 4: 双三次样条插值填充间隙 (公式 16)...")
print("   使用 bicubic 插值重建 Ω_gap 中的 Laplacian 值")

# 保存插值前的 L0
L0_before_file = os.path.join(output_dir, 'L0_MergedLaplacian_BeforeInterp.asc')
write_asc_grid(L0_before_file, L0, land_header)
print(f"   ✓ 插值前 L0 已保存: {L0_before_file}")

# 执行插值
L_filled = bicubic_interpolate_laplacian(L0, mask)

# 5. 保存结果
print("\n5. 保存插值后的 Laplacian...")
L_filled_file = os.path.join(output_dir, 'L_FilledLaplacian.asc')
write_asc_grid(L_filled_file, L_filled, land_header)
print(f"   ✓ 插值后 Laplacian: {L_filled_file}")

# 6. 统计插值效果
print("\n6. 插值效果统计:")
print(f"   插值前 L0 NaN 数量: {np.sum(np.isnan(L0)):,}")
print(f"   插值后 L_filled NaN 数量: {np.sum(np.isnan(L_filled)):,}")
print(f"   成功填充: {np.sum(np.isnan(L0)) - np.sum(np.isnan(L_filled)):,}")

# 检查是否仍有 NaN
remaining_nan = np.sum(np.isnan(L_filled))
if remaining_nan > 0:
    print(f"   ⚠️ 警告：仍有 {remaining_nan:,} 个 NaN 未被填充")
    # 简单的邻近填充
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(L_filled)
    if np.sum(~np.isnan(L_filled)) > 0:
        # 使用最近邻填充剩余的 NaN
        indices = distance_transform_edt(nan_mask, return_distances=False, 
                                         return_indices=True)
        L_filled[nan_mask] = L_filled[tuple(indices)][nan_mask]
        print(f"   使用最近邻填充剩余 NaN 后: {np.sum(np.isnan(L_filled)):,}")

print("\n✅ Step 3+4 完成！")
print("\n📌 下一步: Step 5 - 频域反解得到最终布格异常")
