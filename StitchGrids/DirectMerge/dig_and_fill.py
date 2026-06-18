# dig_and_fill.py
"""
Dig the overlapped region and interpolate the digged
- Only the overlapped region will be re-interpolated
- Non-overlapped region keep the original value
- Avoid the global effect from the Laplacian domain

The smoothness of different interpolations:
    linear            : C⁰ continuous (first derivative discontinuous, has creases/kinks)
    cubic             : C¹ continuous (first derivative continuous, second derivative discontinuous)
    clough_tocher     : C¹ continuous (triangulation-based, suitable for irregular boundaries)
    rbf (cubic)       : C∞ continuous (infinitely smooth, but computationally slow)
    thin_plate_spline : C² continuous (second derivative continuous, suitable for terrain data)

Recommendations: clough_tocher (C¹) 或 cubic (C¹)
"""

import numpy as np
import os
import time
from scipy.ndimage import binary_dilation
from scipy.interpolate import CloughTocher2DInterpolator, griddata

# ============================================================
# Controlling Parameters
# ============================================================

# 拼接带向外扩展宽度 (像素)
#   建议范围: 3 ~ 20 像素
#   你的网格间距 300m:
#       3像素 = 0.9 km (窄带，快速)
#       5像素 = 1.5 km (中等)
#       10像素 = 3.0 km (推荐起始值)
#       20像素 = 6.0 km (宽带，平滑但耗时)
GAP_PADDING = 10  # 像素

# 插值方法 (可选: 'clough_tocher', 'cubic', 'linear', 'rbf')
#   clough_tocher : C¹ 连续，三角形剖分，推荐 (平衡速度和平滑度)
#   cubic         : C¹ 连续，griddata 实现，简单稳定
#   linear        : C⁰ 连续，最快但不平滑
#   rbf           : C∞ 连续，最平滑但计算慢 (大数据不推荐)
INTERP_METHOD = 'clough_tocher'  # 推荐

# 进度显示间隔 (每处理 N 个点输出一次)
PROGRESS_INTERVAL = 200000  # 每 20 万点输出一次

# RBF 参数 (仅当 INTERP_METHOD='rbf' 时使用)
RBF_KERNEL = 'cubic'     # 核函数: 'cubic', 'thin_plate_spline', 'gaussian'
RBF_EPSILON = 2.0        # 平滑参数

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
output_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
output_file = os.path.join(output_dir, 'BouguerInterpGap.asc')

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
# 带进度的填充函数
# ============================================================

def fill_with_progress(g_final, gap_rows, gap_cols, interpolated, interval=200000):
    """带进度显示的填充"""
    n_points = len(gap_rows)
    print(f"   开始填充 {n_points:,} 个点 (每 {interval:,} 点输出一次进度)...")
    
    fill_count = 0
    last_progress = 0
    
    for idx, (r, c) in enumerate(zip(gap_rows, gap_cols)):
        if not np.isnan(interpolated[idx]):
            g_final[r, c] = interpolated[idx]
            fill_count += 1
        
        # 进度显示
        if (idx + 1) % interval == 0 or idx + 1 == n_points:
            progress_pct = 100 * (idx + 1) / n_points
            elapsed = time.time() - fill_start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_points - idx - 1) / rate if rate > 0 else 0
            print(f"      进度: {idx+1:,}/{n_points:,} ({progress_pct:.1f}%) "
                  f"耗时: {elapsed:.1f}s 剩余: {remaining:.1f}s")
            last_progress = idx + 1
    
    return g_final, fill_count

# ============================================================
# 主程序
# ============================================================

# 记录总开始时间
total_start_time = time.time()

print("=" * 60)
print("拼接带挖掉 + 插值填充法")
print("=" * 60)
print(f"  拼接带宽度: {GAP_PADDING} 像素 (~{GAP_PADDING * 300 / 1000:.1f} km)")
print(f"  插值方法: {INTERP_METHOD}")
if INTERP_METHOD == 'clough_tocher':
    print(f"    连续性: C¹ (一阶导数连续)")
elif INTERP_METHOD == 'cubic':
    print(f"    连续性: C¹ (一阶导数连续)")
elif INTERP_METHOD == 'linear':
    print(f"    连续性: C⁰ (一阶导数不连续)")
elif INTERP_METHOD == 'rbf':
    print(f"    连续性: C∞ (无限光滑)")
print("=" * 60)

# 1. 读取数据
print("\n1. 读取数据...")
step_start = time.time()
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)

print(f"   Land: {land_data.shape}, 有效点: {np.sum(~np.isnan(land_data))}")
print(f"   Sea:  {sea_data.shape}, 有效点: {np.sum(~np.isnan(sea_data))}")
print(f"   耗时: {time.time() - step_start:.2f}s")

# 2. 对齐 Sea
print("\n2. 对齐 Sea 到 Land 网格...")
step_start = time.time()
sea_aligned = align_sea_to_land(sea_data, sea_header, land_header)
print(f"   对齐后 Sea 有效点: {np.sum(~np.isnan(sea_aligned))}")
print(f"   耗时: {time.time() - step_start:.2f}s")

# 3. 创建初始复合场
print("\n3. 创建初始复合场 (Sea 优先)...")
step_start = time.time()
g0 = sea_aligned.copy()
land_only_mask = np.isnan(sea_aligned) & ~np.isnan(land_data)
g0[land_only_mask] = land_data[land_only_mask]
print(f"   初始复合场有效点: {np.sum(~np.isnan(g0))}")
print(f"   耗时: {time.time() - step_start:.2f}s")

# 4. 识别拼接带
print("\n4. 识别拼接带...")
step_start = time.time()
land_valid = ~np.isnan(land_data)
sea_valid = ~np.isnan(sea_aligned)

boundary_sea = binary_dilation(sea_valid, iterations=1) & land_valid
boundary_land = binary_dilation(land_valid, iterations=1) & sea_valid
boundary = boundary_sea | boundary_land

gap_mask = binary_dilation(boundary, iterations=GAP_PADDING)
gap_mask = gap_mask & (land_valid | sea_valid)

print(f"   边界点数: {np.sum(boundary)}")
print(f"   拼接带点数: {np.sum(gap_mask)}")
print(f"   拼接带占有效区域: {100 * np.sum(gap_mask) / np.sum(land_valid | sea_valid):.2f}%")
print(f"   耗时: {time.time() - step_start:.2f}s")

# 5. 挖掉拼接带
print("\n5. 挖掉拼接带...")
step_start = time.time()
g_masked = g0.copy()
g_masked[gap_mask] = np.nan
print(f"   挖掉后有效点: {np.sum(~np.isnan(g_masked))}")
print(f"   耗时: {time.time() - step_start:.2f}s")

# 6. 获取有效数据点
valid_rows, valid_cols = np.where(~np.isnan(g_masked))
valid_values = g_masked[valid_rows, valid_cols]
points = np.column_stack([valid_cols, valid_rows])

gap_rows, gap_cols = np.where(gap_mask)
interp_points = np.column_stack([gap_cols, gap_rows])

print(f"   插值参考点: {len(points):,}")
print(f"   需要插值点: {len(interp_points):,}")

# 7. 插值
print(f"\n6. 插值填充 (方法: {INTERP_METHOD})...")
step_start = time.time()

if INTERP_METHOD == 'clough_tocher':
    print("   正在构建三角剖分...")
    interp = CloughTocher2DInterpolator(points, valid_values)
    print("   正在执行插值...")
    interpolated = interp(interp_points)
    
elif INTERP_METHOD == 'cubic':
    print("   正在执行 cubic 插值...")
    interpolated = griddata(points, valid_values, interp_points, 
                            method='cubic', fill_value=np.nan)
    
elif INTERP_METHOD == 'linear':
    print("   正在执行 linear 插值...")
    interpolated = griddata(points, valid_values, interp_points, 
                            method='linear', fill_value=np.nan)
    
elif INTERP_METHOD == 'rbf':
    from scipy.interpolate import RBFInterpolator
    print(f"   正在构建 RBF 插值器 (kernel={RBF_KERNEL})...")
    rbf = RBFInterpolator(points, valid_values, 
                          kernel=RBF_KERNEL, epsilon=RBF_EPSILON)
    print("   正在执行插值...")
    interpolated = rbf(interp_points)
    
elif INTERP_METHOD == 'thin_plate_spline':
    from scipy.interpolate import RBFInterpolator
    print(f"   正在构建 Thin Plate Spline 插值器 (C² 连续)...")
    rbf = RBFInterpolator(points, valid_values, kernel='thin_plate_spline')
    print("   正在执行插值...")
    interpolated = rbf(interp_points)

else:
    raise ValueError(f"未知插值方法: {INTERP_METHOD}")

interp_time = time.time() - step_start
print(f"   插值计算耗时: {interp_time:.2f}s")

# 8. 带进度显示的填充
print("\n7. 填充拼接带...")
fill_start_time = time.time()
g_final = g0.copy()
g_final, fill_count = fill_with_progress(g_final, gap_rows, gap_cols, 
                                          interpolated, PROGRESS_INTERVAL)
fill_time = time.time() - fill_start_time
print(f"   填充耗时: {fill_time:.2f}s")
print(f"   成功填充: {fill_count:,} / {len(interp_points):,}")

# 9. 验证
non_gap = ~gap_mask & ~np.isnan(g0)
if np.allclose(g_final[non_gap], g0[non_gap], equal_nan=True):
    print("   ✓ 验证通过: 非拼接带完全保留原始值")
else:
    print("   ⚠ 警告: 非拼接带被意外修改")

# 10. 统计结果
print("\n8. 结果统计:")
print(f"   最终有效点: {np.sum(~np.isnan(g_final))}")
print(f"   最小值: {np.nanmin(g_final):.2f} mGal")
print(f"   最大值: {np.nanmax(g_final):.2f} mGal")
print(f"   平均值: {np.nanmean(g_final):.2f} mGal")
print(f"   标准差: {np.nanstd(g_final):.6f} mGal")

# 分区域统计
land_only = ~np.isnan(land_data) & np.isnan(sea_aligned)
sea_only = ~np.isnan(sea_aligned) & np.isnan(land_data)
overlap = ~np.isnan(land_data) & ~np.isnan(sea_aligned)

print(f"\n   纯陆区均值: {np.nanmean(g_final[land_only]):.2f} mGal (原始: {np.nanmean(land_data[land_only]):.2f})")
print(f"   纯海区均值: {np.nanmean(g_final[sea_only]):.2f} mGal (原始: {np.nanmean(sea_aligned[sea_only]):.2f})")
print(f"   重叠区均值: {np.nanmean(g_final[overlap]):.2f} mGal")

# 11. 保存
print("\n9. 保存结果...")
step_start = time.time()
write_asc_grid(output_file, g_final, land_header)
print(f"   ✓ 已保存: {output_file}")
print(f"   耗时: {time.time() - step_start:.2f}s")

# 12. 总耗时统计
total_time = time.time() - total_start_time
print("\n" + "=" * 60)
print(f"✅ 拼接带插值填充完成！")
print(f"   总耗时: {total_time:.2f}s ({total_time/60:.1f} 分钟)")
print("=" * 60)
