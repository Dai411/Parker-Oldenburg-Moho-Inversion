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

# Splicing strip width extension (pixels)
#   Reocommended range: 3~20 pixels
#   Your grid spacing is 300m (intervals of the .asc):
#       3  pixels = 0.9 km (Narrow band, fast)
#       5  pixels = 1.5 km (Medium)
#       10 pixels = 3.0 km (recommended starting parameter)
#       20 pixels = 6.0 km (Broad band，smooth but time-consuming)
GAP_PADDING = 10  # Pixels

# Available Interpolation Methods: 'clough_tocher', 'cubic', 'linear', 'rbf')
#   clough_tocher : C¹ continous， triangulation, recommended (balance in processing speed and smoothness)
#   cubic         : C¹ continous， implemented with griddata, simple and robust
#   linear        : C⁰ continous， fastest but not smoothest
#   rbf           : C∞ conitnous， smoothest but very slow computation (Not recommended for big data)
INTERP_METHOD = 'clough_tocher'  # Recomended

# Progress display interval (outputs once every N points processed)
PROGRESS_INTERVAL = 200000  # Outputs once after processing every 200,000 points

# RBF parameters (Using when When INTERP_METHOD='rbf')
RBF_KERNEL = 'cubic'     # Kernel Function: 'cubic', 'thin_plate_spline', 'gaussian'
RBF_EPSILON = 2.0        # Smoothing parameter

# ============================================================
# File Path
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
output_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
output_file = os.path.join(output_dir, 'BouguerInterpGap.asc')

# ============================================================
# Data Reading and Writing Function
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
    """Align the Sea to Land grids"""
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
# Filling Function with progress
# ============================================================

def fill_with_progress(g_final, gap_rows, gap_cols, interpolated, interval=200000):
    """Fill (interpolate) with progress"""
    n_points = len(gap_rows)
    print(f"   Start filling {n_points:,} points (outputting progress every {interval:,} points)...")
    
    fill_count = 0
    last_progress = 0
    
    for idx, (r, c) in enumerate(zip(gap_rows, gap_cols)):
        if not np.isnan(interpolated[idx]):
            g_final[r, c] = interpolated[idx]
            fill_count += 1
        
        # Displaying interpolation progress
        if (idx + 1) % interval == 0 or idx + 1 == n_points:
            progress_pct = 100 * (idx + 1) / n_points
            elapsed = time.time() - fill_start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_points - idx - 1) / rate if rate > 0 else 0
            print(f"      Prohress: {idx+1:,}/{n_points:,} ({progress_pct:.1f}%) "
                  f"Elapsed: {elapsed:.1f}s Remaining: {remaining:.1f}s")
            last_progress = idx + 1
    
    return g_final, fill_count

# ============================================================
# Main Programme
# ============================================================

# Record starting time
total_start_time = time.time()

print("=" * 60)
print("Spliced patch removal + interpolation filling")
print("=" * 60)
print(f"  Width of Spliched Pathc: {GAP_PADDING} Pixels (~{GAP_PADDING * 300 / 1000:.1f} km)")
print(f"  Interpolation Method: {INTERP_METHOD}")
if INTERP_METHOD == 'clough_tocher':
    print(f"    Continuity: C¹ (Continuous first derivative)")
elif INTERP_METHOD == 'cubic':
    print(f"    Continuity: C¹ (Continuous first derivative)")
elif INTERP_METHOD == 'linear':
    print(f"    Continuity: C⁰ (Disontinuous first derivative)")
elif INTERP_METHOD == 'rbf':
    print(f"    Continuity: C∞ (Infinitely smooth)")
print("=" * 60)

# 1. 
print("\n1. Reading data...")
step_start = time.time()
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)

print(f"   Land: {land_data.shape}, Valid points: {np.sum(~np.isnan(land_data))}")
print(f"   Sea:  {sea_data.shape}, Valid points: {np.sum(~np.isnan(sea_data))}")
print(f"   Elapsed: {time.time() - step_start:.2f}s")

# 2. Align two .asc files
print("\n2. Align Sea to Land grids...")
step_start = time.time()
sea_aligned = align_sea_to_land(sea_data, sea_header, land_header)
print(f"   Valid points after alignment: {np.sum(~np.isnan(sea_aligned))}")
print(f"   Elapsed: {time.time() - step_start:.2f}s")

# 3. Build initial composite filed
print("\n3. Build composite field (Sea prefered)...")
step_start = time.time()
g0 = sea_aligned.copy()
land_only_mask = np.isnan(sea_aligned) & ~np.isnan(land_data)
g0[land_only_mask] = land_data[land_only_mask]
print(f"   Valid points in the composite field: {np.sum(~np.isnan(g0))}")
print(f"   Elapsed: {time.time() - step_start:.2f}s")

# 4. Confirm the spliced patch
print("\n4. Identifying the spliced patch...")
step_start = time.time()
land_valid = ~np.isnan(land_data)
sea_valid = ~np.isnan(sea_aligned)

boundary_sea = binary_dilation(sea_valid, iterations=1) & land_valid
boundary_land = binary_dilation(land_valid, iterations=1) & sea_valid
boundary = boundary_sea | boundary_land

gap_mask = binary_dilation(boundary, iterations=GAP_PADDING)
gap_mask = gap_mask & (land_valid | sea_valid)

print(f"   Numbers of border: {np.sum(boundary)}")
print(f"   Numbers of spliced patch: {np.sum(gap_mask)}")
print(f"   The ratio of spliced patch: {100 * np.sum(gap_mask) / np.sum(land_valid | sea_valid):.2f}%")
print(f"   Elapsed: {time.time() - step_start:.2f}s")

# 5. Remove the patch
print("\n5. Dig (remove) the overlapped patch...")
step_start = time.time()
g_masked = g0.copy()
g_masked[gap_mask] = np.nan
print(f"   Valid points after removal: {np.sum(~np.isnan(g_masked))}")
print(f"   Elapsed: {time.time() - step_start:.2f}s")

# 6. Obtain the valid points
valid_rows, valid_cols = np.where(~np.isnan(g_masked))
valid_values = g_masked[valid_rows, valid_cols]
points = np.column_stack([valid_cols, valid_rows])

gap_rows, gap_cols = np.where(gap_mask)
interp_points = np.column_stack([gap_cols, gap_rows])

print(f"   Interpolation reference point: {len(points):,}")
print(f"   Points need to be interpolated: {len(interp_points):,}")

# 7. Interpolation
print(f"\n6. Fill (interpolation) (Method: {INTERP_METHOD})...")
step_start = time.time()

if INTERP_METHOD == 'clough_tocher':
    print("   Building the triangulation...")
    interp = CloughTocher2DInterpolator(points, valid_values)
    print("   Interpolating...")
    interpolated = interp(interp_points)
    
elif INTERP_METHOD == 'cubic':
    print("   Processing the cubic interpolation...")
    interpolated = griddata(points, valid_values, interp_points, 
                            method='cubic', fill_value=np.nan)
    
elif INTERP_METHOD == 'linear':
    print("   Processing the linear interpolatin...")
    interpolated = griddata(points, valid_values, interp_points, 
                            method='linear', fill_value=np.nan)
    
elif INTERP_METHOD == 'rbf':
    from scipy.interpolate import RBFInterpolator
    print(f"   Buidling the RBF interpolator (kernel={RBF_KERNEL})...")
    rbf = RBFInterpolator(points, valid_values, 
                          kernel=RBF_KERNEL, epsilon=RBF_EPSILON)
    print("   Interpolating...")
    interpolated = rbf(interp_points)
    
elif INTERP_METHOD == 'thin_plate_spline':
    from scipy.interpolate import RBFInterpolator
    print(f"   Building the 'Thin Plate Spline' interpolator (C² continous)...")
    rbf = RBFInterpolator(points, valid_values, kernel='thin_plate_spline')
    print("   Interpolating...")
    interpolated = rbf(interp_points)

else:
    raise ValueError(f"Unknown interpolating method: {INTERP_METHOD}")

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
