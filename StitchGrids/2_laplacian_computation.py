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
    Discrete Laplacian（5-point second order finite difference)- Equation (13)
    If  any node from stencil is NaN，output NaN - Equation (14)
    """
    # 5-point finite difference kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    # Mark the NaN locations
    nan_mask = np.isnan(data)
    
    # For convolution Replace NaN to 0
    data_clean = np.nan_to_num(data, nan=0.0)
    
    # Convolution Computation
    laplacian = convolve2d(data_clean, kernel, mode='same', boundary='symm')
    laplacian = laplacian / (cellsize * cellsize)
    
    # Expand NaN mask： If  any node from stencil is NaN
    extended_nan = nan_mask.copy()
    if nan_mask.shape[0] > 1:
        extended_nan[1:, :] |= nan_mask[:-1, :]      # Up
        extended_nan[:-1, :] |= nan_mask[1:, :]      # Down
    if nan_mask.shape[1] > 1:
        extended_nan[:, 1:] |= nan_mask[:, :-1]      # Left
        extended_nan[:, :-1] |= nan_mask[:, 1:]      # Right
    
    laplacian[extended_nan] = np.nan
    
    return laplacian

def align_grid_to_land(data, data_header, land_header):
    """
    Align random grid data to land grid
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

# ===== Main =====
print("=" * 60)
print("Step 2: Independent Laplacian Computation (Document2 Step 2)")
print("=" * 60)

# 1. Read .asc file header
print("\n1. Read Header from .asc...")
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

# 2. Read Original Data
print("\n2. Read Original Gravity Anomaly Data...")
land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)
print(f"   Land Data Shape: {land_data.shape}")
print(f"   Sea Data Shape:  {sea_data.shape}")

# 3. Laplacian  Computation（On original grids, independent）
print("\n3. Discrete Laplacian Computation- Equation (13)...")
cellsize = land_header['cellsize']

print("   Land Laplacian Computation...")
land_laplacian = compute_laplacian_fast(land_data, cellsize)

print("   Sea Laplacian Computation...")
sea_laplacian = compute_laplacian_fast(sea_data, cellsize)

print(f"   Land Laplacian valid Points: {np.sum(~np.isnan(land_laplacian)):,}")
print(f"   Sea Laplacian valid Points:  {np.sum(~np.isnan(sea_laplacian)):,}")

# 4. Align Sea Laplacian to Land Grids（Prepared for Step 3）
print("\n4. Align Sea Laplacian to Land Grids...")
sea_laplacian_aligned = align_grid_to_land(sea_laplacian, sea_header, land_header)
print(f"   Aligned Sea Laplacian Valid Points: {np.sum(~np.isnan(sea_laplacian_aligned)):,}")

# 5. Save
print("\n5. Save Laplacian Computation Results...")

land_laplacian_file = os.path.join(output_dir, 'Land_Laplacian.asc')
write_asc_grid(land_laplacian_file, land_laplacian, land_header)
print(f"   ✓ Land Laplacian: {land_laplacian_file}")

sea_laplacian_original_file = os.path.join(output_dir, 'Sea_Laplacian_Original.asc')
write_asc_grid(sea_laplacian_original_file, sea_laplacian, sea_header)
print(f"   ✓ Sea Laplacian (Original Data): {sea_laplacian_original_file}")

sea_laplacian_aligned_file = os.path.join(output_dir, 'Sea_Laplacian_Aligned.asc')
write_asc_grid(sea_laplacian_aligned_file, sea_laplacian_aligned, land_header)
print(f"   ✓ Sea Laplacian (Aligned to Land): {sea_laplacian_aligned_file}")

# 6. Statics
print("\n6. Laplacian Statics:")
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

print("\n✅ Step 2 Fined！")
print("\n📌 Next: Step 3 - Merging Laplacian (Using Priority Mask) and Interpolating Gap")
