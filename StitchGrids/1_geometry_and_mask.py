import numpy as np
import os

# ===== Set File Path =====
data_dir = 'C:/.../BouguerAnomaly'
land_file = os.path.join(data_dir, 'BouguerLandGrid.asc')
sea_file = os.path.join(data_dir, 'BouguerSeaGrid.asc')
output_dir = 'C:/.../BouguerAnomaly/StitchGrids'
# =========================

def read_asc_header(filename):
    """Read ESRI ASCII Rater Header"""
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
    """Read ESRI ASCII Raster Data"""
    if header is None:
        header = read_asc_header(filename)
    data = np.loadtxt(filename, skiprows=6)
    nodata = header['nodata_value']
    data[data == nodata] = np.nan
    return data

def write_asc_grid(filename, data, header):
    """Write data into ESRI ASCII raster"""
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
    Align Coordinates and Size from Sea Raster to Land Raster
    Return：aligned_sea (Using Land Size)
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

# ===== Main =====
print("=" * 60)
print("Step 1: Geometry Setup and Priority Masking (Document2 Step 1)")
print("=" * 60)

# 1. Read Header Info.
print("\n1. Read Header from .asc Files...")
land_header = read_asc_header(land_file)
sea_header = read_asc_header(sea_file)

print(f"   Land Grid: {land_header['ncols']} x {land_header['nrows']}")
print(f"   Sea Grid:  {sea_header['ncols']} x {sea_header['nrows']}")
print(f"   cellsize: {land_header['cellsize']} m (Identical)")

# 2. Read Original Data
print("\n2. Read Original Gravity Data...")
land_data = read_asc_grid(land_file, land_header)
sea_data = read_asc_grid(sea_file, sea_header)
print(f"   Land Valid Points: {np.sum(~np.isnan(land_data))}")
print(f"   Sea Valid Points:  {np.sum(~np.isnan(sea_data))}")

# 3. Aligh Sea Grid to Land Grid
print("\n3. Align Sea to Land Grids (Define Global Domain Ω)...")
sea_aligned = align_sea_to_land_grid(sea_data, sea_header, land_header)

# 4. Build Preferetial Mask m(x,y) - Equation (11)
print("\n4. Build Priority Mask m(x,y)")
print("   Rule: Sea Prefered (mask=1), Land Filled (mask=0), Null (mask=NaN)")

mask = np.full_like(land_data, np.nan, dtype=np.float32)

# Condition 1: (x,y) ∈ Sea Grid → mask = 1
mask[~np.isnan(sea_aligned)] = 1.0

# Condition 2: (x,y) ∈ Land Grid 且 Sea 为 NaN → mask = 0
mask[(~np.isnan(land_data)) & np.isnan(sea_aligned)] = 0.0

# Condition 3: Other → mask = NaN (Already initialized as NaN)

print(f"   mask=1 (Sea Prefered): {np.sum(mask == 1):,}")
print(f"   mask=0 (Land Filled Region): {np.sum(mask == 0):,}")
print(f"   mask=NaN (NoData): {np.sum(np.isnan(mask)):,}")

# 5. Build Initial Composite filed g0(x,y) - Equation (12)
print("\n5. Build Initial Composite Gravity Field g0(x,y)")
g0 = np.full_like(land_data, np.nan)

# g0 = g_sea if mask = 1
g0[mask == 1] = sea_aligned[mask == 1]

# g0 = g_land if mask = 0
g0[mask == 0] = land_data[mask == 0]

# g0 = NaN elsewhere (already)

print(f"   g0 Valid Points (Grids): {np.sum(~np.isnan(g0)):,}")

# 6. Save 
print("\n6. Save Middle Files...")

# Save Priority Mask
mask_file = os.path.join(output_dir, 'PriorityMask.asc')
write_asc_grid(mask_file, mask, land_header)
print(f"   ✓ Priority Mask: {mask_file}")

# Save Initial Composite Field (Used only for visualization, no for following merging）
g0_file = os.path.join(output_dir, 'InitialComposite_g0.asc')
write_asc_grid(g0_file, g0, land_header)
print(f"   ✓ Initial Composite Field g0: {g0_file}")

# Save Aligned Sea Data (For folloing steps)）
sea_aligned_file = os.path.join(output_dir, 'Sea_Aligned_To_LandGrid.asc')
write_asc_grid(sea_aligned_file, sea_aligned, land_header)
print(f"   ✓ Sea Aligned File: {sea_aligned_file}")

# 7. Static（Identify and analyse GAP）
print("\n7. Gap Analysis (Ω_gap)...")
# Gap Region: valid mask but invalid g0? Actually, according to the documentation, 
# the gap is a NaN region generated by the stencil boundary during Laplacian merging. 
# Here, we'll first record the valid boundary.

# Locate the land-sea boundary (the transition area where the mask ranges from 1 to 0)
from scipy.ndimage import binary_dilation, binary_erosion

mask_sea_region = (mask == 1)
mask_land_region = (mask == 0)

# Land-sea boundary: The area where the sea expands and intersects with the land.
boundary = binary_dilation(mask_sea_region, iterations=1) & mask_land_region
print(f"   Number of points on the land-sea boundary (potential suture line): {np.sum(boundary):,}")

print("\n✅ Step 1 Finished！")
print("\n📌 Next: Step 2 - Independent Laplacian Computation")
