# example_laplacian_merge_demo.py
"""
Toy dataset demo: Laplacian domain gravity grid merging
- Land grid: 10x10
- Sea grid: 8x8
- Overlap region: bottom-right corner
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2, fftfreq

# ============================================================
# 1. Create toy dataset
# ============================================================
print("=" * 70)
print("1. Create toy dataset")
print("=" * 70)

# Land grid: 10x10, simulate onshore Bouguer anomaly (smaller, negative)
np.random.seed(42)
land_full = np.random.randn(10, 10) * 10 - 20  # mean ~ -20
# Simulate offshore area as NaN (rows 5-9, cols 5-9)
land_full[5:, 5:] = np.nan
# Add some onshore features
land_full[2:5, 2:5] = -30
land_full[0:3, 7:10] = -15

# Sea grid: 8x8, simulate offshore Bouguer anomaly (larger, positive)
sea_full = np.random.randn(8, 8) * 15 + 50  # mean ~ +50
# Simulate coastal area with NaNs
sea_full[0:2, 0:2] = np.nan
sea_full[6:8, 6:8] = np.nan
# Add some offshore features
sea_full[3:6, 3:6] = 80

print("Land grid (10x10, NaN = offshore):")
print(np.round(land_full, 2))
print(f"\nLand valid points: {np.sum(~np.isnan(land_full))}/100")
print(f"Land range: [{np.nanmin(land_full):.2f}, {np.nanmax(land_full):.2f}]")

print("\nSea grid (8x8, NaN = onshore/islands):")
print(np.round(sea_full, 2))
print(f"\nSea valid points: {np.sum(~np.isnan(sea_full))}/64")
print(f"Sea range: [{np.nanmin(sea_full):.2f}, {np.nanmax(sea_full):.2f}]")

# ============================================================
# 2. Step 1: Align to same grid + priority mask
# ============================================================
print("\n" + "=" * 70)
print("2. Step 1: Alignment + Priority Mask (offshore priority)")
print("=" * 70)

# Position of Sea within Land grid (Sea covers bottom-right 8x8 of Land)
# Sea's (0,0) corresponds to Land's (2,2)
sea_offset_row = 2
sea_offset_col = 2

# Create aligned Sea grid (same size as Land)
sea_aligned = np.full_like(land_full, np.nan)
for i in range(8):
    for j in range(8):
        sea_aligned[sea_offset_row + i, sea_offset_col + j] = sea_full[i, j]

print("Sea aligned to Land grid (bottom-right 8x8):")
print(np.round(sea_aligned, 2))

# Priority mask: 1=offshore priority, 0=onshore fill
mask = np.full_like(land_full, np.nan)
mask[~np.isnan(sea_aligned)] = 1      # Offshore region
mask[(~np.isnan(land_full)) & np.isnan(sea_aligned)] = 0  # Onshore-only region

print("\nPriority mask (1=Offshore, 0=Onshore, NaN=No data):")
print(mask)

# Initial composite field g0
g0 = np.full_like(land_full, np.nan)
g0[mask == 1] = sea_aligned[mask == 1]
g0[mask == 0] = land_full[mask == 0]

print("\nInitial composite field g0 (direct merge, with step):")
print(np.round(g0, 2))

# ============================================================
# 3. Step 2: Independent Laplacian computation
# ============================================================
print("\n" + "=" * 70)
print("3. Step 2: Independent Laplacian computation (5-point stencil)")
print("=" * 70)

def compute_laplacian_5pt(grid, dx=1.0, dy=1.0):
    """5-point Laplacian, returns NaN at edges"""
    ny, nx = grid.shape
    laplacian = np.full_like(grid, np.nan)
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Check if all 5 points are valid
            neighbors = [grid[i,j], grid[i+1,j], grid[i-1,j], grid[i,j+1], grid[i,j-1]]
            if any(np.isnan(n) for n in neighbors):
                laplacian[i,j] = np.nan
            else:
                laplacian[i,j] = (grid[i+1,j] + grid[i-1,j] - 2*grid[i,j]) / (dx*dx) + \
                                 (grid[i,j+1] + grid[i,j-1] - 2*grid[i,j]) / (dy*dy)
    return laplacian

# Compute Laplacian independently on original grids
land_laplacian = compute_laplacian_5pt(land_full)
sea_laplacian = compute_laplacian_5pt(sea_full)

print("Land Laplacian (on original 10x10 grid):")
print(np.round(land_laplacian, 2))

print("\nSea Laplacian (on original 8x8 grid):")
print(np.round(sea_laplacian, 2))

# Align Sea Laplacian to Land grid
sea_laplacian_aligned = np.full_like(land_full, np.nan)
for i in range(8):
    for j in range(8):
        sea_laplacian_aligned[sea_offset_row + i, sea_offset_col + j] = sea_laplacian[i, j]

# ============================================================
# 4. Step 3: Merge Laplacian
# ============================================================
print("\n" + "=" * 70)
print("4. Step 3: Merge Laplacian (Equation 15)")
print("=" * 70)

L0 = np.full_like(land_full, np.nan)
L0[mask == 1] = sea_laplacian_aligned[mask == 1]  # Offshore region
L0[mask == 0] = land_laplacian[mask == 0]         # Onshore region

print("Merged Laplacian L0 (with gaps Ω_gap):")
print(np.round(L0, 2))

# Identify gaps: mask valid but L0 is NaN
gap_mask = (~np.isnan(mask)) & np.isnan(L0)
print(f"\nGap Ω_gap points: {np.sum(gap_mask)}")

# ============================================================
# 5. Step 4: Bicubic spline interpolation for gaps
# ============================================================
print("\n" + "=" * 70)
print("5. Step 4: Bicubic spline interpolation for Ω_gap")
print("=" * 70)

def interpolate_gaps(L0, gap_mask):
    """Fill gaps using griddata"""
    ny, nx = L0.shape
    valid_mask = ~np.isnan(L0)
    
    if np.sum(valid_mask) < 4:
        return L0
    
    # Valid points coordinates and values
    valid_rows, valid_cols = np.where(valid_mask)
    valid_vals = L0[valid_mask]
    
    # Positions to interpolate
    gap_rows, gap_cols = np.where(gap_mask)
    
    if len(gap_rows) == 0:
        return L0
    
    # Use griddata for interpolation
    points = np.column_stack([valid_cols, valid_rows])
    interp_points = np.column_stack([gap_cols, gap_rows])
    
    interpolated = griddata(points, valid_vals, interp_points, method='cubic')
    
    L_filled = L0.copy()
    for idx, (r, c) in enumerate(zip(gap_rows, gap_cols)):
        if not np.isnan(interpolated[idx]):
            L_filled[r, c] = interpolated[idx]
    
    return L_filled

L_filled = interpolate_gaps(L0, gap_mask)

print("Interpolated Laplacian L_filled:")
print(np.round(L_filled, 2))
print(f"\nRemaining NaN count: {np.sum(np.isnan(L_filled))}")

# Save valid_mask for later use (important!)
valid_mask = ~np.isnan(L_filled)
print(f"Original valid region points: {np.sum(valid_mask)}")

# ============================================================
# 6. Step 5: Frequency domain inversion
# ============================================================
print("\n" + "=" * 70)
print("6. Step 5: Frequency domain inversion ∇²g = f")
print("=" * 70)

def solve_poisson_fft(f, dx=1.0, dy=1.0, alpha=1e-6):
    """Solve Poisson equation with Tikhonov regularization"""
    ny, nx = f.shape
    
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    
    H = (2/(dx*dx)) * (np.cos(KX*dx) - 1) + (2/(dy*dy)) * (np.cos(KY*dy) - 1)
    
    # Tikhonov regularization: H / (H^2 + alpha^2) replaces 1/H
    # In this case, when H is very small，denominator is not too huge
    H_safe = H**2 + alpha**2
    filter_reg = H / H_safe
    
    F = fft2(np.nan_to_num(f, nan=0.0))
    G = F * filter_reg  # Note: This is multiplication, not division.
    g = np.real(ifft2(G))
    
    return g, H

# Replace NaN with 0 for FFT
f_source = np.nan_to_num(L_filled, nan=0.0)
g_solution, H = solve_poisson_fft(f_source)

# Restore original valid region (keep NaN positions)
final = g_solution.copy()
final[~valid_mask] = np.nan

print("Bouguer anomaly from frequency domain inversion:")
print(np.round(final, 2))

# ============================================================
# 7. Validation and correction (Onshore-only region)
# ============================================================
print("\n" + "=" * 70)
print("7. Validation and correction (Onshore-only region)")
print("=" * 70)

# Onshore-only region: mask=0 and original Land is valid
land_only_mask = (mask == 0) & ~np.isnan(land_full)

diff = land_full[land_only_mask] - final[land_only_mask]
offset = np.nanmean(diff)

print(f"Onshore-only region points: {np.sum(land_only_mask)}")
print(f"Final vs Land difference mean: {offset:.6f} mGal")

if abs(offset) > 0.01:
    print(f"DC offset detected: {offset:.4f}, applying correction...")
    final_corrected = final + offset
else:
    print("No significant offset detected")
    final_corrected = final

print("\nCorrected final Bouguer anomaly:")
print(np.round(final_corrected, 2))

# ============================================================
# 8. Visualization
# ============================================================
print("\n" + "=" * 70)
print("8. Visualization")
print("=" * 70)

# Handle NaN for plotting
land_plot = np.where(np.isnan(land_full), np.nan, land_full)
sea_plot = np.where(np.isnan(sea_full), np.nan, sea_full)
g0_plot = np.where(np.isnan(g0), np.nan, g0)
final_plot = np.where(np.isnan(final), np.nan, final)
final_corrected_plot = np.where(np.isnan(final_corrected), np.nan, final_corrected)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))

# Set colormap and consistent ranges
vmin_g = -50
vmax_g = 100
vmin_lap = -200
vmax_lap = 200

# Row 1: Land, Sea, Aligned Sea, mask
im1 = axes[0,0].imshow(land_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[0,0].set_title('Land Original Data (10x10)', fontsize=12)
axes[0,0].set_xlabel('Column'); axes[0,0].set_ylabel('Row')
plt.colorbar(im1, ax=axes[0,0], label='mGal')

im2 = axes[0,1].imshow(sea_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[0,1].set_title('Sea Original Data (8x8)', fontsize=12)
axes[0,1].set_xlabel('Column'); axes[0,1].set_ylabel('Row')
plt.colorbar(im2, ax=axes[0,1], label='mGal')

im3 = axes[0,2].imshow(sea_aligned, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[0,2].set_title('Sea Aligned to Land Grid', fontsize=12)
axes[0,2].set_xlabel('Column'); axes[0,2].set_ylabel('Row')
plt.colorbar(im3, ax=axes[0,2], label='mGal')

im4 = axes[0,3].imshow(mask, cmap='viridis', vmin=0, vmax=1)
axes[0,3].set_title('Priority Mask (1=Offshore, 0=Onshore)', fontsize=12)
axes[0,3].set_xlabel('Column'); axes[0,3].set_ylabel('Row')
plt.colorbar(im4, ax=axes[0,3], label='mask')

# Row 2: Initial composite, Laplacian merge, inversion, final corrected
im5 = axes[1,0].imshow(g0_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[1,0].set_title('Initial Composite g0 (Direct Merge)\nWith Step', fontsize=12)
axes[1,0].set_xlabel('Column'); axes[1,0].set_ylabel('Row')
plt.colorbar(im5, ax=axes[1,0], label='mGal')

# Handle Laplacian plot (clip range)
lap_plot = np.where(np.isnan(L_filled), np.nan, L_filled)
im6 = axes[1,1].imshow(lap_plot, cmap='coolwarm', vmin=vmin_lap, vmax=vmax_lap)
axes[1,1].set_title('Merged + Interpolated Laplacian f', fontsize=12)
axes[1,1].set_xlabel('Column'); axes[1,1].set_ylabel('Row')
plt.colorbar(im6, ax=axes[1,1], label='Laplacian')

im7 = axes[1,2].imshow(final_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[1,2].set_title('Frequency Domain Result (with DC offset)', fontsize=12)
axes[1,2].set_xlabel('Column'); axes[1,2].set_ylabel('Row')
plt.colorbar(im7, ax=axes[1,2], label='mGal')

im8 = axes[1,3].imshow(final_corrected_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[1,3].set_title('Final Result (Corrected)\nSeamless Merge', fontsize=12)
axes[1,3].set_xlabel('Column'); axes[1,3].set_ylabel('Row')
plt.colorbar(im8, ax=axes[1,3], label='mGal')

plt.tight_layout()
plt.savefig('laplacian_merge_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("9. Result Comparison")
print("=" * 70)
print(f"Direct merge g0:      min={np.nanmin(g0):.2f}, max={np.nanmax(g0):.2f}, mean={np.nanmean(g0):.2f}")
print(f"Before correction:     min={np.nanmin(final):.2f}, max={np.nanmax(final):.2f}, mean={np.nanmean(final):.2f}")
print(f"After correction:      min={np.nanmin(final_corrected):.2f}, max={np.nanmax(final_corrected):.2f}, mean={np.nanmean(final_corrected):.2f}")
print(f"Onshore-only region diff after correction: {np.nanmean(land_full[land_only_mask] - final_corrected[land_only_mask]):.6f} mGal")

# Print key mathematical formulas
print("\n" + "=" * 70)
print("10. Key Mathematical Formulas")
print("=" * 70)
print("Discrete Laplacian (5-point stencil):")
print("  ∇²g_ij = (g_i+1,j + g_i-1,j - 2g_ij)/Δx² + (g_i,j+1 + g_i,j-1 - 2g_ij)/Δy²")
print("\nFrequency domain Poisson solution:")
print("  g = F⁻¹{ F{f} / H(k) }")
print("  H(k) = 2/Δx²·(cos(kx·Δx)-1) + 2/Δy²·(cos(ky·Δy)-1)")
print("\nPriority mask merge:")
print("  L0 = L_sea if mask=1, L_land if mask=0, NaN elsewhere")
print("\nBicubic spline interpolation:")
print("  L_int(x,y) = ΣᵢΣⱼ a_ij xⁱ yʲ,  ∀(x,y)∈Ω_gap")
