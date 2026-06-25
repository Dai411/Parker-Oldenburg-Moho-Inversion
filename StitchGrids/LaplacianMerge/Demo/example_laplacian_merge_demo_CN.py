# example_laplacian_merge_demo_CN.py
"""
玩具数据集演示：Laplacian 域重力网格拼接
- Land 网格: 10x10
- Sea 网格: 8x8
- 重叠区域: 右下角
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2, fftfreq

# ============================================================
# 1. 创建玩具数据集
# ============================================================
print("=" * 70)
print("1. 创建玩具数据集")
print("=" * 70)

# Land 网格: 10x10，模拟陆区布格异常 (值较小，负异常)
np.random.seed(42)
land_full = np.random.randn(10, 10) * 10 - 20  # 均值约 -20
# 模拟海区位置为 NaN（第6-10行，第6-10列）
land_full[5:, 5:] = np.nan
# 再加一些陆区特征
land_full[2:5, 2:5] = -30
land_full[0:3, 7:10] = -15

# Sea 网格: 8x8，模拟海区布格异常 (值较大，正异常)
sea_full = np.random.randn(8, 8) * 15 + 50  # 均值约 +50
# 模拟海岸线附近有 NaN
sea_full[0:2, 0:2] = np.nan
sea_full[6:8, 6:8] = np.nan
# 加一些海区特征
sea_full[3:6, 3:6] = 80

print("Land 网格 (10x10, NaN 为海区):")
print(np.round(land_full, 2))
print(f"\nLand 有效值: {np.sum(~np.isnan(land_full))}/100")
print(f"Land 范围: [{np.nanmin(land_full):.2f}, {np.nanmax(land_full):.2f}]")

print("\nSea 网格 (8x8, NaN 为陆区/岛屿):")
print(np.round(sea_full, 2))
print(f"\nSea 有效值: {np.sum(~np.isnan(sea_full))}/64")
print(f"Sea 范围: [{np.nanmin(sea_full):.2f}, {np.nanmax(sea_full):.2f}]")

# ============================================================
# 2. Step 1: 对齐到同一网格 + 优先级掩码
# ============================================================
print("\n" + "=" * 70)
print("2. Step 1: 对齐 + 优先级掩码 (海区优先)")
print("=" * 70)

# Sea 在 Land 网格中的位置（假设 Sea 覆盖 Land 的右下角 8x8）
# 即 Sea 的 (0,0) 对应 Land 的 (2,2)
sea_offset_row = 2
sea_offset_col = 2

# 创建对齐后的 Sea 网格（与 Land 同尺寸）
sea_aligned = np.full_like(land_full, np.nan)
for i in range(8):
    for j in range(8):
        sea_aligned[sea_offset_row + i, sea_offset_col + j] = sea_full[i, j]

print("Sea 对齐到 Land 网格后的位置 (右下角 8x8):")
print(np.round(sea_aligned, 2))

# 优先级掩码: 1=Sea优先, 0=Land填充
mask = np.full_like(land_full, np.nan)
mask[~np.isnan(sea_aligned)] = 1      # Sea 区域
mask[(~np.isnan(land_full)) & np.isnan(sea_aligned)] = 0  # Land 独有区域

print("\n优先级掩码 mask (1=Sea, 0=Land, NaN=无数据):")
print(mask)

# 初始复合场 g0
g0 = np.full_like(land_full, np.nan)
g0[mask == 1] = sea_aligned[mask == 1]
g0[mask == 0] = land_full[mask == 0]

print("\n初始复合场 g0 (直接拼接，有台阶):")
print(np.round(g0, 2))

# ============================================================
# 3. Step 2: 独立计算 Laplacian
# ============================================================
print("\n" + "=" * 70)
print("3. Step 2: 独立计算 Laplacian (5点差分)")
print("=" * 70)

def compute_laplacian_5pt(grid, dx=1.0, dy=1.0):
    """5点差分 Laplacian，边缘返回 NaN"""
    ny, nx = grid.shape
    laplacian = np.full_like(grid, np.nan)
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # 检查 5 点是否都有效
            neighbors = [grid[i,j], grid[i+1,j], grid[i-1,j], grid[i,j+1], grid[i,j-1]]
            if any(np.isnan(n) for n in neighbors):
                laplacian[i,j] = np.nan
            else:
                laplacian[i,j] = (grid[i+1,j] + grid[i-1,j] - 2*grid[i,j]) / (dx*dx) + \
                                 (grid[i,j+1] + grid[i,j-1] - 2*grid[i,j]) / (dy*dy)
    return laplacian

# 在原始网格上独立计算 Laplacian
land_laplacian = compute_laplacian_5pt(land_full)
sea_laplacian = compute_laplacian_5pt(sea_full)

print("Land Laplacian (在原始 10x10 网格):")
print(np.round(land_laplacian, 2))

print("\nSea Laplacian (在原始 8x8 网格):")
print(np.round(sea_laplacian, 2))

# 对齐 Sea Laplacian 到 Land 网格
sea_laplacian_aligned = np.full_like(land_full, np.nan)
for i in range(8):
    for j in range(8):
        sea_laplacian_aligned[sea_offset_row + i, sea_offset_col + j] = sea_laplacian[i, j]

# ============================================================
# 4. Step 3: 合并 Laplacian
# ============================================================
print("\n" + "=" * 70)
print("4. Step 3: 合并 Laplacian (公式 15)")
print("=" * 70)

L0 = np.full_like(land_full, np.nan)
L0[mask == 1] = sea_laplacian_aligned[mask == 1]  # Sea 区域
L0[mask == 0] = land_laplacian[mask == 0]         # Land 区域

print("合并后的 Laplacian L0 (有间隙 Ω_gap):")
print(np.round(L0, 2))

# 识别间隙: mask 有效但 L0 为 NaN
gap_mask = (~np.isnan(mask)) & np.isnan(L0)
print(f"\n间隙 Ω_gap 点数: {np.sum(gap_mask)}")

# ============================================================
# 5. Step 4: 双三次样条插值填充间隙
# ============================================================
print("\n" + "=" * 70)
print("5. Step 4: 双三次样条插值填充 Ω_gap")
print("=" * 70)

def interpolate_gaps(L0, gap_mask):
    """用 griddata 填充间隙"""
    ny, nx = L0.shape
    valid_mask = ~np.isnan(L0)
    
    if np.sum(valid_mask) < 4:
        return L0
    
    # 有效点坐标和值
    valid_rows, valid_cols = np.where(valid_mask)
    valid_vals = L0[valid_mask]
    
    # 需要插值的位置
    gap_rows, gap_cols = np.where(gap_mask)
    
    if len(gap_rows) == 0:
        return L0
    
    # 使用 griddata 插值
    points = np.column_stack([valid_cols, valid_rows])
    interp_points = np.column_stack([gap_cols, gap_rows])
    
    interpolated = griddata(points, valid_vals, interp_points, method='cubic')
    
    L_filled = L0.copy()
    for idx, (r, c) in enumerate(zip(gap_rows, gap_cols)):
        if not np.isnan(interpolated[idx]):
            L_filled[r, c] = interpolated[idx]
    
    return L_filled

L_filled = interpolate_gaps(L0, gap_mask)

print("插值后的 Laplacian L_filled:")
print(np.round(L_filled, 2))
print(f"\n剩余 NaN 数量: {np.sum(np.isnan(L_filled))}")

# 保存 valid_mask 供后续使用（重要！）
valid_mask = ~np.isnan(L_filled)  # 原始有效区域掩码
print(f"原始有效区域点数: {np.sum(valid_mask)}")

# ============================================================
# 6. Step 5: 频域反解
# ============================================================
print("\n" + "=" * 70)
print("6. Step 5: 频域反解 ∇²g = f")
print("=" * 70)

def solve_poisson_fft(f, dx=1.0, dy=1.0):
    """频域求解 Poisson 方程: ∇²g = f"""
    ny, nx = f.shape
    
    # 波数
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    
    # 离散拉普拉斯传递函数 H(k)
    H = (2/(dx*dx)) * (np.cos(KX*dx) - 1) + (2/(dy*dy)) * (np.cos(KY*dy) - 1)
    
    # 避免除零 (设置 DC 分量的 H 为一个小的负数)
    H[np.abs(H) < 1e-8] = -1e-8
    
    # FFT -> 除法 -> IFFT
    F = fft2(np.nan_to_num(f, nan=0.0))
    G = F / H
    g = np.real(ifft2(G))
    
    return g, H

# 将 Laplacian 源场 f 中的 NaN 替换为 0 用于 FFT
f_source = np.nan_to_num(L_filled, nan=0.0)
g_solution, H = solve_poisson_fft(f_source)

# 恢复原始有效区域（NaN 位置保留）
final = g_solution.copy()
final[~valid_mask] = np.nan

print("频域反解得到的布格异常 g_final:")
print(np.round(final, 2))

# ============================================================
# 7. 验证与校正 (Land 独有区域)
# ============================================================
print("\n" + "=" * 70)
print("7. 验证与校正 (Land 独有区域)")
print("=" * 70)

# Land 独有区域: mask=0 且原始 Land 有效
land_only_mask = (mask == 0) & ~np.isnan(land_full)

diff = land_full[land_only_mask] - final[land_only_mask]
offset = np.nanmean(diff)

print(f"Land 独有区域点数: {np.sum(land_only_mask)}")
print(f"Final 与 Land 差异均值: {offset:.6f} mGal")

if abs(offset) > 0.01:
    print(f"检测到 DC 偏移 {offset:.4f}，进行校正...")
    final_corrected = final + offset
else:
    print("无明显偏移，无需校正")
    final_corrected = final

print("\n校正后的最终布格异常:")
print(np.round(final_corrected, 2))

# ============================================================
# 8. 可视化
# ============================================================
print("\n" + "=" * 70)
print("8. 可视化结果")
print("=" * 70)

# 处理 NaN 用于绘图
land_plot = np.where(np.isnan(land_full), np.nan, land_full)
sea_plot = np.where(np.isnan(sea_full), np.nan, sea_full)
g0_plot = np.where(np.isnan(g0), np.nan, g0)
final_plot = np.where(np.isnan(final), np.nan, final)
final_corrected_plot = np.where(np.isnan(final_corrected), np.nan, final_corrected)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))

# 设置 colormap 和 vmin/vmax 保持一致
vmin_g = -50
vmax_g = 100
vmin_lap = -200
vmax_lap = 200

# 第一行: Land, Sea, 对齐 Sea, mask
im1 = axes[0,0].imshow(land_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[0,0].set_title('Land 原始数据 (10x10)', fontsize=12)
axes[0,0].set_xlabel('列'); axes[0,0].set_ylabel('行')
plt.colorbar(im1, ax=axes[0,0], label='mGal')

im2 = axes[0,1].imshow(sea_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[0,1].set_title('Sea 原始数据 (8x8)', fontsize=12)
axes[0,1].set_xlabel('列'); axes[0,1].set_ylabel('行')
plt.colorbar(im2, ax=axes[0,1], label='mGal')

im3 = axes[0,2].imshow(sea_aligned, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[0,2].set_title('Sea 对齐到 Land 网格', fontsize=12)
axes[0,2].set_xlabel('列'); axes[0,2].set_ylabel('行')
plt.colorbar(im3, ax=axes[0,2], label='mGal')

im4 = axes[0,3].imshow(mask, cmap='viridis', vmin=0, vmax=1)
axes[0,3].set_title('优先级掩码 (1=Sea, 0=Land)', fontsize=12)
axes[0,3].set_xlabel('列'); axes[0,3].set_ylabel('行')
plt.colorbar(im4, ax=axes[0,3], label='mask')

# 第二行: 初始复合场, Laplacian 合并, 频域反解, 最终校正
im5 = axes[1,0].imshow(g0_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[1,0].set_title('初始复合场 g₀ (直接拼接)\n有台阶', fontsize=12)
axes[1,0].set_xlabel('列'); axes[1,0].set_ylabel('行')
plt.colorbar(im5, ax=axes[1,0], label='mGal')

# 处理 Laplacian 绘图（限制范围）
lap_plot = np.where(np.isnan(L_filled), np.nan, L_filled)
im6 = axes[1,1].imshow(lap_plot, cmap='coolwarm', vmin=vmin_lap, vmax=vmax_lap)
axes[1,1].set_title('合并+插值后的 Laplacian f', fontsize=12)
axes[1,1].set_xlabel('列'); axes[1,1].set_ylabel('行')
plt.colorbar(im6, ax=axes[1,1], label='Laplacian')

im7 = axes[1,2].imshow(final_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[1,2].set_title('频域反解结果 (有 DC 偏移)', fontsize=12)
axes[1,2].set_xlabel('列'); axes[1,2].set_ylabel('行')
plt.colorbar(im7, ax=axes[1,2], label='mGal')

im8 = axes[1,3].imshow(final_corrected_plot, cmap='RdBu', vmin=vmin_g, vmax=vmax_g)
axes[1,3].set_title('最终结果 (校正后)\n无缝拼接', fontsize=12)
axes[1,3].set_xlabel('列'); axes[1,3].set_ylabel('行')
plt.colorbar(im8, ax=axes[1,3], label='mGal')

plt.tight_layout()
plt.savefig('laplacian_merge_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("9. 结果对比")
print("=" * 70)
print(f"直接拼接 g₀:      min={np.nanmin(g0):.2f}, max={np.nanmax(g0):.2f}, mean={np.nanmean(g0):.2f}")
print(f"频域反解(校正前):  min={np.nanmin(final):.2f}, max={np.nanmax(final):.2f}, mean={np.nanmean(final):.2f}")
print(f"最终校正后:        min={np.nanmin(final_corrected):.2f}, max={np.nanmax(final_corrected):.2f}, mean={np.nanmean(final_corrected):.2f}")
print(f"Land 独有区域差异校正后: {np.nanmean(land_full[land_only_mask] - final_corrected[land_only_mask]):.6f} mGal")

# 打印关键数学公式
print("\n" + "=" * 70)
print("10. 关键数学公式回顾")
print("=" * 70)
print("离散 Laplacian (5点差分):")
print("  ∇²g_ij = (g_i+1,j + g_i-1,j - 2g_ij)/Δx² + (g_i,j+1 + g_i,j-1 - 2g_ij)/Δy²")
print("\n频域 Poisson 求解:")
print("  g = F⁻¹{ F{f} / H(k) }")
print("  H(k) = 2/Δx²·(cos(kx·Δx)-1) + 2/Δy²·(cos(ky·Δy)-1)")
print("\n优先级掩码合并:")
print("  L0 = L_sea if mask=1, L_land if mask=0, NaN elsewhere")
print("\n双三次样条插值:")
print("  L_int(x,y) = ΣᵢΣⱼ a_ij xⁱ yʲ,  ∀(x,y)∈Ω_gap")
