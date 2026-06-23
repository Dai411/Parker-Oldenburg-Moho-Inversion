import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os

# ===== 设置路径 =====
data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly/StitchGrids'
merged_file = os.path.join(data_dir, 'BouguerMerged.asc')
output_fig = os.path.join(data_dir, 'merge_validation.png')
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

# 读取数据
print("读取合并后的数据...")
header = read_asc_header(merged_file)
data = read_asc_grid(merged_file, header)

# 计算梯度
print("计算梯度...")
dy_grad = np.gradient(data, axis=0)  # 南北方向梯度
dx_grad = np.gradient(data, axis=1)  # 东西方向梯度
grad_magnitude = np.sqrt(dx_grad**2 + dy_grad**2)

# 识别尖刺（梯度超过 99.5% 分位数的点）
threshold = np.nanpercentile(grad_magnitude, 99.5)
spikes = (grad_magnitude > threshold) & (~np.isnan(grad_magnitude))
print(f"梯度阈值 (99.5%): {threshold:.4f} mGal/格")
print(f"尖刺点数: {np.sum(spikes)}")

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 布格异常全图
ax1 = axes[0, 0]
norm = TwoSlopeNorm(vmin=-188, vcenter=0, vmax=292)
im1 = ax1.imshow(data, cmap='RdBu_r', norm=norm, origin='upper')
ax1.set_title('Bouguer Anomaly (Merged)', fontsize=12)
ax1.set_xlabel('Column')
ax1.set_ylabel('Row')
plt.colorbar(im1, ax=ax1, label='mGal')

# 2. 梯度幅值图
ax2 = axes[0, 1]
im2 = ax2.imshow(grad_magnitude, cmap='hot', origin='upper', vmax=5)
ax2.set_title('Gradient Magnitude (log scale)', fontsize=12)
ax2.set_xlabel('Column')
ax2.set_ylabel('Row')
plt.colorbar(im2, ax=ax2, label='mGal/grid')
# 标记尖刺位置
spike_rows, spike_cols = np.where(spikes)
ax2.scatter(spike_cols, spike_rows, c='cyan', s=1, alpha=0.5, label=f'Spikes (n={np.sum(spikes)})')
ax2.legend(markerscale=10)

# 3. 尖刺位置分布
ax3 = axes[1, 0]
ax3.hist(grad_magnitude[~np.isnan(grad_magnitude)], bins=100, range=(0, 10))
ax3.axvline(threshold, color='red', linestyle='--', label=f'99.5% threshold = {threshold:.2f}')
ax3.set_title('Gradient Magnitude Distribution', fontsize=12)
ax3.set_xlabel('mGal/grid')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.set_yscale('log')

# 4. 沿某条剖面的梯度（穿过尖刺最多的区域）
ax4 = axes[1, 1]
# 找到尖刺最密集的行
if np.sum(spikes) > 0:
    spike_row_counts = np.sum(spikes, axis=1)
    row_with_most_spikes = np.argmax(spike_row_counts)
    
    # 绘制该行的布格异常
    ax4.plot(data[row_with_most_spikes, :], 'b-', alpha=0.7, label='Bouguer anomaly')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('mGal', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    # 添加梯度作为第二纵轴
    ax4_twin = ax4.twinx()
    ax4_twin.plot(grad_magnitude[row_with_most_spikes, :], 'r-', alpha=0.5, label='Gradient')
    ax4_twin.set_ylabel('mGal/grid', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    # 标记尖刺
    spike_cols_in_row = np.where(spikes[row_with_most_spikes, :])[0]
    ax4.scatter(spike_cols_in_row, data[row_with_most_spikes, spike_cols_in_row], 
                c='red', s=30, zorder=5, label=f'Spikes (row={row_with_most_spikes})')
    
    ax4.set_title(f'Profile at Row {row_with_most_spikes}', fontsize=12)
    ax4.legend(loc='upper left')
else:
    ax4.text(0.5, 0.5, 'No spikes detected', transform=ax4.transAxes, 
             ha='center', va='center', fontsize=14)
    ax4.set_title('Profile', fontsize=12)

plt.tight_layout()
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"\n✓ 图片已保存: {output_fig}")

# 打印尖刺位置信息
if np.sum(spikes) > 0:
    print(f"\n=== 尖刺详细信息 ===")
    # 列出前10个尖刺
    spike_indices = list(zip(spike_rows, spike_cols))[:10]
    print("前10个尖刺位置 (row, col):")
    for r, c in spike_indices:
        print(f"  ({r}, {c}): 异常值={data[r,c]:.2f} mGal, 梯度={grad_magnitude[r,c]:.2f}")
    
    # 检查尖刺是否集中在某条线（可能是拼接边界）
    spike_rows_unique = np.unique(spike_rows)
    spike_cols_unique = np.unique(spike_cols)
    print(f"\n尖刺分布: {len(spike_rows_unique)} 行 × {len(spike_cols_unique)} 列")
    
    if len(spike_rows_unique) < 10:
        print(f"  主要出现在行: {spike_rows_unique}")
    if len(spike_cols_unique) < 10:
        print(f"  主要出现在列: {spike_cols_unique}")

# 显示图片（如果是在交互环境）
plt.show()
