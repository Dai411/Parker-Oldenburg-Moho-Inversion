# initial_moho.py
"""
创建简化的初始 Moho 模型 (最终修正版)
- 网格规格与 BouguerModelled.asc 相同
- 基于 IODP 402 U1615 站位 (40°11'N, 12°38'E) 设置中心深度
- 第勒尼安海中心 ~9 km，边缘渐深至 ~30 km
- 陆地部分加厚，平滑过渡
"""

import numpy as np
import os
from scipy.ndimage import gaussian_filter

# ============================================================
# 参数配置
# ============================================================

# U1614 站位投影坐标 (IODP Expedition 402)
# 经纬度: 40°11'N, 12°38'E
# 墨卡托投影 (标准纬线 40°N)
TYRRHENIAN_CENTER_X = 574000   # 米 (574 km)
TYRRHENIAN_CENTER_Y = 4250000  # 米 (4250 km)

# 中心最小 Moho 深度 (km) - U1614 处地幔剥露
MIN_DEPTH = 9

# 边缘最大 Moho 深度 (km)
MAX_DEPTH = 30

# 高斯过渡半径 (km) - 根据 Vavilov 盆地尺度
TRANSITION_RADIUS = 80   # km

# 陆地加厚 (km)
LAND_THICKEN = 10

# 高斯平滑 sigma (像素，用于平滑陆地边界)
SMOOTH_SIGMA = 5

# 强制中心区域深度 (避免被平滑拉高)
FORCE_CENTER_DEPTH = True
CENTER_FORCE_X1 = 568000   # 米 (中心区域 X 范围)
CENTER_FORCE_X2 = 580000   # 米
CENTER_FORCE_Y1 = 4240000  # 米 (中心区域 Y 范围)
CENTER_FORCE_Y2 = 4260000  # 米

# ============================================================
# 文件路径
# ============================================================

data_dir = 'C:/Users/yangln/Desktop/Postdoc/CNR_Italy/Maps/BouguerAnomaly'
model_file = os.path.join(data_dir, 'BouguerModelled.asc')
output_dir = os.path.join(data_dir, 'MohoInversion')
output_file = os.path.join(output_dir, 'InitialMoho.asc')

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

# ============================================================
# 主程序
# ============================================================

print("=" * 60)
print("创建简化初始 Moho 模型 (最终修正版)")
print("=" * 60)
print(f"  U1614 站位中心: ({TYRRHENIAN_CENTER_X/1000:.0f}, {TYRRHENIAN_CENTER_Y/1000:.0f}) km")
print(f"  中心深度: {MIN_DEPTH} km (基于 IODP 402 地幔剥露)")
print(f"  边缘深度: {MAX_DEPTH} km")
print(f"  过渡半径: {TRANSITION_RADIUS} km")
print(f"  陆地加厚: {LAND_THICKEN} km")
print(f"  强制中心深度: {FORCE_CENTER_DEPTH}")

# 1. 读取网格参数
print("\n1. 读取网格参数...")
header = read_asc_header(model_file)
nx = header['ncols']
ny = header['nrows']
xll = header['xllcorner']
yll = header['yllcorner']
cellsize = header['cellsize']

print(f"   网格尺寸: {ny} x {nx}")
print(f"   范围: X [{xll:.0f}, {xll + nx*cellsize:.0f}] m")
print(f"   范围: Y [{yll:.0f}, {yll + ny*cellsize:.0f}] m")

# 转换为 km 显示
xll_km = xll / 1000
yll_km = yll / 1000
xmax_km = (xll + nx*cellsize) / 1000
ymax_km = (yll + ny*cellsize) / 1000
print(f"   范围 (km): X [{xll_km:.0f}, {xmax_km:.0f}], Y [{yll_km:.0f}, {ymax_km:.0f}]")

# 2. 创建网格坐标
print("\n2. 创建网格坐标...")
x = xll + (np.arange(nx) + 0.5) * cellsize
y = yll + (np.arange(ny) + 0.5) * cellsize
X, Y = np.meshgrid(x, y)

# 3. 计算到第勒尼安海中心的距离 (km)
print("\n3. 计算径向距离...")
dist_km = np.sqrt((X - TYRRHENIAN_CENTER_X)**2 + (Y - TYRRHENIAN_CENTER_Y)**2) / 1000

# 4. 创建径向 Moho 深度 (km)
print(f"\n4. 创建径向模型...")

# 高斯型过渡
moho_depth = MAX_DEPTH - (MAX_DEPTH - MIN_DEPTH) * np.exp(-dist_km**2 / (TRANSITION_RADIUS**2))

print(f"   径向模型中心深度: {np.min(moho_depth):.1f} km")
print(f"   径向模型边缘深度: {np.max(moho_depth):.1f} km")

# 5. 添加陆地加厚
print(f"\n5. 添加陆地加厚 ({LAND_THICKEN} km)...")

# 陆地掩码 - 基于网格实际范围
land_mask = np.zeros_like(moho_depth, dtype=bool)

# 意大利本土 (亚平宁山脉区域)
land_mask = land_mask | ((X > 550000) & (X < 750000) & (Y > 4200000) & (Y < 4283150))

# 撒丁岛
land_mask = land_mask | ((X > 680000) & (X < 760000) & (Y > 4300000) & (Y < 4450000))

# 科西嘉岛
land_mask = land_mask | ((X > 820000) & (X < 880000) & (Y > 4180000) & (Y < 4283150))

# 西西里岛
land_mask = land_mask | ((X > 520000) & (X < 680000) & (Y > 4080000) & (Y < 4250000))

print(f"   陆地掩码点数: {np.sum(land_mask)}")

# 应用陆地加厚
moho_depth[land_mask] += LAND_THICKEN

# 6. 平滑
print(f"\n6. 平滑过渡 (sigma={SMOOTH_SIGMA} 像素)...")
moho_depth = gaussian_filter(moho_depth, sigma=SMOOTH_SIGMA)

# 7. 强制中心深度 (避免被平滑拉高)
if FORCE_CENTER_DEPTH:
    print(f"\n7. 强制设置第勒尼安海中心深度...")
    center_mask = (X > CENTER_FORCE_X1) & (X < CENTER_FORCE_X2) & \
                  (Y > CENTER_FORCE_Y1) & (Y < CENTER_FORCE_Y2)
    moho_depth[center_mask] = MIN_DEPTH
    print(f"   中心区域: X [{CENTER_FORCE_X1/1000:.0f}, {CENTER_FORCE_X2/1000:.0f}] km, "
          f"Y [{CENTER_FORCE_Y1/1000:.0f}, {CENTER_FORCE_Y2/1000:.0f}] km")
    print(f"   强制深度: {MIN_DEPTH} km, 点数: {np.sum(center_mask)}")
    
    # 再轻度平滑 (sigma=2)
    moho_depth = gaussian_filter(moho_depth, sigma=2)
    
    # 再次强制 (确保不被平滑过度)
    moho_depth[center_mask] = MIN_DEPTH

# 8. 限制合理范围
moho_depth = np.clip(moho_depth, 8, 45)

# 9. 统计
print("\n8. 统计结果:")
print(f"   最小值: {np.min(moho_depth):.1f} km")
print(f"   最大值: {np.max(moho_depth):.1f} km")
print(f"   平均值: {np.mean(moho_depth):.1f} km")
print(f"   标准差: {np.std(moho_depth):.1f} km")

# 分区域统计
center_row = int((TYRRHENIAN_CENTER_Y - yll) / cellsize)
center_col = int((TYRRHENIAN_CENTER_X - xll) / cellsize)
if 0 <= center_row < ny and 0 <= center_col < nx:
    print(f"\n   U1614 站位中心: {moho_depth[center_row, center_col]:.1f} km")
else:
    print(f"\n   U1614 站位索引超出范围: row={center_row}, col={center_col}")
    print(f"   有效范围: row 0-{ny-1}, col 0-{nx-1}")

if np.sum(land_mask) > 0:
    print(f"   陆地区域平均: {np.mean(moho_depth[land_mask]):.1f} km")
else:
    print(f"   警告: 陆地掩码为空，请检查坐标范围")

# 检查中心区域
center_region_mask = (X > CENTER_FORCE_X1) & (X < CENTER_FORCE_X2) & \
                     (Y > CENTER_FORCE_Y1) & (Y < CENTER_FORCE_Y2)
if np.sum(center_region_mask) > 0:
    print(f"   中心区域平均: {np.mean(moho_depth[center_region_mask]):.1f} km")

# 10. 保存
print("\n9. 保存初始模型...")
write_asc_grid(output_file, moho_depth, header)
print(f"   ✓ 已保存: {output_file}")

print("\n" + "=" * 60)
print("✅ 初始 Moho 模型创建完成！")
print("=" * 60)
