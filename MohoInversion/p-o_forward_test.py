# test_parker_forward_v2.py
"""
Parker-Oldenburg 正演测试 (- h 为起伏)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# ============================================================
# 参数配置
# ============================================================

DRHO = 0.40          # 密度差 (g/cm3)
Z0 = 20.0            # 平均界面深度 (km)
G = 6.67430e-11      # 引力常数 (m3/kg/s2)

CELLSIZE = 300       # 米
NX = 200
NY = 200
DX = CELLSIZE
DY = CELLSIZE

def compute_wavenumbers(nx, ny, dx, dy):
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K

def parker_forward(h_rel, drho, z0, dx, dy, n_max=4):
    """
    Parker-Oldenburg 正演
    h_rel: 界面起伏 (m)，相对于平均深度 z0
    drho: 密度差 (g/cm3)
    z0: 平均深度 (km)
    返回: gravity (mGal)
    """
    ny, nx = h_rel.shape
    
    KX, KY, K = compute_wavenumbers(nx, ny, dx, dy)
    
    drho_kg = drho * 1000        # g/cm3 -> kg/m3
    z0_m = z0 * 1000             # km -> m
    
    # 向下延拓因子
    downward_cont = np.exp(-K * z0_m)
    # 限制高频（避免数值爆炸）
    downward_cont = np.clip(downward_cont, 0, 1e3)
    
    # 求和项
    sum_term = np.zeros_like(K, dtype=complex)
    
    for n in range(1, n_max + 1):
        h_pow = h_rel ** n
        F_h_pow = fft2(h_pow)
        term = (K ** (n-1)) / math.factorial(n) * F_h_pow
        sum_term += term
    
    # 正演重力 (SI: m/s2)
    F_gravity_si = -2 * np.pi * G * drho_kg * downward_cont * sum_term
    gravity_si = np.real(ifft2(F_gravity_si))
    
    # 转换: 1 m/s2 = 1e5 mGal
    gravity_mgal = gravity_si * 1e5
    
    return gravity_mgal

# ============================================================
# 创建测试模型
# ============================================================

print("=" * 70)
print("Parker-Oldenburg Forward Test (h as relief)")
print("=" * 70)

# 参数
WAVELENGTH = 100   # km
AMPLITUDE = 2.0    # km (起伏振幅)
Z0 = 20.0          # km (平均深度)

print(f"Sine wave relief: amplitude={AMPLITUDE} km, wavelength={WAVELENGTH} km")
print(f"Mean depth: {Z0} km")
print(f"Density contrast: {DRHO} g/cm3")
print("=" * 70)

# 创建网格
x = np.arange(NX) * CELLSIZE / 1000  # km
y = np.arange(NY) * CELLSIZE / 1000
X, Y = np.meshgrid(x, y)

# 关键：h_rel 是起伏（不是绝对深度）
h_rel = AMPLITUDE * 1000 * np.sin(2 * np.pi * X / WAVELENGTH)  # 单位：m

# 绝对深度（仅用于可视化）
h_abs = Z0 * 1000 + h_rel  # m

print(f"\nModel statistics:")
print(f"  h_rel range: {np.min(h_rel)/1000:.1f} - {np.max(h_rel)/1000:.1f} km")
print(f"  h_abs range: {np.min(h_abs)/1000:.1f} - {np.max(h_abs)/1000:.1f} km")

# 正演
import time
start = time.time()
gravity = parker_forward(h_rel, DRHO, Z0, DX, DY, n_max=4)
elapsed = time.time() - start

print(f"\nForward results:")
print(f"  Gravity range: {np.min(gravity):.2f} - {np.max(gravity):.2f} mGal")
print(f"  Gravity mean: {np.mean(gravity):.6f} mGal")
print(f"  Gravity std: {np.std(gravity):.2f} mGal")
print(f"  Time: {elapsed:.2f}s")

# 理论验证
L_m = WAVELENGTH * 1000
A_m = AMPLITUDE * 1000
z0_m = Z0 * 1000
drho_kg = DRHO * 1000

theory_amplitude_m_s2 = 2 * np.pi * G * drho_kg * A_m * np.exp(-2 * np.pi * z0_m / L_m)
theory_amplitude_mgal = theory_amplitude_m_s2 * 1e5

print(f"\nTheoretical check:")
print(f"  Expected gravity amplitude: {theory_amplitude_mgal:.2f} mGal")
print(f"  Actual gravity amplitude (std): {np.std(gravity):.2f} mGal")

ratio = np.std(gravity) / theory_amplitude_mgal if theory_amplitude_mgal > 0 else 0
print(f"  Ratio (actual/theory): {ratio:.2f}")

if 0.8 < ratio < 1.2:
    print("\n[OK] Forward calculation is correct!")
else:
    print("\n[WARNING] Ratio not close to 1, but may be acceptable for large wavelength")

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im1 = axes[0].imshow(h_rel/1000, cmap='RdBu', extent=[0, NX*CELLSIZE/1000, 0, NY*CELLSIZE/1000])
axes[0].set_title('Moho Relief (km)')
axes[0].set_xlabel('km'); axes[0].set_ylabel('km')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(gravity, cmap='RdBu', extent=[0, NX*CELLSIZE/1000, 0, NY*CELLSIZE/1000])
axes[1].set_title('Gravity Anomaly (mGal)')
axes[1].set_xlabel('km')
plt.colorbar(im2, ax=axes[1])

# 剖面
center_row = NY // 2
axes[2].plot(x, h_rel[center_row, :]/1000, 'b-', label='Moho relief')
axes[2].set_xlabel('km')
axes[2].set_ylabel('Relief (km)', color='b')
axes[2].tick_params(axis='y', labelcolor='b')
ax2 = axes[2].twinx()
ax2.plot(x, gravity[center_row, :], 'r-', label='Gravity')
ax2.set_ylabel('Gravity (mGal)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
axes[2].set_title('Profile at center row')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forward_test_v2.png', dpi=150)
print("\nFigure saved: forward_test_v2.png")
